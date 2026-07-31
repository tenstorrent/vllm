# TT Decode Reload Contract

Async device sampling deliberately lets vLLM's host token state trail the TT
device by one decode step. Consequently, only the vLLM runner can decide
whether host tensors are authoritative. A contract-aware tt-metal generator
(`decode_input_update_contract >= 1`) receives four independent boolean
commands on every decode:

| Command | Effect |
| --- | --- |
| `reload_inputs` | Copy all forward inputs: token, position, RoPE inputs, and page tables. |
| `reload_page_table` | Copy only page-table inputs. Ignored when `reload_inputs` is true. |
| `reload_sampling_params` | Upload temperature, top-k/top-p, penalties, seeds, and logprob configuration. |
| `reset_sampling_state` | Rebuild mutable penalty/RNG state for the current layout. |

`decode_layout_changed` is an internal vLLM lifecycle signal: for an explicit
contract adapter, the planner translates it into the four commands without
forwarding the signal itself. For a legacy adapter, vLLM translates it to the
old `reset_batch` keyword on device-sampling calls. `slot_remap` remains data:
it is composed by vLLM until a device-sampling submission consumes it, then
tt-metal applies it once before sampling state is reset or advanced.

## Mode definitions

- **Host sampling**: tt-metal returns logits and vLLM selects the token. Host
  token and position tensors are authoritative, so every decode performs a
  full input reload.
- **Device sampling**: tt-metal selects the token. A supporting model writes
  that token directly into the persistent input buffer used by the next decode
  and advances its persistent position in the forward trace.
- **Transition decode**: the first decode, the first decode after prefill, a
  batch-layout or sampling-mode change, or a resume. Host state is
  authoritative again; pending work drains before a full reload and any
  required sampling-state reset.
- **Steady device decode**: the request layout and sampling mode are unchanged
  after a valid device-sampling decode. Token and position remain
  device-resident, so no full reload occurs.
- **Page-table-only refresh**: a steady device decode whose KV block mapping
  changed. Only page-table trace inputs are copied; token, position, and RoPE
  state must remain untouched.

## Transition table

| Transition | Inputs | Page only | Sampling params | Sampling state |
| --- | ---: | ---: | ---: | ---: |
| First decode or prefill → decode | reload | no | reload on device | reset on device |
| Batch add/remove/reuse/condense or resume | reload | no | reload on device | reset on device |
| Host → device sampling | reload | no | reload | reset |
| Steady host sampling | reload every step | no | n/a | n/a |
| Steady device sampling | keep resident | only if changed | keep | keep |
| Decode tracing disabled | reload every step | no | on transition | on transition |
| Model without `supports_async_decode` | reload every step | no | on transition | on transition |

Any transition requiring a full input or sampling-state update drains pending
decode work first. Page-table-only refresh is overlap-safe because page tables
are scheduler-authoritative even while host token/position tensors are stale.

## Correctness argument

The device invariant immediately after a device-sampling decode step `k` is:
the persistent token slot contains sampled token `t_k`, and the persistent
position is the position at which `t_k` must be consumed by step `k+1`.

The base case is a full reload. vLLM first finalizes the prior non-steady step,
filters finished, resumed, and replaced request identities, updates continuing
host state (including a live request temporarily unscheduled this step), and
copies authoritative token/position/layout tensors.

For the induction step, a steady device decode issues no full or sampling-state
reload. The trace therefore consumes the device-resident `t_k` and position,
writes exactly one KV position, advances position exactly once, and sampling
writes `t_(k+1)` back to the persistent token slot. A page-table-only copy can
change KV address mapping but cannot overwrite token or position state.
Readback is observational and may complete later.

Host sampling always performs a full reload from the accepted host token.
Layout, resume, prefill, and sampling-mode transitions break the steady
invariant and therefore drain and re-establish the base case.

A completed async result is applied only to the captured request object when it
is still live and was not finished/resumed. Reused request IDs fail the
object-identity check, and their cached runner-output rows are replaced with an
empty token list before scheduler update. vLLM's scheduler independently
ignores outputs for requests that no longer exist, so cancelled speculative
work cannot append runner state or emit an extra client token.

## Requirements for `supports_async_decode`

For a version-1 adapter, set
`model_capabilities["supports_async_decode"] = True` only when the adapter
satisfies every requirement below:

1. **Split submission and readback**: `decode_forward(...,
   read_from_device=False)` submits decode without synchronizing the result, and
   `read_decode_output(..., async_read=True)` can read that exact submission
   later. Readback must be observational: it cannot advance position, sample
   another token, or otherwise mutate decode state.
2. **Persistent token feedback**: device sampling writes the selected token into
   the same persistent token buffer consumed by the next decode. Writing only
   to a separate output tensor is insufficient.
3. **Single position advance**: each successful decode forward advances the
   persistent position exactly once. Sampling and readback must not advance it.
   After step `k`, the resident token and position must describe the input to
   step `k+1`.
4. **Independent page-table refresh**: the model can copy changed page-table
   inputs without copying or rebinding token, position, or RoPE inputs.
5. **Exact command handling**: the adapter honors `reload_inputs`,
   `reload_page_table`, `reload_sampling_params`, and
   `reset_sampling_state` independently. It must not add model-local mode or
   tensor comparisons that turn a page-table-only update into a full reload.
6. **Sampling-state ordering**: slot remaps are applied before parameter/state
   reset; RNG and penalty state are reset only when requested; seed advancement
   happens exactly once per sampled token.
7. **Stable-buffer lifetime**: persistent decode and sampling buffers remain
   valid until the submitted step is read back and until the next command
   explicitly replaces their contents.

The capability is fail-closed. If any requirement is not met, leave
`supports_async_decode` absent or `False`. vLLM disables async scheduling for
that model and requests a full forward-input reload on every decode, while
still using the negotiated generator interface. A legacy adapter may already
advertise `supports_async_decode`; vLLM preserves that adapter's existing
reload and overlap behavior, but warns that correctness is not guaranteed
until the adapter also implements and advertises contract version 1.

## Contract negotiation

Model adapters opt in by setting `decode_input_update_contract = 1`. vLLM sends
the four explicit commands only to adapters advertising version 1 or newer.
Adapters without the attribute, or with version 0, receive the legacy
`reset_batch` keyword on device-sampling calls and never receive unknown
command keywords. Their reload and overlap behavior remains unchanged from the
pre-contract path, including any model-local heuristics. vLLM logs a warning
because those heuristics may observe stale host state under async decode and
cannot provide the version-1 correctness guarantees. This compatibility path
allows the vLLM change to land before individual tt-metal adapters are
refactored.

| vLLM | tt-metal adapter | Result |
| --- | --- | --- |
| Old | Legacy / version 0 | Supported: existing behavior |
| New | Legacy / version 0 | Compatibility path: legacy behavior with a warning |
| New | Version 1+ | Supported: explicit commands and eligible resident overlap |
| Old | Strict version 1 | Unsupported: old vLLM omits the required commands |

The supported rollout order is therefore vLLM first, followed by tt-metal
adapter migrations. Advertising version 1 before implementing every command
is an adapter bug and should fail loudly rather than silently falling back.
Versions greater than 1 must remain backward-compatible supersets of version 1;
a breaking interface requires a distinct negotiation key or supported range.

`model_capabilities["supports_async_decode"]` remains independent of contract
versioning. It controls async scheduling and certifies that sampled-token
feedback can remain device-resident between decode steps. A version-1 model
without that capability receives a conservative full-input reload command on
every decode.

The refactored tt-metal implementation includes the unconditional decode-only
seed initialization also addressed by
[tt-metal#51556](https://github.com/tenstorrent/tt-metal/pull/51556):
`reset_sampling_state=True` forces seed initialization for first-decode and
layout transitions even when both the requested and cached seed are `None`.
The implementation does not depend on that PR.
