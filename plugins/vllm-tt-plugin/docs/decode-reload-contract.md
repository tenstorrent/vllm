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
for a version-1 adapter it is composed by vLLM until the next accepted decode
submission in either sampling mode. tt-metal applies it once before the decode
reads any persistent per-slot state. Dormant state that the decode cannot read,
such as a device sampler during host sampling, may be remapped immediately
after successful submission; applying its non-idempotent remap before a call
that can fail would corrupt a retry. Persistent state includes model-owned
recurrent or convolution state as well as device sampling state. Version-0
adapters retain the legacy device-sampling-only delivery and consumption
behavior.

Two superficially reasonable implementations are incorrect:

1. **Deliver the remap only for device sampling.** A host-sampling decode can
   still change the vLLM slot layout. Withholding `slot_remap` leaves
   model-owned slot state, such as recurrent/conv buffers or cached RoPE
   deltas, attached to the prior request.
2. **Apply a delivered remap only inside the active device-sampling call.** A
   model may retain slot-indexed sampler state even while that decode samples
   on the host. Skipping the dormant sampler leaves seed/RNG/penalty state in
   the old slots, so a later return to device sampling resumes the wrong
   request's state.

The rule is therefore delivery on every version-1 decode and exactly-once
application by every slot-owning subsystem. An authoritative rebuild may
replace a subsystem's remap, but merely not using that subsystem this step may
not. Version-0 adapters retain their historical remap behavior unchanged.

A remap cannot express slot reuse. When a new request takes a slot there is no
predecessor state to gather from, so vLLM keeps that slot's entry the identity
and signals the event through `decode_layout_changed`. Sampling state is then
invalidated by `reset_sampling_state`. Model-owned per-slot state — recurrent,
convolution, cached RoPE deltas — has no equivalent command in version 1, so an
adapter that keeps such state must rebuild the reused slot from the reloaded
forward inputs. A future version should carry the reused slots explicitly rather
than leave that inference to the adapter.

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
filters finished and resumed requests, updates continuing host state (including
a live request temporarily unscheduled this step), and copies authoritative
token/position/layout tensors.

For the induction step, a steady device decode issues no full or sampling-state
reload. The trace therefore consumes the device-resident `t_k` and position,
writes exactly one KV position, advances position exactly once, and sampling
writes `t_(k+1)` back to the persistent token slot. A page-table-only copy can
change KV address mapping but cannot overwrite token or position state.
Readback is observational and may complete later.

Host sampling always performs a full reload from the accepted host token.
Layout, resume, prefill, and sampling-mode transitions break the steady
invariant and therefore drain and re-establish the base case.

A completed async result is applied only when its captured internal request ID
is still live and was not finished/resumed. vLLM assigns a fresh internal ID to
every accepted request, so aborting and resubmitting the same external ID cannot
attach an old result to the new request. Cached runner-output rows for explicitly
finished/resumed requests are replaced with an empty token list before scheduler
update. vLLM's scheduler independently ignores outputs for requests that no
longer exist, so cancelled speculative work cannot append runner state or emit
an extra client token.

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
7. **Host input authority**: the `tokens` and `start_pos` arguments are
   authoritative only when `reload_inputs` is true. When it is false they are
   deliberately one step behind, and the adapter must derive nothing from them —
   not forward inputs, and not sampling state. Deriving an RNG counter from
   `start_pos` on a steady step makes the sampled stream depend on when the
   asynchronous readback landed, so the same request and seed stop reproducing.
   An adapter that ties per-token seeds to the absolute decode position must do
   so only on a reloading step and advance its own resident counter otherwise.
8. **Complete slot remapping**: on every version-1 decode, `slot_remap` applies
   to all persistent state indexed by the vLLM batch slot, even when that step
   samples on the host. This includes model-internal recurrent/convolution
   state and dormant device-sampler state; a full forward-input reload does
   not implicitly repair either one.

   One exemption: state that is not addressable by vLLM slot cannot be
   remapped, only reset. Unseeded on-device RNG is the known case — its state
   is a per-core hardware PRNG register that no operation can move between
   cores, and the adapter is expected to leave it in place. An adapter must
   declare any such state rather than silently skip a remap, and vLLM's commit
   of the mapping is valid for it by exemption.
9. **Stable-buffer lifetime**: persistent decode and sampling buffers remain
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
command keywords. They also retain device-sampling-only `slot_remap` delivery.
Their reload and overlap behavior remains unchanged from the pre-contract path,
including any model-local heuristics. vLLM logs a warning because those
heuristics may observe stale host state under async decode and cannot provide
the version-1 correctness guarantees. This compatibility path allows the vLLM
change to land before individual tt-metal adapters are refactored.

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
