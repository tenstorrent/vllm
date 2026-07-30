# TT Decode Reload Contract

Async device sampling deliberately lets vLLM's host token state trail the TT
device by one decode step. Consequently, only the vLLM runner can decide
whether host tensors are authoritative. The paired tt-metal generator receives
four independent boolean commands on every decode:

| Command | Effect |
| --- | --- |
| `reload_inputs` | Copy all forward inputs: token, position, RoPE inputs, and page tables. |
| `reload_page_table` | Copy only page-table inputs. Ignored when `reload_inputs` is true. |
| `reload_sampling_params` | Upload temperature, top-k/top-p, penalties, seeds, and logprob configuration. |
| `reset_sampling_state` | Rebuild mutable penalty/RNG state for the current layout. |

`reset_batch` no longer decides reload behavior in the vLLM path. `slot_remap`
remains data: it is composed by vLLM until a device-sampling submission
consumes it, then tt-metal applies it once before sampling state is reset or
advanced.

## Transition table

| Transition | Inputs | Page only | Sampling params | Sampling state |
| --- | ---: | ---: | ---: | ---: |
| First decode or prefill → decode | reload | no | reload on device | reset on device |
| Batch add/remove/reuse/condense or resume | reload | no | reload on device | reset on device |
| Host → device sampling | reload | no | reload | reset |
| Steady host sampling | reload every step | no | n/a | n/a |
| Steady device sampling | keep resident | only if changed | keep | keep |
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

## Deployment coupling

The vLLM and tt-metal changes are one required contract and should be deployed
as a pinned pair. vLLM always sends all four commands; there is no per-model
contract-version negotiation or fallback to an older tt-metal generator.
`model_capabilities["supports_async_decode"]` remains the sole per-model gate.
It both controls async scheduling and certifies that sampled-token feedback can
remain device-resident between decode steps. Models without it still receive
the explicit four-command contract, but vLLM conservatively requests a full
forward-input reload on every step.
