# TT Decode Reload Contract

Async device sampling deliberately lets vLLM's host token state trail the TT
device by one decode step. Consequently, only the vLLM runner can decide
whether host tensors are authoritative. A contract-aware tt-metal generator
(`decode_input_update_contract >= 1`) receives four independent boolean
commands on every decode:

## Commands

| Command | Effect |
| --- | --- |
| `reload_inputs` | Copy all forward inputs: token, position, RoPE inputs, and page tables. Subsumes `reload_page_table`. |
| `reload_page_table` | Copy only page-table inputs. vLLM never sets it together with `reload_inputs`. |
| `reload_sampling_params` | Upload temperature, top-k/top-p, penalties, seeds, and logprob configuration. |
| `reset_sampling_state` | Rebuild mutable penalty/RNG state for the current layout. |

The first two are not independent: `reload_inputs` already copies page tables,
so an adapter that treats them as two disjoint switches never copies the page
table on a transition step, and the device then addresses the previous batch's
KV blocks with no error. vLLM asserts the pair is never both true, so the only
legal readings are "everything", "page tables only", and "nothing".

`reset_sampling_state` implies `reload_inputs`, also asserted on the vLLM side.
That is what makes requirement 7 satisfiable: an adapter may align seed counters
from `start_pos` on a state reset precisely because the same step restages it.

### Command defaults

vLLM sends all four explicitly on every version-1 decode, so an adapter needs no
defaults. Where an adapter does default one, the only permitted value is the
host-authoritative one: `reload_inputs=True` and the other three `False`. Any
other default silently reuses device state a caller did not ask to keep. An
adapter that absorbs unrecognised keywords through `**kwargs` should still reject
the pre-contract `reset_batch` by name, so a vLLM too old to send the commands
fails with a message instead of being interpreted as a full reload.

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

State the forward *does* read has to be remapped before it, so those remaps are
necessarily applied on a call that can fail. A raised decode therefore leaves the
model-owned half applied while vLLM's own commit is still pending. vLLM does not
retry a failed decode submission: under gathered DP the exception surfaces through
the gather future and is fatal to the engine, and on the single-engine path the
pending remap is retired only on success so the next step re-establishes host
authority. An adapter must not treat a raised `decode_forward` as resumable.

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
invalidated by `reset_sampling_state`. Model-owned per-slot state (recurrent,
convolution, cached RoPE deltas) has no equivalent command in version 1, so an
adapter that keeps such state must rebuild the reused slot from the reloaded
forward inputs. A future version should carry the reused slots explicitly rather
than leave that inference to the adapter.

`slot_remap` is in **global** slot indices. Under gathered multi-process DP vLLM
offsets each rank's local `[0, max_num_seqs)` mapping by `rank * max_num_seqs`, so
a generator holding per-rank state must rebase its own slice before indexing that
state, and must check that the width it received matches the stride it assumed. A
mapping that moves a request between ranks is an error, not a move. Non-DP and
lane deployments send one rank's worth, where local and global coincide.

The remap and the layout signal are retired together, at the boundary where a
decode submission is accepted. Retiring one without the other would leave a
remap pending while the rebuild that repairs its destructive effect on a vacated
slot has already been consumed.

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
- **Chunked-prefill continuation**: a further prompt chunk of a request that is
  already resident. It is prefill, but it is neither a new nor a resumed
  request, so batch membership is unchanged and the layout prediction alone
  accepts it. vLLM classifies it from the scheduler output's context phase, not
  from a scheduled-token count: the last chunk of a prompt can be a single
  token, and a decode row may legitimately be scheduled several.
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
| Chunked-prefill continuation of a resident request | reload | no | reload on device | reset on device |
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

"Prefill" there includes an intermediate or final chunk of an already-resident
request. Such a step leaves batch membership intact, so the membership-based
layout prediction accepts it and the context-phase test is the only thing that
rejects it. Missing it would submit a prefill over a pending decode readback
whose token has not yet reached host state.

A completed async result is applied only to requests that are still live and
were neither finished nor resumed since the step was submitted. Request ids are
client-supplied and may be reused, so an abort followed by a resubmit under the
same id appears in one scheduler output as both a finished id and a new request.
The finished id is what marks the step's result invalid, which is why rejection
is keyed on the scheduler's lifecycle events rather than on identity of the
cached request state. Runner-output rows for those ids are replaced with an empty
token list before the scheduler update, so a cancelled step can neither append
runner state nor emit an extra client token.

## Requirements for `decode_input_update_contract = 1`

Requirements keep their original numbers so existing references stay valid; the
two headings say which decision each number belongs to.

Every version-1 adapter must satisfy requirements 5, 6, 8 and 9, whatever it
advertises for `supports_async_decode`. vLLM sends the four commands and delivers
`slot_remap` to every version-1 adapter, in both sampling modes, so these are
obligations of the version, not of the capability.

<ol start="5">
<li>

**Exact command handling**: the adapter honors `reload_inputs`,
`reload_page_table`, `reload_sampling_params`, and `reset_sampling_state`. It
must not add model-local mode or tensor comparisons that turn a page-table-only
update into a full reload. An adapter that cannot execute part of the contract
must reject the combination loudly, naming the offending command (see "Partial
adapters" below); quietly ignoring a command is what this contract exists to
forbid.

</li>
<li>

**Sampling-state ordering**: slot remaps are applied before parameter/state
reset; RNG and penalty state are reset only when requested; seed advancement
happens exactly once per sampled token.

</li>
</ol>

<ol start="8">
<li>

**Complete slot remapping**: on every version-1 decode, `slot_remap` applies to
all persistent state indexed by the vLLM batch slot, even when that step samples
on the host. This includes model-internal recurrent/convolution state and dormant
device-sampler state; a full forward-input reload does not implicitly repair
either one.

One exemption: state that is not addressable by vLLM slot cannot be remapped,
only reset. Unseeded on-device RNG is the known case: its state is a per-core
hardware PRNG register that no operation can move between cores, and the adapter
is expected to leave it in place. An adapter must declare any such state rather
than silently skip a remap, and vLLM's commit of the mapping is valid for it by
exemption. "Declare" means naming the state and the reason in the adapter's own
documentation, next to the code that skips it; the exemption covers only state
with no move primitive, never state the adapter finds inconvenient to move.

</li>
<li>

**Stable-buffer lifetime**: persistent decode and sampling buffers remain valid
until the submitted step is read back and until the next command explicitly
replaces their contents.

</li>
</ol>

## Additional requirements for `supports_async_decode`

Set `model_capabilities["supports_async_decode"] = True` only when the adapter
also satisfies requirements 1 to 4 and 7. These are what let vLLM submit a decode
whose host token and position state is deliberately one step behind.

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

<ol start="7">
<li>

**Host input authority**: the `tokens` and `start_pos` arguments are
authoritative only when `reload_inputs` is true. When it is false they are
deliberately one step behind, and the adapter must derive nothing from them: not
forward inputs, and not sampling state. Deriving an RNG counter from `start_pos`
on a steady step makes the sampled stream depend on when the asynchronous
readback landed, so the same request and seed stop reproducing. An adapter that
ties per-token seeds to the absolute decode position must do so only on a
reloading step and advance its own resident counter otherwise. vLLM guarantees
`reset_sampling_state` implies `reload_inputs`, so a state reset is always a step
on which those arguments may be trusted.

</li>
</ol>

The capability is fail-closed. If any requirement is not met, leave
`supports_async_decode` absent or `False`. vLLM disables async scheduling for
that model and requests a full forward-input reload on every decode, while
still using the negotiated generator interface. A legacy adapter may already
advertise `supports_async_decode`; vLLM preserves that adapter's existing
reload and overlap behavior, but logs that correctness is not guaranteed
until the adapter also implements and advertises contract version 1.

### Partial adapters

An adapter may advertise version 1 while being structurally unable to execute a
command combination, provided both hold:

- it leaves `supports_async_decode` absent or `False`, which is what stops vLLM
  from ever planning the combination it cannot execute, and
- it rejects that combination with an error naming the command, rather than
  degrading silently.

Requiring a full input reload is the common case: an adapter that rebuilds all
host inputs every decode cannot honor `reload_inputs=False`. Such an adapter is
conformant, not buggy. The unconditional part of the contract, requirements 5, 6,
8 and 9, still applies to it in full, which is what makes `slot_remap` delivery
useful to a host-sampling model that owns per-slot recurrent state.

Because the two keys interact this way, an adapter must not flip
`supports_async_decode` on without first removing its rejections. The rejection
comment should say so at the site.

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

Overlap eligibility under gathered multi-process DP (`data_parallel_size > 1`)
therefore requires version 1. Submit-before-finalize is authorised from a plan a
version-0 adapter is never sent, so those adapters keep the finalize-before-submit
order and forgo the overlap rather than reload from host tensors that are
deliberately one step behind the device. Single-process lane DP is unaffected: it
reports `data_parallel_size == 1` and its overlap behavior predates the contract.

### Where the version is declared, and what that arms

**The marker belongs on the generator that implements the commands, not on each
leaf adapter, and is therefore inherited by every subclass of that generator.**
That is the decided placement. Its direct consequence: when a shared generator
declares version 1, gathered-DP decode overlap becomes eligible for every adapter
built on it that also advertises `supports_async_decode`, without any further
per-adapter decision. Arming is not staged one adapter at a time.

This is the right trade because the commands are implemented once, in the shared
generator, and a subclass that inherits that implementation genuinely implements
the contract. The alternative, a per-leaf marker, is fail-open in the other and
worse direction: a leaf that forgets the marker silently drops to the legacy call
shape and its reload decisions revert to model-local heuristics with no error.

Two obligations follow for tt-metal:

- Moving the marker up onto a generator is a change to the arming set. Audit
  every existing subclass against the requirements above before doing it, not
  only the adapter that motivated the move.
- A subclass that overrides `decode_forward` no longer inherits the
  implementation the marker attests to. Re-declaring the marker is not what makes
  it conformant: the override must itself execute every command, or the subclass
  must set `decode_input_update_contract = 0` and take the legacy path.

| vLLM | tt-metal adapter | Result |
| --- | --- | --- |
| Old | Legacy / version 0 | Supported: existing behavior |
| New | Legacy / version 0 | Compatibility path: legacy behavior with a warning |
| New | Version 1+ | Supported: explicit commands and eligible resident overlap |
| Old | Strict version 1 | Unsupported: old vLLM omits the required commands |

The supported rollout order is therefore vLLM first, followed by tt-metal
adapter migrations. Advertising version 1 while quietly ignoring a command is an
adapter bug; rejecting a combination it cannot execute is not (see "Partial
adapters").

Versions greater than 1 must remain backward-compatible supersets of version 1;
a breaking interface requires a distinct negotiation key or supported range.
There is no handshake in the other direction: an adapter cannot read the
plugin's version, and a version-1 plugin sends exactly the four commands above
to anything advertising 1 or newer. So any command a later version adds must be
keyword-only with a default that reproduces version-1 behavior, or a version-2
adapter breaks against a version-1 plugin. Making a new command required is a
breaking interface and needs a new negotiation key.

`model_capabilities["supports_async_decode"]` is a separate key with a separate
question: it controls async scheduling and certifies that sampled-token feedback
can remain device-resident between decode steps. A version-1 model without that
capability receives a conservative full-input reload command on every decode.
The keys are not fully orthogonal in practice, because leaving the capability off
is what makes a partial adapter safe; see "Partial adapters".

Not every tt-metal generator is migrated. Deliberate version-0 holdouts are
listed in tt-metal's `models/common/sampling/README.md`, which is authoritative
for which generator stacks still take the legacy path and why.

The refactored tt-metal implementation includes the unconditional decode-only
seed initialization also addressed by
[tt-metal#51556](https://github.com/tenstorrent/tt-metal/pull/51556):
`reset_sampling_state=True` forces seed initialization for first-decode and
layout transitions even when both the requested and cached seed are `None`.
The implementation does not depend on that PR.
