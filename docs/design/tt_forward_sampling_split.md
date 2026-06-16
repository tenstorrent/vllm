# TT Forward and Sampling Split

This note summarizes the TT plugin refactor that separates model forward
execution from token sampling. The goal is to match the normal vLLM V1 control
flow:

```text
schedule -> execute_model -> get_grammar_bitmask -> sample_tokens -> scheduler update
```

Before this change, TT sampled inside the model execution path. That forced
structured-output grammar data to be available before TT execution could start.
After this change, TT forward produces a `TTForwardOutput` payload, and sampling
consumes that payload later through `sample_tokens(grammar_output)` or the
gathered-DP equivalent.

## Motivation

The immediate motivation is structured outputs. vLLM computes grammar masks from
scheduler state, and those masks naturally belong at sampling time: they
constrain which next tokens may be selected from logits. The normal V1 GPU path
therefore launches forward first, computes `GrammarOutput` while forward is in
flight, and passes the result into `sample_tokens()`.

TT used to require grammar data before `execute_model()` because TT sampled
inside that call. That made structured-output handling awkward in three ways:

- Forward execution had to wait for grammar data even though the model does not
  need grammar to compute logits.
- Async batch-queue paths needed TT-specific deferral logic when grammar tokens
  were pending.
- DP had to carry structured-output information through gathered model inputs,
  even though the actual constraint should be applied at sampling time.

The split removes that mismatch. Forward execution now produces a sampling
payload, and grammar constraints are consumed by the sampling phase.

## Shared Concepts

`TTModelRunner.execute_model()` is now forward-only for non-DP execution. It
updates runner state, builds `TTModelInput`, submits prefill or decode to the TT
model, stores a `TTForwardOutput`, and returns `None` when sampling is required.

`TTModelRunner.sample_tokens(grammar_output)` reads the stored forward payload,
reorders the grammar bitmask into TT batch order if needed, applies it to host
logits, runs the host sampler, and then applies sampled tokens to runner state.
When device sampling is enabled and the model exposes a deferred sampling entry
point (`sample_decode_on_device` / `sample_prefill_on_device`, probed by
`_can_defer_device_sampling`), the forward leaves sampling out of
`decode_forward`/`prefill_forward`: the stored payload is the device sampling
input, and `sample_tokens()` (or DP finalization) runs the device sampler via
`_sample_deferred_device_output`. When the model lacks those entry points, the
forward still samples on device internally and the stored payload contains token
IDs, so `sample_tokens()` only finalizes, normalizes, packs, and applies the
already-sampled output. Either way this refactor keeps host control flow and the
vLLM-visible sampling phase separate from the forward.

`TTModelInput` now carries `has_structured_outputs` instead of carrying the
actual grammar bitmask during input preparation. That flag is known from
`SchedulerOutput`, so the runner can decide whether device sampling is allowed
without waiting for the concrete grammar bitmask.

## Sync Non-DP

Old ordering:

```text
host schedule
host compute/get grammar
host build TTModelInput with grammar_bitmask
device forward
host/device sampling inside execute_model
host scheduler update
```

New ordering:

```text
host schedule
host submit execute_model non-blocking
device forward
host computes GrammarOutput while forward is running
host waits for execute_model future
host calls sample_tokens(grammar_output)
host applies grammar to logits when host sampling
host samples and updates runner state
host scheduler update
```

The key ordering change is that grammar production moves after forward
submission. For host sampling, the TT model returns logits or a logits-like
payload; grammar masking happens immediately before sampling. For device
sampling, structured outputs have already disabled that path, so device-sampled
tokens are only used for unconstrained requests.

## Sync DP

TT DP still uses the plugin-specific gathered-DP engine because the scheduler is
per DP rank but the TT device execution may run on local DP rank 0 with a merged
batch.

Old ordering:

```text
each rank schedules
each rank computes grammar
each rank builds TTModelInput with grammar_bitmask
rank 0 gathers/forwards inputs to device ranks
device rank concatenates DP inputs
device rank runs forward and sampling together
rank 0 scatters sampled token IDs/logprobs
each rank applies local result and updates scheduler
```

New ordering:

```text
each rank schedules
each rank builds TTModelInput without grammar_bitmask
rank 0 gathers/forwards inputs to device ranks
device rank concatenates DP inputs
device rank runs gathered forward and stores TTForwardOutput
each rank computes local GrammarOutput
rank 0 gathers GrammarOutput plus local request-index maps
device rank samples stored gathered forward output
rank 0 scatters sampled token IDs/logprobs
each rank applies local result and updates scheduler
```

The important detail is that grammar is gathered before token scatter. Applying
grammar after `apply_dp_execution_result()` would be too late because that path
only receives sampled token IDs, not logits. The DP sampling step therefore
receives a per-rank grammar payload and applies the bitmask to each DP slice of
the merged logits before tokens are packed for scatter.

## Async Non-DP

Normal vLLM async batch-queue logic can now be used for TT non-DP execution
because TT exposes the same `execute_model() -> None` then `sample_tokens()`
contract as the GPU runner.

Old ordering:

```text
host schedule
if grammar pending, defer whole TT execution
otherwise compute grammar
submit TT execute_model_with_grammar
TT forward and sampling happen in one worker call
host queues ModelRunnerOutput future
host scheduler update after future resolves
```

New ordering:

```text
host schedule
submit TT execute_model
TT submits decode/prefill and stores TTForwardOutput
host computes GrammarOutput separately
host queues/executes sample_tokens(grammar_output)
host scheduler update after sampling output resolves
```

For async decode, the TT async decode controller still manages submitted decode
state, read events, and steady-decode safety checks. The structured-output
decision no longer depends on a ready `GrammarOutput`; it uses
`SchedulerOutput` plus request sampling parameters to decide whether steady
decode/device sampling is legal.

Structured outputs disable the steady decode fast path because grammar-masked
host sampling must see the correct logits and bitmask together. No sneaky device
sampling, no "trust me bro" grammar.

## Async DP

Async DP keeps the TT gathered-DP engine because it must coordinate collectives
across ranks. The queue can still overlap a submitted gathered forward with
later host scheduling and grammar work.

Old ordering:

```text
each rank schedules
grammar may be computed before DP gather
dp_gather_submit gathers model inputs that may include grammar-derived fields
concat_and_execute_dp runs forward plus sampling
future resolves to packed token/logprob result
dp_gather_finalize scatters result
each rank updates scheduler
```

New ordering:

```text
each rank schedules
dp_gather_submit gathers model inputs without grammar_bitmask
concat_and_execute_dp runs gathered forward and stores DP TTForwardOutput
future resolves when forward payload is ready
dp_gather_finalize computes/gathers per-rank GrammarOutput
sample_dp_forward_output samples the stored gathered forward payload
dp_gather_finalize scatters packed token/logprob result
each rank updates scheduler
```

The async DP queue now treats gathered forward completion and gathered sampling
as separate phases. This preserves the useful overlap: grammar computation can
happen after forward submission, while token scatter still waits until
grammar-constrained sampling has produced final token IDs.

## Structured Outputs

Structured-output processing moved from input construction to sampling.

Previously:

- The engine computed `GrammarOutput` before TT model execution.
- `_prepare_model_inputs()` converted `GrammarOutput.grammar_bitmask` to a torch
  tensor and reordered it into persistent TT batch order.
- `TTModelInput.grammar_bitmask` was carried through non-DP and DP paths.
- `_get_output_tokens()` applied that already-packed bitmask before host
  sampling.

Now:

- `_prepare_model_inputs()` only computes `has_structured_outputs`.
- `has_structured_outputs` disables device sampling before forward is submitted.
- `sample_tokens(grammar_output)` or `sample_dp_forward_output(...)` receives
  the actual grammar data.
- The grammar bitmask is reordered at sampling time using the request IDs and
  request-index map that belong to the rank being sampled.
- Host logits are masked immediately before the host sampler runs.

For DP, the grammar payload includes both the `GrammarOutput` and the rank's
`req_id_to_index` mapping. That mapping matters because each rank has its own
local scheduler and persistent batch order, while sampling happens on a merged
batch on device ranks.

Device sampling remains disabled for structured outputs. This is deliberate:
grammar-constrained device sampling is not assumed to exist yet. In particular,
optimized model implementations such as Galaxy Llama 3 70B switch to their
device sampler when `sampling_params` is passed, so structured-output requests
must keep forcing the logits-returning host-sampling path.

## Advantages, Drawbacks, and Performance

The main advantage is correctness and architectural alignment. TT now follows
the same high-level contract as the shared V1 engine: forward runs first,
grammar is produced from scheduler state, and sampling consumes both the forward
payload and the grammar payload. That means structured-output data is no longer
a special TT-only input to model execution. Less bespoke plumbing, fewer weird
corners. Always nice when the architecture stops wearing a fake mustache.

The split also improves overlap opportunities. In the non-DP path, the engine
can submit TT forward before computing `GrammarOutput`, so grammar-mask work can
run while the device is busy. In gathered DP, the same idea applies at the rank
level: ranks can compute local grammar after gathered forward has been
submitted, and gathered sampling waits only when it actually needs those grammar
payloads.

For structured-output requests, the expected performance effect is usually
positive or neutral on the host side because the forward pass is no longer
blocked by grammar-mask construction. However, structured outputs still force
host sampling today. That means constrained requests may still pay the cost of
returning logits to host and running host-side mask application plus sampling.
The split does not make structured outputs device-fast by itself; it makes the
sequencing correct and creates the place where device-fast support can plug in.

For unconstrained requests that already use device sampling, performance should
be close to the previous behavior. The TT model still performs device sampling
inside the model call when `perform_device_sampling=True`; `sample_tokens()`
mostly finalizes the output. The extra host phase is a small control-flow cost,
not a new logits readback. In practice, the large cost remains TT execution and
any device/host transfers already required by the selected sampling mode.

The main drawback is statefulness. Forward now leaves a stored payload
(`TTForwardOutput` or the gathered-DP equivalent) that must be consumed exactly
once by sampling. Bugs in ordering become easier to express: calling
`sample_tokens()` without a prior forward, losing a DP grammar payload, or
sampling twice would all be invalid. The code now relies more directly on the
standard vLLM invariant that `sample_tokens()` immediately follows an
`execute_model()` result of `None`.

DP also becomes conceptually more complex. There are now two DP phases after
scheduling: gathered forward, then gathered sampling. This is necessary because
grammar must be applied before token scatter, but it adds another collective
object path for per-rank grammar output and request-index maps.

## Device Grammar Support

The deferred-sampling mechanism that grammar-aware device sampling needs is
already in place. When a forward defers sampling, `_sample_deferred_device_output`
calls `sample_decode_on_device` / `sample_prefill_on_device` with a `bitmask`
argument, and `_device_sampling_bitmask` reorders the per-rank grammar into the
device batch order. So the constrained path

```text
forward produces device sampling payload
sample_tokens passes grammar bitmask to device sampling/finalization
device applies grammar bitmask and samples
host receives sampled token IDs/logprobs
```

is wired end to end.

What is still pending is *turning it on for structured batches*.
`check_perform_device_sampling()` returns `False` whenever
`has_structured_outputs` is set (for both prefill and decode), so structured
requests continue to take the host path:

```text
forward returns logits -> host applies grammar bitmask -> host samples
```

To enable device grammar sampling, that guard must change from "any structured
output disables device sampling" to "does this model/device support grammar
bitmask application for this batch?". Once it does, structured-output decode
batches gain the largest win - avoiding full logits readback and host-side
bitmask application - and gathered-DP sampling can consume the per-rank grammar
payloads on device instead of forcing host sampling for the merged logits.

## Plugin-Specific Subclasses

The non-DP TT engine used to need plugin-specific step behavior because TT did
not follow the shared vLLM split. `TTExecutionMixin` implemented custom
`step()` and `step_with_batch_queue_tt()` methods that computed grammar before
execution and called `execute_model_with_grammar()`.

That behavior is no longer needed. TT now exposes the same worker contract as
the normal V1 engine:

```text
execute_model(scheduler_output) -> None
sample_tokens(grammar_output) -> ModelRunnerOutput
```

As a result:

- `TTExecutionMixin` is removed.
- `TTEngineCore` can directly inherit `EngineCore`.
- `TTEngineCoreProc` can directly inherit `EngineCoreProc`.
- `execute_model_with_grammar()` is removed from `TTWorker`.

The `TTEngineCore` and `TTEngineCoreProc` class names may still exist as thin
plugin-selected classes because `TTPlatform` wires engine class paths through
configuration. They no longer need custom sequencing logic. The tiny subclass
shell is boring on purpose; boring is good here.

`TTDPEngineCoreProc` still needs plugin-specific subclassing. DP gather is not a
plain upstream engine flow: it has TT-specific rank coordination, gather/scatter
collectives, local-DP-rank device ownership, forced prefill/decode scheduling,
and the new gathered-DP forward/sampling split.