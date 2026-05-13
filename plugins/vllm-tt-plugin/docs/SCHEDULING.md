# TT Scheduling and Execution Flow

This note summarizes how TT execution works in the current TT vLLM integration, how scheduling and queueing behave today, and how that differs from upstream vLLM.

The emphasis here is on the concepts:

- sync vs async
- non-DP vs gathered-DP
- what is scheduled locally vs what is coordinated globally

Code pointers are intentionally minimal. If you need them, the main entry points are `plugins/vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py`, `plugins/vllm-tt-plugin/src/vllm_tt_plugin/engine.py`, and `plugins/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py` in the TT vLLM tree.

## Short Version

The current TT path is more specialized than upstream vLLM:

- A TT step is treated as either all-prefill or all-decode.
- TT does not support mixed prefill+decode batches.
- TT does not support chunked prefill at the scheduling level.
- CPU-device work overlap is a decode optimization.
- Non-DP and gathered-DP use different execution paths.
- Gathered-DP is not just "the same thing on more ranks"; it adds a global mode negotiation and a gather/execute/scatter step around every DP execution.

Upstream vLLM is more general:

- The scheduler is token-budget based, not phase-based.
- A single step can naturally include both "prefill-like" and "decode-like" progress.
- Chunked prefill is part of the normal scheduler model.
- Async queueing is a generic executor pipeline mechanism, not a TT-specific decode overlap mechanism.

## Mental Model

There are still two important scheduler-side collections:

- `waiting`: requests not yet admitted into active execution
- `running`: requests already admitted and holding active scheduler state

What TT changes is not the existence of those queues, but the rules for choosing what kind of work a step may contain and how that scheduled work is executed afterward.

For TT, it is useful to think in terms of two batch types:

- prefill batch: admits new work from `waiting`
- decode batch: advances already-running work from `running`

That is a simplification compared with upstream vLLM, but it matches how the TT path is intentionally organized today.

## Current TT Execution Flow

### 1. Request admission

New requests enter the scheduler's `waiting` queue.

The TT scheduler prefers to admit waiting work first, because TT wants to form a clean prefill step when possible. If no prefill can be admitted, and there are already-running decode requests, the scheduler can fall back to a decode-only step so progress continues and KV pressure can relax.

### 2. Scheduler decision

The current TT scheduler enforces two important rules:

- no mixed prefill+decode batch
- no chunked prefill

So each TT scheduling step picks one of:

- prefill-only
- decode-only
- empty step

This is the main conceptual difference from upstream.

### 3. Engine step selection

After scheduling, the engine uses one of three execution styles:

1. Synchronous path
2. Non-DP async path
3. Gathered-DP async path

Which path is used depends mostly on:

- whether async scheduling is enabled
- whether data parallel execution requires gathered coordination

### 4. Worker/model-runner execution

The TT worker forwards scheduled work into the TT model runner.

At that point, the important split is:

- prefill is effectively synchronous
- decode may be synchronous or asynchronously overlapped

Even when a call crosses an async-looking executor boundary, TT prefill still behaves like a synchronous step in practice. The meaningful overlap optimization is decode.

### 5. State update

When model output is available, the scheduler updates request state, moves requests between `waiting` and `running` as needed, and emits outputs back to the engine.

In async decode mode, there can be a controlled one-step lag between:

- submitting decode work to the device
- applying the completed result back to scheduler/host state

That lag is intentional and is what creates host/device overlap.

## How TT Scheduling Works Now

### Local scheduler policy

The TT scheduler behaves like this:

- if there is admissible waiting work, prefer prefill
- otherwise, advance decode work
- if waiting exists but cannot be admitted as full prefill, decode can run to free capacity

The reason for this policy is simple: TT currently wants a homogeneous batch type per step, and it wants full-prefill admission rather than upstream-style incremental prefill progress.

### Why TT uses an async-style scheduler even in TT-specific flows

TT uses an async-capable scheduler base because decode overlap needs output placeholders: a request can be scheduled one step ahead before the previous step's output has been fully applied on the host.

This does not mean the whole TT path is fully asynchronous.

It means:

- the scheduler can safely "reserve" the next decode token
- the engine can submit the next decode step before the prior result is fully retired

That mechanism matters most for steady-state decode.

## Queueing in Non-DP TT

### What is queued

In non-DP mode, the engine keeps a host-side queue of in-flight scheduled steps. Conceptually, each queue entry is:

- the scheduled batch description
- a future or future-like handle for its output

The queue is used with a "fill before blocking" policy:

- if more work can be scheduled, submit it first
- only block on the oldest in-flight result when needed

### What async really means here

For TT non-DP, async primarily means:

- decode submission can run ahead of host-side result application
- device readback/finalization can complete later
- the host can spend that time scheduling the next decode step

It does not mean:

- arbitrary prefill/decode mixing
- unlimited queue depth of useful overlapped TT work
- every model step is non-blocking in the same way

In practice:

- prefill remains a synchronous-style step
- decode can use the steady async path when invariants hold
- during steady-state decode, one or two steps are submitted to the device at any time. Thus the device should see no gaps in execution.

### Threading and waiting in non-DP

There are three different mechanisms involved, and they serve different purposes.

#### 1. Engine batch queue

The engine keeps a small queue of in-flight scheduled steps. In TT async non-DP, that queue depth is `2` when async scheduling is enabled.

This queue defines how many scheduled steps can be outstanding. It is an engine/executor queue, not a TT-model-internal queue.

#### 2. Executor-side future resolution

When the engine submits `execute_model(..., non_block=True)`, it gets back a future or future-like object.

What happens next depends on the executor mode:

- uniprocess mode: if the TT runner returns an async output wrapper, a single background `ThreadPoolExecutor` thread calls `get_output()`
- multiprocess mode: the worker process itself calls `get_output()` before sending the response back to the engine

In uniprocess mode there is one background output thread. In multiprocess mode the waiting happens inside the worker process.

#### 3. TT decode completion signaling

Inside the TT runner, async decode completion is tracked with `threading.Event` objects plus lock-protected deques.

These structures do not execute the decode. They only track whether a submitted async decode step has finished host-side finalization.

The important pieces are:

- `_pending_async_events`: decode steps submitted but not yet marked complete
- `_completed_decode_steps`: decode results that have completed readback/finalization but have not yet been applied back to runner state
- `_steady_decode_lock`: protects those deques

The event is set only after `get_output()` has finished finalizing the decode output. So the event means "this decode result is ready to apply", not merely "submission happened".

#### 3a. Basic TT async mechanism

Under the hood, TT async decode is built on asynchronous host readback, not on a separate device-side execution thread managed by vLLM.

The basic sequence is:

1. Submit decode work with `decode_forward(..., read_from_device=False)`.
2. Ask the model to start host readback with `read_decode_output(..., async_read=True)`.
3. Keep the returned read events with the submission record.
4. Later, during finalization, wait on those read events with `ttnn.event_synchronize(...)`.
5. Only after those events complete, convert the decode output into normal host tensors and sampling results.

So the low-level meaning of "non-blocking" here is:

- do not immediately read the decode output back to the host
- issue host read requests asynchronously
- defer the blocking wait until finalization time

This is why the higher-level future/event bookkeeping exists. It is tracking when those asynchronous readbacks have become safe to consume.

#### 4. Where the code actually waits

There are two important wait points in non-DP TT:

- engine-level wait: when the batch queue cannot be filled further, the engine blocks on the oldest queued future
- runner-level drain wait: before leaving the steady decode fast path, TT may wait for all pending async decode events and then apply the completed steps

So the host does not continuously poll. It mostly waits at explicit boundaries:

- `future.result()` at the engine/executor boundary
- `event.wait()` when the runner must drain pending async decode work

### When steady async decode is allowed

TT only keeps decode overlapped when the batch is "steady" enough. In plain terms, overlap is allowed only when the batch shape and sampling path are stable.

Overlap is disabled and pending async work is drained when correctness would otherwise become ambiguous, for example when there is:

- prompt activity or resumed prefill work
- layout change in the decode batch
- structured output bookkeeping
- penalties or host-side logits processing
- host-only sampling requirements
- logprobs or other features that force a more synchronous path

So the TT async path is best understood as a fast path for steady decode, not as a universal async execution model.

## Queueing in Gathered-DP TT

Gathered-DP is conceptually different from non-DP.

### What stays local

Each DP rank still has its own local scheduler state:

- local `waiting`
- local `running`
- local admission decisions

### What becomes global

Before a gathered-DP step is executed, ranks first negotiate a global batch mode:

- force prefill
- force decode

That negotiation is necessary because the gathered TT execution wants all ranks to participate in the same kind of step. Without that, one rank could try to admit prefill while another tries to advance decode, which would break the merged execution model.

### The gathered-DP execution shape

Once the mode is chosen:

1. Each rank schedules locally under that forced mode.
2. Each rank builds its local TT model input or an empty contribution.
3. Per-rank inputs are gathered into one merged execution payload.
4. The merged TT batch is executed on rank 0. The model (and tt-runtime) is responsible for submitting the work from rank 0 to the mesh.
5. Results are read on rank 0 and split back by DP rank.
6. Each rank applies only its own local result to its local scheduler state.

That is the defining difference between gathered-DP and non-DP.

### Queueing model in gathered-DP

Gathered-DP does not behave like a generic multi-entry engine queue.

Conceptually, it keeps at most one gathered execution step in flight:

- previous gathered step may still be finishing
- current step may be prepared and submitted
- completion of the previous step may happen before or after the next submit, depending on whether overlap is safe

So this code is closer to a controlled one-step pipeline than to a broad queue of independent futures.
However, since the non-DP queue has depth of just 2 the host-device overlap timeline is similar in effect.

### Threading and waiting in gathered-DP

Gathered-DP uses less executor-side threading than the non-DP uniprocess path.

The important waiting mechanisms are:

- synchronous collectives for global mode negotiation and input gather
- an in-flight gathered handle representing the previous decode submission
- synchronous finalization when the previous gathered step must be retired

At the lowest level, gathered-DP decode uses the same TT readback mechanism as non-DP decode:

- submit decode without immediate host readback
- arm asynchronous readback
- later wait on read events with `ttnn.event_synchronize(...)`

The difference is not the TT read primitive. The difference is the surrounding control flow: gathered-DP wraps that same read/finalize mechanism in cross-rank collectives and a single in-flight gathered handle.

The async part is narrower than in non-DP:

- decode submission may return a future-like gathered handle
- finalization of that handle can be delayed until the next step
- but all ranks still have to enter the collectives in the same order

So gathered-DP is a synchronized collective loop with a possible one-step decode overlap.

### Sync vs async in gathered-DP

Gathered-DP is mixed-mode:

- host-side gather/orchestration is synchronous
- prefill execution is effectively synchronous
- decode may use a one-step-ahead async submission/finalization pattern

So in "DP async", the collective coordination remains synchronous. The async part is mostly the decode execution/readback overlap inside that coordinated loop.

## Non-DP vs Gathered-DP

| Topic | Non-DP TT | Gathered-DP TT |
| --- | --- | --- |
| Scheduler state | Local only | Local per rank |
| Batch type per step | Prefill-only or decode-only | Same, globally forced across ranks |
| Queueing shape | Host-side in-flight batch queue | Usually one gathered step in flight |
| Prefill behavior | Synchronous-style | Synchronous-style |
| Decode overlap | Yes, when steady | Yes, but only after global coordination says it is safe |
| Cross-rank coordination | None | Required every step |
| Execution payload | Local TT model input | Gathered merged TT model input |

## TT vs Upstream vLLM

### 1. Scheduling model

Upstream uses a generic token-budget scheduler.

The important property is that upstream scheduling is not organized around an explicit "prefill batch vs decode batch" split. A request simply has computed tokens and target tokens, and the scheduler assigns more token work subject to budgets and constraints.

That naturally supports:

- mixed progress across different requests in one step
- chunked prefill
- a more uniform scheduling model across backends

The current TT path is more constrained:

- it treats prefill and decode as separate batch modes
- it avoids mixed prefill+decode batches
- it avoids chunked prefill

### 2. Async queueing model

Upstream has a generic batch-queue path that lets the engine keep the executor fed before blocking on the oldest completed result.

That queueing model is backend-agnostic. It is mainly about executor pipelining.

TT inherits that idea, but changes the meaning of "async":

- in TT non-DP, the useful overlap is mostly decode steady-state overlap
- in TT gathered-DP, the queueing becomes a single in-flight gathered pipeline rather than the generic upstream queue model

### 2a. Waiting model

Upstream generic async queueing is mostly expressed as futures around executor work.

The TT path adds more explicit decode-completion bookkeeping:

- a background output thread in uniprocess mode, or blocking output handling inside the worker process in multiprocess mode
- TT decode completion events
- a lock-protected queue of completed decode steps that are applied later

So compared with upstream, TT has more explicit "submit now, finalize later, apply later" logic around decode completion.

### 3. Sampling boundary

In upstream , execution and token sampling are more naturally separable.

In the TT path, more of the decode and sampling behavior is bundled into TT-specific execution handling because the device/host readback path and device-sampling path are part of the TT execution contract.

This is one reason TT needs extra execution helpers instead of using only the generic upstream engine path.

### 4. DP behavior

Upstream does not use the TT-style gathered-DP execution contract described above.

The TT gathered-DP path adds several concepts that are TT-specific:

- global prefill/decode mode negotiation
- gather of per-rank TT inputs into one merged execution payload
- scatter/apply of per-rank outputs back into local scheduler state
- conservative global checks before allowing steady decode overlap

## Practical Takeaways

If you want the most accurate mental model for the current TT stack, use this:

- The scheduler is local, but TT execution rules are specialized.
- TT prefill and decode are treated as different batch modes.
- Async mainly means decode overlap, not fully async end-to-end execution.
- Non-DP uses a local queue of in-flight steps.
- Gathered-DP uses a globally coordinated one-step pipeline.
- Upstream vLLM `main` is more general and less phase-constrained than the TT path.

That is the core reason the TT branch currently has TT-specific scheduler and engine-step code rather than fitting entirely inside the upstream generic scheduler/executor flow.
