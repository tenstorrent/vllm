# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import pickle
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
import torch.distributed as dist

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.v1.engine.core import DPEngineCoreProc, DPGatherHandle

logger = init_logger(__name__)
_T = TypeVar("_T")


def _unwrap_single_worker_future(future: Future[list[_T]]) -> Future[_T]:
    single_future: Future[_T] = Future()

    def _set_single_result(done_future: Future[list[_T]]) -> None:
        try:
            results = done_future.result()
            assert len(results) == 1
            single_future.set_result(results[0])
        except Exception as exc:
            single_future.set_exception(exc)

    future.add_done_callback(_set_single_result)
    return single_future


def _completed_dp_gather_future(
    core: DPEngineCoreProc,
) -> Future[tuple[torch.Tensor, list]]:
    parallel_config = core.vllm_config.parallel_config
    world = parallel_config.data_parallel_size
    batch_size = core.vllm_config.scheduler_config.max_num_seqs
    future: Future[tuple[torch.Tensor, list]] = Future()
    future.set_result(
        (torch.zeros((world, batch_size, 1), dtype=torch.int32), [None] * world)
    )
    return future


def _dp_any_rank_has_scheduler_requests(core: DPEngineCoreProc) -> bool:
    local_has_requests = 1 if core.scheduler.has_requests() else 0
    has_requests_t = torch.tensor([local_has_requests], dtype=torch.int32)
    try:
        dist.all_reduce(has_requests_t, op=dist.ReduceOp.SUM, group=core.dp_group)
    except RuntimeError as e:
        # During shutdown, peers may close connections mid-collective.
        # Exit gracefully to allow coordinated shutdown.
        if "Connection closed by peer" in str(e):
            logger.debug("Collective failed during shutdown, exiting gracefully")
            raise SystemExit() from e
        raise
    return int(has_requests_t.item()) > 0


def _dp_negotiate_forced_mode(core: DPEngineCoreProc) -> int:
    # Max-consecutive-decoding guard: if there are waiting prefills
    # and we decoded for dp_max_consec_decodes steps consecutively,
    # force one prefill step. Otherwise, prefer decode while running.
    # Can't automatically set intent to be prefill if there are any
    # waiting requests since we could get stuck in a loop where intent
    # is prefill but it doesn't get scheduled.
    has_running = bool(getattr(core.scheduler, "running", []))
    has_waiting = bool(getattr(core.scheduler, "waiting", False))
    must_prefill = has_waiting and core.dp_decode_streak >= core.dp_max_consec_decodes
    local_prefill_intent = (
        1 if (has_waiting and (must_prefill or not has_running)) else 0
    )
    intent_tensor = torch.tensor([local_prefill_intent], dtype=torch.int32)
    core.dlog("before_intent_allreduce intent_tensor=%s", intent_tensor)
    dist.all_reduce(intent_tensor, op=dist.ReduceOp.MAX, group=core.dp_group)
    forced_mode = int(intent_tensor.item())  # 1=prefill, 0=decode
    core.dlog("after_intent_allreduce forced_mode=%d", forced_mode)
    # Record forced_mode so it can be used by DP gather submission.
    core._dp_gather_forced_mode = forced_mode
    return forced_mode


def _dp_apply_forced_mode(core: DPEngineCoreProc, forced_mode: int | None) -> None:
    set_mode = getattr(core.scheduler, "set_forced_mode", None)
    if callable(set_mode):
        set_mode(forced_mode)


def _dp_update_decode_streak(
    core: DPEngineCoreProc,
    scheduler_output: SchedulerOutput | None,
    forced_mode: int | None,
) -> None:
    if scheduler_output is None or forced_mode is None:
        return
    # This is intentionally stale by at most one step in async DP mode:
    # forced-mode negotiation for step N+1 runs before update_from_output(N).
    if scheduler_output.total_num_scheduled_tokens > 0 and forced_mode == 0:
        core.dp_decode_streak += 1
    else:
        core.dp_decode_streak = 0


def step_dp_with_batch_queue(
    core: DPEngineCoreProc,
) -> tuple[dict[int, EngineCoreOutputs] | None, bool]:
    assert core.batch_queue is not None

    global_has_requests = _dp_any_rank_has_scheduler_requests(core)
    prev_handle = core._dp_in_flight
    if not global_has_requests and prev_handle is None:
        return {}, False

    forced_mode: int | None = None
    # DP-gather forced scheduling mode:
    #   None -> use the scheduler's default policy
    #   0    -> force decode-only scheduling
    #   1    -> force prefill-only scheduling
    scheduler_output: SchedulerOutput | None = None
    model_executed = False
    current_overlap_ok = False
    if global_has_requests:
        forced_mode = _dp_negotiate_forced_mode(core)
        if core.scheduler.has_requests():
            _dp_apply_forced_mode(core, forced_mode)
            scheduler_output = core.scheduler.schedule()
            _dp_apply_forced_mode(core, None)
            _dp_update_decode_streak(core, scheduler_output, forced_mode)
            if not core.is_ec_producer:
                model_executed = scheduler_output.total_num_scheduled_tokens > 0
        if forced_mode == 0:  # decode-only scheduling
            # All DP ranks must enter this collective once decode mode is chosen,
            # even if a particular rank has no local batch this step. Ranks with
            # no local work pass scheduler_output=None, which the worker treats
            # as steady-eligible, and the MIN reduction keeps the global decision
            # conservative.
            current_overlap_ok = _dp_can_attempt_steady_decode_from_scheduler(
                core, scheduler_output
            )

    def _finalize_previous(
        handle: DPGatherHandle,
    ) -> dict[int, EngineCoreOutputs]:
        model_output = dp_gather_finalize(core, handle)
        if handle.scheduler_output is None:
            return {}
        return core.scheduler.update_from_output(handle.scheduler_output, model_output)

    finalize_before_submit = prev_handle is not None and (
        not global_has_requests or not prev_handle.overlap_ok or not current_overlap_ok
    )

    engine_core_outputs: dict[int, EngineCoreOutputs] | None = {}
    if finalize_before_submit:
        assert prev_handle is not None
        engine_core_outputs = _finalize_previous(prev_handle)
        prev_handle = None

    if (
        scheduler_output is not None
        and scheduler_output.pending_structured_output_tokens
    ):
        core.scheduler.get_grammar_bitmask(scheduler_output)

    next_handle: DPGatherHandle | None = None
    if global_has_requests:
        next_handle = dp_gather_submit(
            core,
            scheduler_output,
            overlap_ok=current_overlap_ok,
        )

    if not finalize_before_submit and prev_handle is not None:
        engine_core_outputs = _finalize_previous(prev_handle)

    core._dp_in_flight = next_handle

    if not global_has_requests:
        return engine_core_outputs, False

    return engine_core_outputs, model_executed


def _dp_can_attempt_steady_decode_from_scheduler(
    core: DPEngineCoreProc,
    scheduler_output: SchedulerOutput | None,
) -> bool:
    local_overlap_ok = int(
        core.model_executor.collective_rpc(
            "can_attempt_steady_dp_decode_from_scheduler",
            args=(scheduler_output,),
        )[0]
    )
    overlap_ok_t = torch.tensor([local_overlap_ok], dtype=torch.int32)
    dist.all_reduce(overlap_ok_t, op=dist.ReduceOp.MIN, group=core.dp_group)
    overlap_ok = bool(overlap_ok_t.item())
    core.dlog("steady_decode_overlap_ok=%s", overlap_ok)
    return overlap_ok


def dp_gather_submit(
    core: DPEngineCoreProc,
    scheduler_output: SchedulerOutput | None,
    *,
    overlap_ok: bool = False,
) -> DPGatherHandle:
    """Prepare and submit one gathered-DP execution step.

    Collects per-rank DP inputs, negotiates the merged execution shape, and
    returns a `DPGatherHandle` consumed by `dp_gather_finalize()` on the
    next step.

    This path is mixed-mode rather than purely sync or async:
    - it is always synchronous for the host-side gather/orchestration work
    - for prefill, the worker executes synchronously even though we pass
      `non_block=True` through the executor boundary
    - for decode, the worker uses the async decode submit path and returns
      a future-like handle that is completed later in `dp_gather_finalize()`
    """
    from vllm.v1.engine.core import DPGatherHandle

    parallel_config = core.vllm_config.parallel_config
    group = core.dp_group
    rank = core.dp_rank
    local_rank = parallel_config.data_parallel_rank_local
    world = parallel_config.data_parallel_size

    local_has_requests = scheduler_output is not None
    if scheduler_output is not None:
        core.dlog("enter_gather tokens=%d", scheduler_output.total_num_scheduled_tokens)

    # Detect decode vs prefill using forced_mode negotiated earlier.
    # 0=decode, 1=prefill.
    assert hasattr(core, "_dp_gather_forced_mode"), "forced_mode not set"
    is_decode = core._dp_gather_forced_mode == 0

    # Build local dp model input (or None).
    all_local_inputs = core.model_executor.collective_rpc(
        "build_dp_model_input", args=(scheduler_output,)
    )[0]  # type: ignore[var-annotated]
    (
        local_input,
        local_max_blocks,
        local_has_structured,
        local_has_penalties,
        local_reset_batch,
        local_can_sample_device,
        local_needs_logprobs,
        req_ids,
        req_id_to_index,
    ) = all_local_inputs
    max_blocks_decode = None  # Only used for decode.
    any_structured_inputs = False  # Only used for decode.
    any_needs_logprobs = False

    gathered_inputs: Any = None
    if is_decode:
        # Gather max_blocks, has_structured, has_penalties, reset_batch,
        # cannot_sample_on_device, and needs_logprobs from all ranks.
        input_info_t = torch.tensor(
            [
                local_max_blocks,
                local_has_structured,
                local_has_penalties,
                local_reset_batch,
                # Invert so we can use MAX reduction and still compute
                # "all ranks can sample on device".
                1 - local_can_sample_device,
                local_needs_logprobs,
            ],
            dtype=torch.int32,
        )
        dist.all_reduce(input_info_t, op=dist.ReduceOp.MAX, group=group)
        max_blocks_decode = int(input_info_t[0].item())
        any_structured_inputs = input_info_t[1].item() > 0
        any_penalties_inputs = input_info_t[2].item() > 0
        any_reset_batch = input_info_t[3].item() > 0
        all_sample_device = input_info_t[4].item() == 0
        any_needs_logprobs = input_info_t[5].item() > 0

        # Build tensorized gather input for decode.
        decode_inputs: dict[str, Any] = core.model_executor.collective_rpc(
            "build_dp_decode_gather_input",
            args=(
                local_input,
                max_blocks_decode,
                any_structured_inputs,
                any_penalties_inputs,
            ),
        )[0]

        # Decode: use gather with fixed-shape inputs.
        int_local = decode_inputs["int_inputs"]  # 1D int32
        float_local = decode_inputs["float_inputs"]  # 1D float32

        # Only rank 0 needs buffers for the gather operation.
        # Pre-allocate stacked tensors and create list views for gather.
        stacked_int = None
        stacked_float = None
        gather_list_int = None
        gather_list_float = None
        if rank == 0:
            stacked_int = torch.empty((world, *int_local.shape), dtype=int_local.dtype)
            stacked_float = torch.empty(
                (world, *float_local.shape), dtype=float_local.dtype
            )
            gather_list_int = [stacked_int[i] for i in range(world)]
            gather_list_float = [stacked_float[i] for i in range(world)]

        # Gather to rank 0, then send to other device ranks (local rank 0).
        # Note: if num device ranks == num world ranks, then this could be
        # all_gather but we usually have device ranks << world.
        dist.gather(int_local, gather_list_int, dst=0, group=group)
        dist.gather(float_local, gather_list_float, dst=0, group=group)
        if len(core.dp_device_ranks) > 1:
            if rank == 0:
                # Rank 0 sends stacked tensors to other device ranks.
                for dst in core.dp_device_ranks[1:]:
                    dist.send(stacked_int, dst=dst, group=group)
                    dist.send(stacked_float, dst=dst, group=group)
            elif local_rank == 0:  # other device ranks
                # Other device ranks receive from rank 0.
                stacked_int = torch.empty(
                    (world, *int_local.shape), dtype=int_local.dtype
                )
                stacked_float = torch.empty(
                    (world, *float_local.shape), dtype=float_local.dtype
                )
                dist.recv(stacked_int, src=0, group=group)
                dist.recv(stacked_float, src=0, group=group)

        # Gather prompt/output tokens only when they are needed (penalties)
        # and (if sampling on host or the decode batch layout changed).
        gathered_tokens_inputs = None
        if any_penalties_inputs and (not all_sample_device or any_reset_batch):
            # Gather token dicts (containing tensors) to rank 0
            if rank == 0:
                gathered_tokens_inputs = [None for _ in range(world)]
            local_tokens_inputs = decode_inputs["sampling_tokens_inputs"]
            dist.gather_object(
                local_tokens_inputs, gathered_tokens_inputs, dst=0, group=group
            )

            if len(core.dp_device_ranks) > 1:
                if rank == 0:
                    # Rank 0 sends gathered tokens to other device ranks
                    pickled_tokens = pickle.dumps(gathered_tokens_inputs)
                    tokens_tensor = torch.frombuffer(pickled_tokens, dtype=torch.uint8)
                    tokens_size = torch.tensor(
                        [tokens_tensor.numel()], dtype=torch.long
                    )
                    for dst in core.dp_device_ranks[1:]:
                        dist.send(tokens_size, dst=dst, group=group)
                        dist.send(tokens_tensor, dst=dst, group=group)
                elif local_rank == 0:  # other device ranks
                    # Other device ranks receive from rank 0
                    tokens_size = torch.zeros(1, dtype=torch.long)
                    dist.recv(tokens_size, src=0, group=group)
                    tokens_tensor = torch.empty(tokens_size.item(), dtype=torch.uint8)
                    dist.recv(tokens_tensor, src=0, group=group)
                    gathered_tokens_inputs = pickle.loads(
                        tokens_tensor.numpy().tobytes()
                    )

        # Gather host-only sampling params (logprobs, allowed_token_ids,
        # bad_words, logit_bias, min_p, min_tokens) when sampling on host.
        gathered_host_only_sample_params = None
        if not all_sample_device:
            if rank == 0:
                gathered_host_only_sample_params = [None for _ in range(world)]
            local_host_only_sample_params = decode_inputs.get("host_only_sample_params")
            dist.gather_object(
                local_host_only_sample_params,
                gathered_host_only_sample_params,
                dst=0,
                group=group,
            )

            if len(core.dp_device_ranks) > 1:
                if rank == 0:
                    # Rank 0 sends gathered host_only params to device ranks
                    pickled_host_only = pickle.dumps(gathered_host_only_sample_params)
                    host_only_tensor = torch.frombuffer(
                        pickled_host_only, dtype=torch.uint8
                    )
                    host_only_size = torch.tensor(
                        [host_only_tensor.numel()], dtype=torch.long
                    )
                    for dst in core.dp_device_ranks[1:]:
                        dist.send(host_only_size, dst=dst, group=group)
                        dist.send(host_only_tensor, dst=dst, group=group)
                elif local_rank == 0:  # other device ranks
                    # Other device ranks receive from rank 0
                    host_only_size = torch.zeros(1, dtype=torch.long)
                    dist.recv(host_only_size, src=0, group=group)
                    host_only_tensor = torch.empty(
                        host_only_size.item(), dtype=torch.uint8
                    )
                    dist.recv(host_only_tensor, src=0, group=group)
                    gathered_host_only_sample_params = pickle.loads(
                        host_only_tensor.numpy().tobytes()
                    )

        if local_rank == 0:
            gathered_inputs = {
                "int_inputs": stacked_int,
                "float_inputs": stacked_float,
                "sampling_tokens_inputs": gathered_tokens_inputs,
                "host_only_sample_params": gathered_host_only_sample_params,
                "reset_batch": any_reset_batch,
                "all_sample_device": all_sample_device,
            }

    else:
        # Prefill: use gather_object with variable sized inputs.
        gathered_inputs = None
        if rank == 0:
            gathered_inputs = [None for _ in range(world)]  # type: ignore

        # All_reduce to determine if any rank needs logprobs (for prefill).
        logprobs_flag_t = torch.tensor([local_needs_logprobs], dtype=torch.int32)
        dist.all_reduce(logprobs_flag_t, op=dist.ReduceOp.MAX, group=group)
        any_needs_logprobs = logprobs_flag_t[0].item() > 0

        # Gather to rank 0, then send to other device ranks (local rank 0).
        # Note: if num device ranks == num world ranks, then this could be
        # all_gather_object but we usually have device ranks << world.
        dist.gather_object(local_input, gathered_inputs, dst=0, group=group)
        if len(core.dp_device_ranks) > 1:
            if rank == 0:
                # Rank 0 sends gathered list to other device ranks only.
                pickled_data = pickle.dumps(gathered_inputs)
                object_tensor = torch.frombuffer(pickled_data, dtype=torch.uint8)
                size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long)
                for dst in core.dp_device_ranks[1:]:
                    dist.send(size_tensor, dst=dst, group=group)
                    dist.send(object_tensor, dst=dst, group=group)
            elif local_rank == 0:  # other device ranks
                # Other device ranks receive from rank 0.
                size_tensor = torch.zeros(1, dtype=torch.long)
                dist.recv(size_tensor, src=0, group=group)
                object_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8)
                dist.recv(object_tensor, src=0, group=group)
                gathered_inputs = pickle.loads(object_tensor.numpy().tobytes())
    core.dlog("after_inputs_gather")

    should_submit = is_decode or (
        isinstance(gathered_inputs, list)
        and any(x is not None for x in gathered_inputs)
    )
    if should_submit:
        collective_future = cast(
            Future[list[tuple[torch.Tensor, list]]],
            core.model_executor.collective_rpc(
                "concat_and_execute_dp",
                args=(
                    gathered_inputs,
                    is_decode,
                    max_blocks_decode,
                    any_structured_inputs,
                ),
                kwargs={"non_block": True},
                non_block=True,
            ),
        )
        future = _unwrap_single_worker_future(collective_future)
    else:
        future = _completed_dp_gather_future(core)

    return DPGatherHandle(
        future=future,
        scheduler_output=scheduler_output,
        local_has_requests=local_has_requests,
        is_decode=is_decode,
        overlap_ok=overlap_ok,
        any_needs_logprobs=any_needs_logprobs,
        req_ids=req_ids,
        req_id_to_index=req_id_to_index,
    )


def dp_gather_finalize(
    core: DPEngineCoreProc, handle: DPGatherHandle
) -> ModelRunnerOutput:
    parallel_config = core.vllm_config.parallel_config
    group = core.dp_group
    rank = core.dp_rank
    world = parallel_config.data_parallel_size
    logprobs_per_dp: list = [None] * world

    result = handle.future.result()  # Waiting for step output to be ready
    assert isinstance(result, tuple) and len(result) == 2
    send_tensor, logprobs_per_dp = result
    assert isinstance(send_tensor, torch.Tensor)

    # Global rank 0 scatters results to all ranks
    my_ids = torch.empty_like(send_tensor[0])  # Shape (B, 1)
    scatter_list = None
    if rank == 0:
        # Prepare scatter list: split send_tensor along first dimension
        scatter_list = [send_tensor[i] for i in range(world)]
    dist.scatter(my_ids, scatter_list, src=0, group=group)
    core.dlog("after_results_gather my_ids_shape=%s", tuple(my_ids.shape))

    # Scatter logprobs only if any rank needs them
    # (determined in all_reduce).
    my_logprobs_val = None
    if handle.any_needs_logprobs:
        my_logprobs: list = [None]
        logprobs_scatter_list = logprobs_per_dp if rank == 0 else None
        dist.scatter_object_list(my_logprobs, logprobs_scatter_list, src=0, group=group)
        my_logprobs_val = my_logprobs[0]

    # If rank had scheduled tokens, apply results locally and return output
    if handle.local_has_requests:
        output: ModelRunnerOutput = core.model_executor.collective_rpc(
            "apply_dp_execution_result",
            args=(
                my_ids,
                my_logprobs_val,
                handle.req_ids,
                handle.req_id_to_index,
            ),
        )[0]
        return output
    else:
        return EMPTY_MODEL_RUNNER_OUTPUT


def _execute_model_dp_gather(
    core: DPEngineCoreProc, scheduler_output: SchedulerOutput | None
) -> ModelRunnerOutput:
    handle = dp_gather_submit(core, scheduler_output, overlap_ok=False)
    return dp_gather_finalize(core, handle)
