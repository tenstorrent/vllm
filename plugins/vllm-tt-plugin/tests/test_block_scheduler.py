# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from vllm_tt_plugin import config as tt_config
from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode, _PendingOutputs

from vllm.config import (
    CacheConfig,
    DeviceConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import FinishReason
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

BLOCK_SIZE = 16
MAX_MODEL_LEN = 256
CANVAS_LENGTH = 16


def _scheduler(
    *,
    output_tokens_per_step=CANVAS_LENGTH,
    max_model_len=MAX_MODEL_LEN,
    max_num_batched_tokens=None,
    async_scheduling=False,
    enable_prefix_caching=False,
    enable_chunked_prefill=False,
):
    model_config = ModelConfig(
        model="Qwen/Qwen2-0.5B-Instruct",
        dtype="float16",
        seed=42,
    )
    model_config.max_model_len = max_model_len
    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        max_num_batched_tokens=(max_num_batched_tokens or max_model_len),
        max_model_len=max_model_len,
        enable_chunked_prefill=enable_chunked_prefill,
        async_scheduling=async_scheduling,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=BLOCK_SIZE,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=ParallelConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    # The host test environment's platform config hook (CPUPlatform, or
    # TTPlatform itself when ttnn is importable) rewrites async scheduling and
    # chunked-prefill settings for the probe model. Restore the requested
    # values after generic VllmConfig construction so these tests exercise
    # the intended scheduler configuration.
    vllm_config.scheduler_config.async_scheduling = async_scheduling
    vllm_config.scheduler_config.enable_chunked_prefill = enable_chunked_prefill
    vllm_config.scheduler_config.max_num_batched_tokens = (
        max_num_batched_tokens or max_model_len
    )
    tt_config.store_tt_output_tokens_per_step(vllm_config, output_tokens_per_step)

    num_blocks = max_model_len // BLOCK_SIZE + 2
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            )
        ],
    )
    cache_config.num_gpu_blocks = num_blocks
    scheduler = TTScheduler(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        block_size=BLOCK_SIZE,
        log_stats=True,
        structured_output_manager=StructuredOutputManager(vllm_config),
    )
    return scheduler


def _request(
    prompt_len, max_tokens=MAX_MODEL_LEN, *, ignore_eos=True, request_id="req-0"
):
    init_none_hash(sha256)
    return Request(
        request_id=request_id,
        prompt_token_ids=[1] * prompt_len,
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        ),
        pooling_params=None,
        eos_token_id=2,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _schedule_block_zero(scheduler, request):
    scheduler.add_request(request)
    scheduler_output = scheduler.schedule()
    assert scheduler_output.num_scheduled_tokens == {
        request.request_id: request.num_prompt_tokens
    }
    return scheduler_output


def _runner_output(
    scheduler_output: SchedulerOutput,
    tokens: list[int],
) -> ModelRunnerOutput:
    req_ids = list(scheduler_output.num_scheduled_tokens)
    assert len(req_ids) == 1
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_ids[0]: 0},
        sampled_token_ids=[tokens],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_scheduler_uses_real_constructor_and_normalized_capability():
    scheduler = _scheduler()

    assert type(scheduler) is TTScheduler
    assert scheduler._output_tokens_per_step == CANVAS_LENGTH
    assert scheduler._is_block_output_model is True


def test_scheduler_reserves_full_canvas_before_block_zero_dispatch():
    scheduler = _scheduler()
    request = _request(prompt_len=32)

    _schedule_block_zero(scheduler, request)

    assert request.num_output_placeholders == CANVAS_LENGTH
    assert request.num_computed_tokens == 32 + CANVAS_LENGTH - 1


def test_scheduler_reserves_decode_canvas_before_base_dispatch():
    scheduler = _scheduler()
    request = _request(prompt_len=32, max_tokens=CANVAS_LENGTH * 2)
    _schedule_block_zero(scheduler, request)
    returned, stopped = scheduler._update_request_with_output(
        request, list(range(CANVAS_LENGTH))
    )
    assert returned == list(range(CANVAS_LENGTH))
    assert stopped is False

    observed = {}
    allocate_slots = scheduler.kv_cache_manager.allocate_slots

    def capture_reservation(req, *args, **kwargs):
        observed["placeholders"] = req.num_output_placeholders
        observed["computed"] = req.num_computed_tokens
        observed["max_model_len"] = scheduler.max_model_len
        return allocate_slots(req, *args, **kwargs)

    scheduler.kv_cache_manager.allocate_slots = capture_reservation
    scheduler_output = scheduler.schedule()

    assert scheduler_output.num_scheduled_tokens == {request.request_id: 1}
    assert observed == {
        "placeholders": 0,
        "computed": request.num_tokens - 1,
        "max_model_len": MAX_MODEL_LEN,
    }
    assert request.num_output_placeholders == CANVAS_LENGTH
    assert request.num_computed_tokens == request.num_tokens + CANVAS_LENGTH - 1


def test_async_exact_boundary_never_schedules_nonpositive_tokens():
    max_model_len = 262144
    prompt_len = 261376
    scheduler = _scheduler(
        output_tokens_per_step=256,
        max_model_len=max_model_len,
        async_scheduling=True,
    )
    request = _request(prompt_len=prompt_len, max_tokens=768)
    block = list(range(256))

    block_zero = _schedule_block_zero(scheduler, request)
    block_one = scheduler.schedule()
    assert block_one.num_scheduled_tokens == {request.request_id: 1}
    assert request.num_computed_tokens == prompt_len + 2 * 256 - 1
    assert request.num_output_placeholders == 2 * 256

    scheduler.update_from_output(
        block_zero,
        _runner_output(block_zero, block),
    )
    block_two = scheduler.schedule()
    assert block_two.num_scheduled_tokens == {request.request_id: 1}
    assert request.num_computed_tokens == max_model_len - 1
    assert request.num_output_placeholders == 2 * 256

    scheduler.update_from_output(
        block_one,
        _runner_output(block_one, block),
    )
    blocked = scheduler.schedule()
    assert blocked.num_scheduled_tokens == {}
    assert blocked.total_num_scheduled_tokens == 0
    assert request.num_output_placeholders == 256

    outputs = scheduler.update_from_output(
        block_two,
        _runner_output(block_two, block),
    )

    for scheduled in (block_zero, block_one, block_two):
        assert all(value > 0 for value in scheduled.num_scheduled_tokens.values())
    assert request.num_output_tokens == 768
    assert request.num_output_placeholders == 0
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert outputs[0].outputs[0].new_token_ids == block


def test_scheduler_rejects_bypassed_waiting_request_without_canvas_capacity():
    scheduler = _scheduler()
    request = _request(
        prompt_len=MAX_MODEL_LEN - CANVAS_LENGTH + 1,
        max_tokens=1,
    )
    scheduler.add_request(request)

    scheduler_output = scheduler.schedule()

    assert scheduler_output.total_num_scheduled_tokens == 0
    assert scheduler_output.finished_req_ids == {request.request_id}
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED

    # The request was never scheduled, so the base update loop builds no
    # output for it; the scheduler must emit the terminal event itself or
    # the client waits forever.
    outputs = scheduler.update_from_output(
        scheduler_output,
        ModelRunnerOutput(
            req_ids=[],
            req_id_to_index={},
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    (terminal,) = outputs[0].outputs
    assert terminal.request_id == request.request_id
    assert terminal.finish_reason == FinishReason.LENGTH
    assert terminal.new_token_ids == []


def test_max_tokens_trims_final_canvas_and_reclaims_reservation():
    scheduler = _scheduler()
    request = _request(prompt_len=32, max_tokens=CANVAS_LENGTH + 1)
    block_zero = _schedule_block_zero(scheduler, request)
    scheduler.update_from_output(
        block_zero,
        _runner_output(block_zero, list(range(CANVAS_LENGTH))),
    )
    block_one = scheduler.schedule()

    outputs = scheduler.update_from_output(
        block_one,
        _runner_output(block_one, list(range(CANVAS_LENGTH))),
    )

    assert list(request.output_token_ids) == [
        *range(CANVAS_LENGTH),
        0,
    ]
    assert outputs[0].outputs[0].new_token_ids == [0]
    assert request.num_output_placeholders == 0
    assert request.num_computed_tokens == request.num_tokens - 1
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED


def test_eos_trims_canvas_and_reclaims_reservation():
    scheduler = _scheduler()
    request = _request(prompt_len=32, ignore_eos=False)
    block_zero = _schedule_block_zero(scheduler, request)
    canvas = [0, 2, *range(2, CANVAS_LENGTH)]

    outputs = scheduler.update_from_output(
        block_zero,
        _runner_output(block_zero, canvas),
    )

    assert list(request.output_token_ids) == [0, 2]
    assert outputs[0].outputs[0].new_token_ids == [0, 2]
    assert request.num_output_placeholders == 0
    assert request.num_computed_tokens == request.num_tokens - 1
    assert request.status == RequestStatus.FINISHED_STOPPED


@pytest.mark.parametrize("enable_prefix_caching", [False, True])
def test_block_output_cache_policy_follows_prefix_cache_capability(
    enable_prefix_caching,
):
    scheduler = _scheduler(enable_prefix_caching=enable_prefix_caching)
    request = _request(prompt_len=32)
    _schedule_block_zero(scheduler, request)
    cache_calls = []
    scheduler.kv_cache_manager.cache_blocks = lambda *args: cache_calls.append(args)

    scheduler._update_request_with_output(request, list(range(CANVAS_LENGTH)))

    assert scheduler._cache_block_outputs is enable_prefix_caching
    assert bool(cache_calls) is enable_prefix_caching


def test_preemption_discards_outstanding_block_and_resumes_from_prefill():
    scheduler = _scheduler(async_scheduling=True)
    request = _request(prompt_len=32, max_tokens=CANVAS_LENGTH * 2)
    scheduler_output = _schedule_block_zero(scheduler, request)
    pending = _PendingOutputs.for_request(request)
    assert pending.outstanding == 1
    assert request.num_output_placeholders == CANVAS_LENGTH

    scheduler.running.remove(request)
    scheduler._preempt_request(request, timestamp=0.0)

    assert request.status == RequestStatus.PREEMPTED
    assert request.num_computed_tokens == 0
    assert request.num_output_placeholders == 0
    assert pending.stale == 1

    outputs = scheduler.update_from_output(
        scheduler_output,
        _runner_output(scheduler_output, list(range(CANVAS_LENGTH))),
    )
    assert all(not client_output.outputs for client_output in outputs.values())
    assert request.num_output_tokens == 0
    assert pending.outstanding == 0
    assert pending.stale == 0

    scheduler.prev_step_scheduled_req_ids.clear()
    resumed_output = scheduler.schedule()
    assert resumed_output.num_scheduled_tokens == {
        request.request_id: request.num_prompt_tokens
    }
    assert request.status == RequestStatus.RUNNING
    assert request.num_output_placeholders == CANVAS_LENGTH


def test_autoregressive_k1_keeps_async_scheduler_behavior():
    scheduler = _scheduler(output_tokens_per_step=1)
    request = _request(prompt_len=32, max_tokens=2)
    scheduler_output = _schedule_block_zero(scheduler, request)
    cache_calls = []
    scheduler.kv_cache_manager.cache_blocks = lambda *args: cache_calls.append(args)

    outputs = scheduler.update_from_output(
        scheduler_output, _runner_output(scheduler_output, [7])
    )

    assert scheduler._is_block_output_model is False
    assert scheduler._cache_block_outputs is True
    assert list(request.output_token_ids) == [7]
    assert request.num_output_placeholders == 0
    assert cache_calls
    assert outputs[0].outputs[0].new_token_ids == [7]


def test_autoregressive_k1_preserves_partial_chunked_prefill():
    scheduler = _scheduler(
        output_tokens_per_step=1,
        max_num_batched_tokens=32,
        enable_chunked_prefill=True,
    )
    request = _request(prompt_len=64, max_tokens=1)
    scheduler.add_request(request)

    first_chunk = scheduler.schedule()
    assert first_chunk.num_scheduled_tokens == {request.request_id: 32}
    assert request.is_prefill_chunk is True
    assert request.num_output_placeholders == 0

    second_chunk = scheduler.schedule()
    assert second_chunk.num_scheduled_tokens == {request.request_id: 32}
    assert request.is_prefill_chunk is False
    assert request.num_output_placeholders == 1


def test_scheduler_does_not_reserve_temporarily_unscheduled_request():
    scheduler = _scheduler()
    request = _request(prompt_len=32, max_tokens=CANVAS_LENGTH * 2)
    _schedule_block_zero(scheduler, request)
    _, stopped = scheduler._update_request_with_output(
        request, list(range(CANVAS_LENGTH))
    )
    assert stopped is False
    before = (
        request.num_computed_tokens,
        request.num_output_placeholders,
    )

    scheduler.set_forced_mode(TTSchedulingMode.PREFILL_ONLY)
    scheduler_output = scheduler.schedule()

    assert scheduler_output.total_num_scheduled_tokens == 0
    assert (
        request.num_computed_tokens,
        request.num_output_placeholders,
    ) == before


def test_async_final_canvas_in_flight_schedules_no_wasted_step():
    """The in-flight canvas surely satisfies max_tokens; no extra dispatch."""
    scheduler = _scheduler(async_scheduling=True)
    request = _request(prompt_len=32, max_tokens=CANVAS_LENGTH // 2)
    block_zero = _schedule_block_zero(scheduler, request)

    held = scheduler.schedule()

    assert held.total_num_scheduled_tokens == 0
    assert request in scheduler.running
    assert not request.is_finished()

    outputs = scheduler.update_from_output(
        block_zero,
        _runner_output(block_zero, list(range(3, 3 + CANVAS_LENGTH))),
    )
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert len(outputs[0].outputs[0].new_token_ids) == CANVAS_LENGTH // 2
    assert request.num_output_placeholders == 0


def test_held_request_keeps_max_num_seqs_slot():
    """A held request still owns its slot; no replacement is admitted."""
    scheduler = _scheduler(async_scheduling=True)
    r1 = _request(prompt_len=32, max_tokens=CANVAS_LENGTH // 2)
    block_zero = _schedule_block_zero(scheduler, r1)

    r2 = _request(prompt_len=16, max_tokens=CANVAS_LENGTH, request_id="req-1")
    scheduler.add_request(r2)

    held = scheduler.schedule()
    assert held.total_num_scheduled_tokens == 0
    assert not r1.is_finished()
    assert [r.request_id for r in scheduler.running] == ["req-0"]
    assert len(scheduler.running) <= scheduler.max_num_running_reqs

    scheduler.update_from_output(
        block_zero,
        _runner_output(block_zero, list(range(3, 3 + CANVAS_LENGTH))),
    )
    assert r1.is_finished()

    admitted = scheduler.schedule()
    assert admitted.num_scheduled_tokens == {"req-1": r2.num_prompt_tokens}


def test_sync_forced_reset_commits_fresh_canvas_after_resume():
    """pause/reset resume must not swallow the first regenerated canvas."""
    scheduler = _scheduler(async_scheduling=False)
    request = _request(prompt_len=32, max_tokens=3 * CANVAS_LENGTH)
    step_a = _schedule_block_zero(scheduler, request)
    canvas1 = list(range(3, 3 + CANVAS_LENGTH))
    scheduler.update_from_output(step_a, _runner_output(step_a, canvas1))
    assert request.num_output_tokens == CANVAS_LENGTH

    assert scheduler.reset_prefix_cache(reset_running_requests=True)
    assert request.status == RequestStatus.PREEMPTED
    assert request.discard_latest_async_tokens

    scheduler.prev_step_scheduled_req_ids.clear()
    step_b = scheduler.schedule()
    assert step_b.num_scheduled_tokens == {request.request_id: request.num_tokens}

    canvas2 = list(range(103, 103 + CANVAS_LENGTH))
    outputs = scheduler.update_from_output(step_b, _runner_output(step_b, canvas2))

    assert outputs[0].outputs[0].new_token_ids == canvas2
    assert request.num_output_tokens == 2 * CANVAS_LENGTH
    assert request.num_output_placeholders == 0
    assert not request.discard_latest_async_tokens


def test_async_forced_reset_with_running_block_request_is_refused():
    """Async runner state has already applied in-flight canvases; refuse."""
    scheduler = _scheduler(async_scheduling=True)
    request = _request(prompt_len=32)
    _schedule_block_zero(scheduler, request)

    assert scheduler.reset_prefix_cache(reset_running_requests=True) is False
    assert request.status == RequestStatus.RUNNING
    assert request.num_output_placeholders == CANVAS_LENGTH


def test_default_mode_fallback_preserves_finished_req_ids():
    """The discarded prefill-only output consumes ``finished_req_ids``; the
    decode-only fallback must carry them so the model runner still releases
    the finished requests' state."""
    scheduler = _scheduler(async_scheduling=True)
    r1 = _request(prompt_len=32, max_tokens=4 * CANVAS_LENGTH)
    _schedule_block_zero(scheduler, r1)

    r2 = _request(prompt_len=16, max_tokens=CANVAS_LENGTH, request_id="req-1")
    r3 = _request(prompt_len=16, max_tokens=CANVAS_LENGTH, request_id="req-2")
    scheduler.add_request(r2)
    scheduler.add_request(r3)
    scheduler.finish_requests(["req-1"], RequestStatus.FINISHED_ABORTED)

    # r1 running decode + r3 waiting: prefill-only cannot admit (slot taken),
    # so DEFAULT mode discards that output and reschedules decode-only.
    scheduler_output = scheduler.schedule()

    assert scheduler_output.num_scheduled_tokens == {"req-0": 1}
    assert "req-1" in scheduler_output.finished_req_ids


def test_ar_async_preemption_discards_stale_and_resumes():
    """K=1 async keeps stock AsyncScheduler behavior through the TT preempt
    override (stale outputs dropped, placeholders reset, clean resume)."""
    scheduler = _scheduler(output_tokens_per_step=1, async_scheduling=True)
    request = _request(prompt_len=32, max_tokens=8)
    step_a = _schedule_block_zero(scheduler, request)
    step_b = scheduler.schedule()
    assert step_b.num_scheduled_tokens == {"req-0": 1}
    assert request.num_output_placeholders == 2

    scheduler.running.remove(request)
    scheduler._preempt_request(request, timestamp=0.0)
    assert request.status == RequestStatus.PREEMPTED
    assert request.num_output_placeholders == 0

    for stale_step in (step_a, step_b):
        outputs = scheduler.update_from_output(
            stale_step, _runner_output(stale_step, [7])
        )
        assert all(not client_output.outputs for client_output in outputs.values())
    assert request.num_output_tokens == 0

    scheduler.prev_step_scheduled_req_ids.clear()
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {"req-0": request.num_prompt_tokens}
    assert request.status == RequestStatus.RUNNING
    assert request.num_output_placeholders == 1
