# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tests for the lane-DP step seams.

No device execution; the lane row/slot index logic runs on plain tensors with
fake runner/model collaborators. Lane-specific input/output shaping lives on
``TTLaneInputBatch`` (``extract_output``); the thin step orchestration lives on
``TTModelRunner`` (``_execute_lane_step``). Requires the ttnn-enabled
environment because importing the plugin modules pulls in ttnn.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm_tt_plugin.async_decode import (
    SubmittedStepContext,
    TTAsyncDecodeController,
    TTDecodeSubmission,
    TTFinalizedDecode,
)
from vllm_tt_plugin.input_batch import TTLaneInputBatch
from vllm_tt_plugin.lane_scheduler import TTStepPlan
from vllm_tt_plugin.model_runner import TTModelRunner
from vllm_tt_plugin.scheduler import TTScheduler

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

VOCAB = 8
BLOCK = 16
MAX_MODEL_LEN = 256


def _lane_batch(num_lanes=1, per_lane=5):
    return TTLaneInputBatch(
        num_lanes=num_lanes,
        per_lane=per_lane,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * num_lanes * per_lane,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
    )


def _scheduler_output_with_plan(plan: TTStepPlan) -> SchedulerOutput:
    output = SchedulerOutput.make_empty()
    output.total_num_scheduled_tokens = len(plan.scheduled_rows)
    output.num_scheduled_tokens = {req_id: 1 for req_id in plan.scheduled_req_ids}
    output._tt_step_state = SimpleNamespace(plan=plan)
    return output


def _capturing_host_sampler(captured: dict):
    """Fake host sampler: records the logits it sees, returns per-row argmax."""

    def sampler(logits, sampling_metadata):
        captured["logits"] = logits
        return SimpleNamespace(
            sampled_token_ids=logits.argmax(dim=-1), logprobs_tensors=None
        )

    return sampler


# --------------------------------------------------------------------------
# Orchestration guard
# --------------------------------------------------------------------------


def test_finished_request_releases_model_resources_before_row_removal():
    released = []
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_id_to_index={"req-0": 3}),
        model=SimpleNamespace(release_request=released.append),
    )

    TTModelRunner._release_finished_model_request(runner, "req-0")

    assert released == [3]


def test_block_output_updates_full_runner_state_at_exact_capacity():
    output_tokens = []
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=1,
            num_tokens=np.array([0], dtype=np.int32),
            token_ids_cpu=np.zeros((1, MAX_MODEL_LEN), dtype=np.int32),
            req_output_token_ids=[output_tokens],
        ),
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
    )
    block = torch.arange(MAX_MODEL_LEN, dtype=torch.int32).reshape(1, -1)

    TTModelRunner._apply_sampled_tokens_to_state(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [MAX_MODEL_LEN]
    assert runner.input_batch.token_ids_cpu[0].tolist() == list(range(MAX_MODEL_LEN))
    assert output_tokens == list(range(MAX_MODEL_LEN))


def test_block_output_rejects_max_length_overrun_before_state_write():
    output_tokens = []
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=1,
            num_tokens=np.array([1], dtype=np.int32),
            token_ids_cpu=np.zeros((1, MAX_MODEL_LEN), dtype=np.int32),
            req_output_token_ids=[output_tokens],
        ),
        model_config=SimpleNamespace(max_model_len=MAX_MODEL_LEN),
    )
    block = torch.arange(MAX_MODEL_LEN, dtype=torch.int32).reshape(1, -1)

    with pytest.raises(AssertionError, match="exceed the max model length"):
        TTModelRunner._apply_sampled_tokens_to_state(runner, block)

    assert runner.input_batch.num_tokens.tolist() == [1]
    assert output_tokens == []
    assert not runner.input_batch.token_ids_cpu.any()


def _scheduler_request(*, placeholders=1, computed=33):
    return SimpleNamespace(
        discard_latest_async_tokens=False,
        status=RequestStatus.RUNNING,
        num_output_placeholders=placeholders,
        num_computed_tokens=computed,
    )


def test_scheduler_accounts_for_full_256_token_block(monkeypatch):
    monkeypatch.setattr(
        Scheduler,
        "_update_request_with_output",
        lambda self, request, token_ids: (token_ids, False),
    )
    cached = []
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(cache_blocks=lambda *args: cached.append(args))
    )
    request = _scheduler_request()
    block = list(range(256))

    returned, stopped = TTScheduler._update_request_with_output(
        scheduler, request, block
    )

    assert returned == block
    assert stopped is False
    assert request.num_output_placeholders == 0
    assert request.num_computed_tokens == 288
    assert cached == []


@pytest.mark.parametrize(
    ("surviving_tokens", "expected_computed"),
    [
        ([17] * 17, 49),  # max-length trim keeps part of a committed canvas
        ([2], 33),  # EOS at the first position keeps one token
    ],
)
def test_scheduler_accounts_after_stop_trimming(
    monkeypatch, surviving_tokens, expected_computed
):
    monkeypatch.setattr(
        Scheduler,
        "_update_request_with_output",
        lambda self, request, token_ids: (surviving_tokens, True),
    )
    cached = []
    scheduler = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(cache_blocks=lambda *args: cached.append(args))
    )
    request = _scheduler_request()

    returned, stopped = TTScheduler._update_request_with_output(
        scheduler, request, list(range(256))
    )

    assert returned == surviving_tokens
    assert stopped is True
    assert request.num_output_placeholders == 0
    assert request.num_computed_tokens == expected_computed
    assert bool(cached) is (len(surviving_tokens) == 1)


def test_lane_step_rejects_request_specific_rope():
    runner = SimpleNamespace(request_specific_rope=True)
    output = _scheduler_output_with_plan(
        TTStepPlan(
            is_decode=True,
            capacity=2,
            scheduled_req_ids=("req-0",),
            scheduled_rows=(0,),
            input_rows=(0, 1),
            req_id_to_row={"req-0": 0},
            batch_size_per_dp=(2,),
            prefill_empty_slots=None,
        )
    )

    with pytest.raises(NotImplementedError, match="request-specific RoPE"):
        TTModelRunner._execute_lane_step(runner, output)


# --------------------------------------------------------------------------
# TTLaneInputBatch.extract_output (merged device/host read-back)
# --------------------------------------------------------------------------


def test_extract_output_device_decode_returns_scheduled_rows_in_order():
    batch = _lane_batch()
    # Full slot decode output: one sampled token per device slot.
    tt_out = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
    model_input = SimpleNamespace(
        perform_device_sampling=True,
        tt_sampling_params=SimpleNamespace(
            enable_log_probs=torch.zeros(5, dtype=torch.bool)
        ),
        max_num_logprobs=[None],
    )

    sampled, logprobs = batch.extract_output(
        SimpleNamespace(),
        tt_out,
        None,
        model_input,
        scheduled_rows=[4, 1],
        is_decode=True,
    )

    assert sampled.tolist() == [[14], [11]]  # slots 4 and 1, in scheduled order
    assert logprobs is None


def test_extract_output_device_prefill_returns_front_packed_tokens():
    batch = _lane_batch()
    # Prefill returns one token per scheduled request, in plan order.
    tt_out = torch.tensor([20, 21], dtype=torch.int32)
    model_input = SimpleNamespace(
        perform_device_sampling=True,
        tt_sampling_params=SimpleNamespace(
            enable_log_probs=torch.zeros(2, dtype=torch.bool)
        ),
        max_num_logprobs=[None],
    )

    sampled, logprobs = batch.extract_output(
        SimpleNamespace(),
        tt_out,
        None,
        model_input,
        scheduled_rows=[4, 1],
        is_decode=False,
    )

    assert sampled.tolist() == [[20], [21]]
    assert logprobs is None


def test_extract_output_host_decode_samples_full_slot_then_picks_rows():
    captured: dict = {}
    batch = _lane_batch()
    batch.build_merged_sampling_metadata = lambda rows: None  # sampler ignores it
    runner = SimpleNamespace(host_sampler=_capturing_host_sampler(captured))
    # Full slot logits: row r's argmax is token r (vocab>=5).
    logits = torch.full((5, VOCAB), -10.0)
    for r in range(5):
        logits[r, r] = 5.0
    model_input = SimpleNamespace(perform_device_sampling=False, grammar_bitmask=[None])

    sampled, logprobs = batch.extract_output(
        runner, logits, None, model_input, scheduled_rows=[4, 1], is_decode=True
    )

    assert captured["logits"].shape == (5, VOCAB)  # sampled the whole slot batch once
    assert sampled.tolist() == [[4], [1]]  # rows 4 and 1, in scheduled order
    assert logprobs is None


def test_extract_output_host_prefill_scatters_logits_to_stable_rows():
    captured: dict = {}
    batch = _lane_batch()
    batch.build_merged_sampling_metadata = lambda rows: None
    runner = SimpleNamespace(host_sampler=_capturing_host_sampler(captured))
    # Prefill logits: one row per scheduled request (front-packed, plan order).
    prefill_logits = torch.full((2, VOCAB), -10.0)
    prefill_logits[0, 3] = 5.0  # scheduled request 0 -> token 3
    prefill_logits[1, 6] = 5.0  # scheduled request 1 -> token 6
    model_input = SimpleNamespace(perform_device_sampling=False, grammar_bitmask=[None])

    sampled, _ = batch.extract_output(
        runner,
        prefill_logits,
        None,
        model_input,
        scheduled_rows=[4, 1],
        is_decode=False,
    )

    full = captured["logits"]
    assert full.shape == (5, VOCAB)  # scattered onto the full slot grid before sampling
    assert torch.equal(full[4], prefill_logits[0])  # request 0 -> stable row 4
    assert torch.equal(full[1], prefill_logits[1])  # request 1 -> stable row 1
    for gap in (0, 2, 3):
        assert torch.all(full[gap] == 0)  # unscheduled rows left empty
    assert sampled.tolist() == [[3], [6]]


# --------------------------------------------------------------------------
# Shared device submission / async lane decode
# --------------------------------------------------------------------------


def test_submit_prefill_forwards_plan_empty_slots_to_model():
    captured: dict = {}

    class FakeModel:
        def prefill_forward(self, **kwargs):
            captured.update(kwargs)
            return object()

    runner = SimpleNamespace(
        kv_caches=object(),
        trace_mode="none",
        request_specific_rope=False,
        model=FakeModel(),
    )
    model_input = SimpleNamespace(
        input_tokens=torch.zeros((1, 1), dtype=torch.int32),
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        prompt_lens=[1],
        input_positions=torch.zeros((1,), dtype=torch.int32),
        block_tables_per_layer=None,
        multi_modal_kwargs={},
        perform_device_sampling=False,
        prefill_empty_slots=[4],
    )

    # Multi-lane batch (len > 1) so the empty-slots path is exercised; the
    # stable slot 4 must reach the model so decode reads that user's KV slot.
    TTModelRunner.submit_prefill(runner, model_input, [2, 2])

    assert captured["empty_slots"] == [4]


def test_async_lane_decode_uses_batch_extraction():
    calls = []

    class FakeLaneBatch:
        def extract_output(
            self, runner, tt_out, tt_log_probs, model_input, scheduled_rows, is_decode
        ):
            calls.append(
                (runner, tt_out, tt_log_probs, model_input, scheduled_rows, is_decode)
            )
            return torch.tensor([[7]], dtype=torch.int32), None

    runner = SimpleNamespace(lane_batch=FakeLaneBatch())
    controller = TTAsyncDecodeController(runner)
    controller.finalize_decode = lambda submission: TTFinalizedDecode(
        tt_out=torch.tensor([[1, 2, 3]], dtype=torch.float32),
        tt_log_probs=None,
    )
    submission = TTDecodeSubmission(
        tt_out=object(),
        read_events=None,
        batch_size_per_dp=[2],
        sampling_params=object(),
        perform_device_sampling=False,
    )
    context = SubmittedStepContext(
        req_ids=["req-4"],
        req_id_to_index={"req-4": 0},
        request_states=(),
        submit_time_ns=123,
    )
    model_input = object()

    completed = controller.complete_decode_step(
        submission=submission,
        model_input=model_input,
        context=context,
        scheduled_rows=[4],
    )

    assert completed.sampled_token_ids.tolist() == [[7]]
    assert completed.logprobs is None
    assert completed.context is context
    assert len(calls) == 1
    seen_runner, tt_out, tt_log_probs, seen_model_input, scheduled_rows, is_decode = (
        calls[0]
    )
    assert seen_runner is runner
    assert torch.equal(tt_out, torch.tensor([[1, 2, 3]], dtype=torch.float32))
    assert tt_log_probs is None
    assert seen_model_input is model_input
    assert scheduled_rows == [4]
    assert is_decode is True
