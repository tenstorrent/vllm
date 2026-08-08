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
from vllm_tt_plugin.model_input import TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner

from vllm.v1.core.sched.output import SchedulerOutput

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
    batch.build_merged_sampling_metadata = (
        lambda rows, non_sampling_rows=None: None
    )  # sampler ignores it
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
    batch.build_merged_sampling_metadata = lambda rows, non_sampling_rows=None: None
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


def test_extract_output_skips_all_intermediate_prefill_rows():
    batch = _lane_batch()
    runner = SimpleNamespace(
        host_sampler=lambda *args, **kwargs: pytest.fail("sampler must not run")
    )
    model_input = SimpleNamespace(
        perform_device_sampling=False,
        grammar_bitmask=[None],
        intermediate_prefill_mask=torch.tensor([True]),
    )

    sampled, logprobs = batch.extract_output(
        runner,
        torch.zeros((1, VOCAB)),
        None,
        model_input,
        scheduled_rows=[0],
        is_decode=False,
    )

    assert sampled.tolist() == [[0]]
    assert logprobs is None


def test_build_host_generators_preserves_intermediate_request_rng():
    intermediate = torch.Generator().manual_seed(11)
    final = torch.Generator().manual_seed(22)
    intermediate_before = intermediate.get_state().clone()
    final_before = final.get_state().clone()

    class FakeInputBatch:
        sampling = SimpleNamespace(generators={0: intermediate, 1: final})

        def advance_generators(self, rows):
            for row in rows:
                torch.rand(1, generator=self.sampling.generators[row])

    generators = TTModelRunner._build_host_generators(
        FakeInputBatch(), [0, 1], torch.tensor([True, False])
    )

    assert generators[0] is not intermediate
    assert torch.equal(generators[0].get_state(), intermediate_before)
    assert generators[1] is final
    assert torch.equal(intermediate.get_state(), intermediate_before)
    assert not torch.equal(final.get_state(), final_before)


def test_get_output_tokens_skips_all_intermediate_prefill_rows():
    runner = SimpleNamespace(
        host_sampler=lambda *args, **kwargs: pytest.fail("sampler must not run")
    )
    model_input = SimpleNamespace(intermediate_prefill_mask=torch.tensor([True]))

    sampled, logprobs = TTModelRunner._get_output_tokens(
        runner,
        torch.zeros((1, 1, VOCAB)),
        None,
        SimpleNamespace(),
        model_input,
        [1],
        False,
        False,
    )

    assert sampled[0].tolist() == [[0]]
    assert logprobs == [None]


def test_apply_dp_result_suppresses_intermediate_prefill_rows():
    captured: dict = {}
    runner = SimpleNamespace(
        _build_chunked_prefill_output=lambda **kwargs: captured.update(kwargs)
        or "intermediate",
        apply_and_build_runner_output=lambda *args, **kwargs: pytest.fail(
            "intermediate rows must not be applied"
        ),
    )

    output = TTModelRunner.apply_dp_execution_result(
        runner,
        torch.tensor([[17], [23]], dtype=torch.int32),
        req_ids=["intermediate", "final"],
        req_id_to_index={"intermediate": 0, "final": 1},
        intermediate_prefill_mask=torch.tensor([True, False]),
    )

    assert output == "intermediate"
    assert captured["sampled_token_ids"].tolist() == [[17], [23]]
    assert captured["intermediate_mask"].tolist() == [True, False]


def test_finish_lane_sync_suppresses_intermediate_prefill_output():
    model_input = SimpleNamespace(prompt_lens=[3])

    def extract_output(
        runner, tt_out, tt_log_probs, model_input, scheduled_rows, *, is_decode
    ):
        assert is_decode is False
        return torch.tensor([[42]], dtype=torch.int32), None

    def unexpected_final_output(*args, **kwargs):
        raise AssertionError("intermediate prefill must not emit a sampled token")

    runner = SimpleNamespace(
        _apply_grammar_to_input=lambda model_input,
        grammar_output,
        *,
        lane_total: model_input,
        lane_batch=SimpleNamespace(
            extract_output=extract_output,
            req_ids={0: "request"},
            num_tokens=[8],
        ),
        apply_and_build_runner_output=unexpected_final_output,
    )

    def build_chunked_prefill_output(**kwargs):
        return TTModelRunner._build_chunked_prefill_output(runner, **kwargs)

    runner._build_chunked_prefill_output = build_chunked_prefill_output

    output = TTModelRunner._finish_lane_sync(
        runner,
        None,
        tt_out=object(),
        tt_log_probs=None,
        model_input=model_input,
        scheduled_rows=[0],
        is_decode=False,
        lane_total=1,
    )

    assert output.req_ids == ["request"]
    assert output.sampled_token_ids == [[]]


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


def _sampling_params(rows=2):
    """Every field a per-row tensor, as ``submit_decode``'s ``.tolist()`` expects."""
    return TTSamplingParams(
        temperature=torch.ones(rows),
        top_k=torch.zeros(rows, dtype=torch.int32),
        top_p=torch.ones(rows),
        presence_penalty=torch.zeros(rows),
        frequency_penalty=torch.zeros(rows),
        repetition_penalty=torch.ones(rows),
        seed=torch.zeros(rows, dtype=torch.int64),
        num_logprobs=torch.full((rows,), -1, dtype=torch.int32),
        enable_log_probs=torch.zeros(rows, dtype=torch.bool),
    )


@pytest.mark.parametrize("perform_device_sampling", [True, False])
def test_submit_decode_forwards_slot_remap_to_model(perform_device_sampling):
    """``slot_remap`` moves per-slot state (GDN recurrent/conv), so it is not a
    sampling parameter: one client's ``logprobs=5`` flips the whole step to host
    sampling, and the gather still has to run for every co-batched request."""
    captured: dict = {}

    class FakeModel:
        def decode_forward(self, **kwargs):
            captured.update(kwargs)
            return torch.zeros((1, 1), dtype=torch.float32)

    runner = SimpleNamespace(
        kv_caches=object(),
        trace_mode="none",
        request_specific_rope=False,
        model=FakeModel(),
    )
    remap = torch.tensor([1, 0], dtype=torch.int32)
    model_input = SimpleNamespace(
        input_tokens=torch.zeros((2, 1), dtype=torch.int32),
        block_tables=torch.zeros((2, 1), dtype=torch.int32),
        input_positions=torch.zeros((2,), dtype=torch.int32),
        block_tables_per_layer=None,
        unpadded_batch_size=2,
        tt_sampling_params=_sampling_params(),
        perform_device_sampling=perform_device_sampling,
        prompt_tokens=None,
        output_tokens=None,
        reset_batch=False,
        slot_remap=remap,
    )

    controller = TTAsyncDecodeController(runner)
    controller.submit_decode(model_input, read_from_device=True)

    assert torch.equal(captured["slot_remap"], remap)
    assert ("sampling_params" in captured) is perform_device_sampling


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
