# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tests for gathered-DP mode negotiation."""

from types import SimpleNamespace

import pytest
import vllm_tt_plugin.engine as engine_module
from vllm_tt_plugin.engine import TTDPEngineCoreProc
from vllm_tt_plugin.scheduler import TTSchedulingMode

from vllm.v1.core.sched.output import SchedulerOutput


def _core_with_scheduler(scheduler):
    core = TTDPEngineCoreProc.__new__(TTDPEngineCoreProc)
    core.scheduler = scheduler
    core.dp_group = object()
    core.dlog = lambda *args, **kwargs: None
    return core


def test_dp_negotiation_prefers_running_prefill_continuation(monkeypatch):
    """Prefill continuations force gathered DP into prefill mode."""
    scheduler = SimpleNamespace(
        waiting=[],
        running=[SimpleNamespace(is_prefill_chunk=True)],
        max_num_running_reqs=1,
    )
    core = _core_with_scheduler(scheduler)

    def all_reduce(tensor, *, op, group):
        assert tensor.tolist() == [1]
        assert op == engine_module.dist.ReduceOp.MAX
        assert group is core.dp_group

    monkeypatch.setattr(engine_module.dist, "all_reduce", all_reduce)

    assert core._dp_negotiate_forced_mode() == TTSchedulingMode.PREFILL_ONLY
    assert core._dp_gather_forced_mode == TTSchedulingMode.PREFILL_ONLY


def test_dp_zero_prefill_falls_back_collectively(monkeypatch):
    """Zero global prefill progress falls back to decode when one exists."""
    prefill_output = SchedulerOutput.make_empty()
    prefill_output.finished_req_ids.add("finished-prefill")
    prefill_output.free_encoder_mm_hashes.append("encoder-prefill")
    decode_output = SchedulerOutput.make_empty()
    decode_output.total_num_scheduled_tokens = 2
    scheduler = SimpleNamespace(
        running=[SimpleNamespace(is_prefill_chunk=False)],
        has_unfinished_requests=lambda: True,
        schedule=lambda: decode_output,
        set_forced_mode=lambda mode: None,
    )
    core = _core_with_scheduler(scheduler)
    all_reduce_calls = []

    def all_reduce(tensor, *, op, group):
        all_reduce_calls.append(op)
        tensor.copy_(engine_module.torch.tensor([0, 1]))
        assert group is core.dp_group

    monkeypatch.setattr(engine_module.dist, "all_reduce", all_reduce)

    result = core._dp_schedule_with_zero_prefill_fallback(
        TTSchedulingMode.PREFILL_ONLY, prefill_output
    )

    assert result is decode_output
    assert result.finished_req_ids == {"finished-prefill"}
    assert result.free_encoder_mm_hashes == ["encoder-prefill"]
    assert core._dp_gather_forced_mode == TTSchedulingMode.DECODE_ONLY
    assert all_reduce_calls == [engine_module.dist.ReduceOp.SUM]


def test_dp_zero_prefill_fallback_keeps_output_when_rank_has_drained(monkeypatch):
    """A rank with nothing left to schedule keeps its output instead of None."""
    prefill_output = SchedulerOutput.make_empty()
    prefill_output.finished_req_ids.add("finished-prefill")
    prefill_output.free_encoder_mm_hashes.append("encoder-prefill")
    scheduler = SimpleNamespace(
        running=[],
        has_unfinished_requests=lambda: False,
        schedule=lambda: (_ for _ in ()).throw(AssertionError("unexpected reschedule")),
    )
    core = _core_with_scheduler(scheduler)

    def all_reduce(tensor, *, op, group):
        tensor.copy_(engine_module.torch.tensor([0, 1]))

    monkeypatch.setattr(engine_module.dist, "all_reduce", all_reduce)

    result = core._dp_schedule_with_zero_prefill_fallback(
        TTSchedulingMode.PREFILL_ONLY, prefill_output
    )

    assert result is prefill_output
    assert result.finished_req_ids == {"finished-prefill"}
    assert result.free_encoder_mm_hashes == ["encoder-prefill"]
    assert core._dp_gather_forced_mode == TTSchedulingMode.DECODE_ONLY


@pytest.mark.parametrize(
    ("running", "probe"),
    [
        pytest.param(
            [SimpleNamespace(is_prefill_chunk=False)],
            [1, 1],
            id="prefill-progress-on-any-rank",
        ),
        pytest.param([], [0, 0], id="no-running-decode"),
    ],
)
def test_dp_zero_prefill_no_fallback(monkeypatch, running, probe):
    """Prefill remains selected when fallback conditions are incomplete."""
    prefill_output = SchedulerOutput.make_empty()
    scheduler = SimpleNamespace(
        running=running,
        has_requests=lambda: True,
        schedule=lambda: (_ for _ in ()).throw(AssertionError("unexpected reschedule")),
    )
    core = _core_with_scheduler(scheduler)
    core._dp_gather_forced_mode = TTSchedulingMode.PREFILL_ONLY

    def all_reduce(tensor, *, op, group):
        tensor.copy_(engine_module.torch.tensor(probe))

    monkeypatch.setattr(engine_module.dist, "all_reduce", all_reduce)

    result = core._dp_schedule_with_zero_prefill_fallback(
        TTSchedulingMode.PREFILL_ONLY, prefill_output
    )

    assert result is prefill_output
    assert core._dp_gather_forced_mode == TTSchedulingMode.PREFILL_ONLY


def test_dp_step_uses_decode_fallback_output(monkeypatch):
    """Synchronous DP steps execute the decode fallback output."""
    prefill_output = SchedulerOutput.make_empty()
    decode_output = SchedulerOutput.make_empty()
    decode_output.total_num_scheduled_tokens = 1
    modes = []

    def schedule():
        return (
            decode_output
            if modes[-1] == TTSchedulingMode.DECODE_ONLY
            else prefill_output
        )

    scheduler = SimpleNamespace(
        running=[SimpleNamespace(is_prefill_chunk=False)],
        has_requests=lambda: True,
        has_unfinished_requests=lambda: True,
        schedule=schedule,
        set_forced_mode=modes.append,
        get_grammar_bitmask=lambda output: None,
        update_from_output=lambda output, model_output: {"updated": output},
    )
    core = _core_with_scheduler(scheduler)
    core._scheduler_paused = False
    executed = []

    def all_reduce(tensor, *, op, group):
        tensor.copy_(engine_module.torch.tensor([0, 1]))

    monkeypatch.setattr(engine_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(core, "_dp_any_rank_has_scheduler_requests", lambda: True)
    monkeypatch.setattr(
        core,
        "_dp_negotiate_forced_mode",
        lambda: TTSchedulingMode.PREFILL_ONLY,
    )
    monkeypatch.setattr(
        core,
        "_execute_model_dp_gather",
        lambda output, grammar: executed.append(output) or None,
    )
    monkeypatch.setattr(core, "_process_aborts_queue", lambda: None)

    outputs, model_executed = core.step()

    assert outputs["updated"] is decode_output
    assert model_executed is True
    assert executed == [decode_output]
    assert modes[:2] == [
        TTSchedulingMode.PREFILL_ONLY,
        TTSchedulingMode.DECODE_ONLY,
    ]
    assert modes[-1] == TTSchedulingMode.DEFAULT


def test_dp_async_step_submits_decode_after_prefill_fallback(monkeypatch):
    """Async DP steps submit the decode fallback with decode mode active."""
    prefill_output = SchedulerOutput.make_empty()
    decode_output = SchedulerOutput.make_empty()
    decode_output.total_num_scheduled_tokens = 1
    modes = []

    def schedule():
        return (
            decode_output
            if modes[-1] == TTSchedulingMode.DECODE_ONLY
            else prefill_output
        )

    scheduler = SimpleNamespace(
        running=[SimpleNamespace(is_prefill_chunk=False)],
        has_requests=lambda: True,
        has_unfinished_requests=lambda: True,
        schedule=schedule,
        set_forced_mode=modes.append,
        get_grammar_bitmask=lambda output: None,
    )
    core = _core_with_scheduler(scheduler)
    core.batch_queue = object()
    core._dp_in_flight = None
    core.is_ec_producer = False
    submitted = []

    def all_reduce(tensor, *, op, group):
        tensor.copy_(engine_module.torch.tensor([0, 1]))

    def submit(output, grammar, *, overlap_ok):
        submitted.append((output, core._dp_gather_forced_mode, overlap_ok))
        return object()

    monkeypatch.setattr(engine_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(core, "_dp_any_rank_has_scheduler_requests", lambda: True)
    monkeypatch.setattr(
        core,
        "_dp_negotiate_forced_mode",
        lambda: TTSchedulingMode.PREFILL_ONLY,
    )
    monkeypatch.setattr(
        core,
        "_dp_can_attempt_steady_decode_from_scheduler",
        lambda output, grammar: True,
    )
    monkeypatch.setattr(core, "dp_gather_submit", submit)

    outputs, model_executed = core.step_dp_with_batch_queue()

    assert outputs == {}
    assert model_executed is True
    assert submitted == [(decode_output, TTSchedulingMode.DECODE_ONLY, True)]
    assert modes[:2] == [
        TTSchedulingMode.PREFILL_ONLY,
        TTSchedulingMode.DECODE_ONLY,
    ]
    assert modes[-1] == TTSchedulingMode.DEFAULT
