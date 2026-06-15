# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the single-process lane-DP coordinator.

The coordinator is exercised over lightweight fake lane schedulers: a real
``TTScheduler`` needs a device KV cache config, but the coordinator only relies
on a small surface of each lane (``waiting`` / ``running`` length, forced-mode
scheduling, and ``update_from_output``).
"""

from types import SimpleNamespace

from vllm_tt_plugin.lane_scheduler import (
    TTStepPlan,
    TTLaneCoordinator,
    get_tt_step_plan,
)
from vllm_tt_plugin.scheduler import TTSchedulingMode

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutputs


class FakeLane:
    """Minimal stand-in for a per-lane ``TTScheduler``.

    Prefill always schedules zero tokens here (simulating "no full prefill fits"
    under KV pressure); decode schedules one token per running request. Pending
    finished IDs are emitted on the first ``schedule`` call and then drained, so
    tests can assert the coordinator carries them across a prefill->decode
    fallback.
    """

    def __init__(self, waiting=0, running=0, pending_finished=()):
        self.waiting = [object()] * waiting
        self.running = [object()] * running
        self._pending_finished = set(pending_finished)
        self._mode = TTSchedulingMode.DEFAULT
        self.scheduled_modes: list[TTSchedulingMode] = []
        self.update_calls: list[SchedulerOutput] = []
        self._eco: dict[int, EngineCoreOutputs] = {}

    def set_forced_mode(self, mode):
        self._mode = mode

    def schedule(self):
        self.scheduled_modes.append(self._mode)
        finished = self._pending_finished
        self._pending_finished = set()
        out = SchedulerOutput.make_empty()
        out.finished_req_ids = set(finished)
        if self._mode == TTSchedulingMode.DECODE_ONLY and self.running:
            out.num_scheduled_tokens = {f"dec-{id(self)}": len(self.running)}
            out.total_num_scheduled_tokens = len(self.running)
        return out

    def update_from_output(self, scheduler_output, model_runner_output):
        self.update_calls.append(scheduler_output)
        return self._eco


def _make_coordinator(lanes, *, per_lane_max=32, log_stats=False):
    coordinator = TTLaneCoordinator.__new__(TTLaneCoordinator)
    coordinator.lanes = lanes
    coordinator.num_lanes = len(lanes)
    coordinator._per_lane_max = per_lane_max
    coordinator.log_stats = log_stats
    coordinator.structured_output_manager = None
    coordinator.connector = None
    coordinator._last_lane_metadata = None
    coordinator._req_to_lane = {}
    coordinator._req_to_row = {}
    coordinator._free_slots_by_lane = [
        list(range(per_lane_max)) for _ in range(len(lanes))
    ]
    return coordinator


def _scheduled_output(req_ids):
    out = SchedulerOutput.make_empty()
    out.num_scheduled_tokens = {req_id: 1 for req_id in req_ids}
    out.total_num_scheduled_tokens = len(req_ids)
    return out


def test_negotiate_prefill_when_any_lane_wants_prefill():
    # Lane 1 has a queued request and nothing running -> wants prefill.
    coordinator = _make_coordinator([FakeLane(running=2), FakeLane(waiting=1)])
    assert coordinator._negotiate_forced_mode() == TTSchedulingMode.PREFILL_ONLY


def test_negotiate_decode_when_no_lane_wants_prefill():
    coordinator = _make_coordinator([FakeLane(running=2), FakeLane(running=1)])
    assert coordinator._negotiate_forced_mode() == TTSchedulingMode.DECODE_ONLY


def test_idle_step_propagates_finished_req_ids():
    # No lane has work, but one lane still has a finished request to report.
    lanes = [FakeLane(pending_finished={"done-0"}), FakeLane()]
    coordinator = _make_coordinator(lanes)

    output = coordinator.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert output.finished_req_ids == {"done-0"}
    # No lane wanted prefill, so the step is decode-only.
    assert coordinator._last_lane_metadata.is_decode is True
    # Every lane is scheduled (so each drains its own finished set).
    assert lanes[0].scheduled_modes == [TTSchedulingMode.DECODE_ONLY]
    assert lanes[1].scheduled_modes == [TTSchedulingMode.DECODE_ONLY]


def test_decode_fallback_when_forced_prefill_schedules_nothing():
    # Lane 0 has running decodes (and a finished req to report); lane 1 has a
    # queued request that forces prefill. Prefill schedules nothing, so the
    # coordinator must fall back to decode to make progress.
    lane0 = FakeLane(running=2, pending_finished={"done-0"})
    lane1 = FakeLane(waiting=1)
    coordinator = _make_coordinator([lane0, lane1])

    output = coordinator.schedule()

    # Fell back to decode: lane 0's two decodes are scheduled.
    assert output.total_num_scheduled_tokens == 2
    assert coordinator._last_lane_metadata.is_decode is True
    # Finished IDs drained during the discarded prefill pass are carried over.
    assert output.finished_req_ids == {"done-0"}
    # Lane 0 was scheduled once for prefill, then again for the decode fallback.
    assert lane0.scheduled_modes == [
        TTSchedulingMode.PREFILL_ONLY,
        TTSchedulingMode.DECODE_ONLY,
    ]


def test_no_fallback_when_no_running_requests():
    # Forced prefill schedules nothing and there are no running decodes
    # anywhere: nothing to fall back to, so the step stays prefill (empty).
    lane0 = FakeLane(waiting=1)
    lane1 = FakeLane(waiting=1)
    coordinator = _make_coordinator([lane0, lane1])

    output = coordinator.schedule()

    assert output.total_num_scheduled_tokens == 0
    assert coordinator._last_lane_metadata.is_decode is False
    # Only the prefill pass ran (no decode fallback).
    assert lane0.scheduled_modes == [TTSchedulingMode.PREFILL_ONLY]


def test_update_from_output_routes_and_merges_per_lane():
    lane0 = FakeLane()
    lane1 = FakeLane()
    lane0._eco = {0: EngineCoreOutputs(outputs=["a"])}
    lane1._eco = {0: EngineCoreOutputs(outputs=["b"], finished_requests={"x"})}
    coordinator = _make_coordinator([lane0, lane1])
    scheduler_output = coordinator.schedule()

    merged = coordinator.update_from_output(scheduler_output, model_runner_output=None)

    # Each lane received its own SchedulerOutput.
    assert lane0.update_calls == [scheduler_output._tt_step_state.lane_outputs[0]]
    assert lane1.update_calls == [scheduler_output._tt_step_state.lane_outputs[1]]
    # Per-client outputs concatenated and finished sets unioned.
    assert merged[0].outputs == ["a", "b"]
    assert merged[0].finished_requests == {"x"}
    # Stats disabled -> none attached.
    assert merged[0].scheduler_stats is None


def test_update_from_output_no_metadata_returns_empty():
    coordinator = _make_coordinator([FakeLane(), FakeLane()])
    assert coordinator.update_from_output(SchedulerOutput.make_empty(), None) == {}


def test_schedule_attaches_runner_step_plan_with_stable_rows():
    lane0 = FakeLane()
    lane1 = FakeLane()
    coordinator = _make_coordinator([lane0, lane1], per_lane_max=4)
    lane0.schedule = lambda: _scheduled_output(["a", "b"])
    lane1.schedule = lambda: _scheduled_output(["c"])
    coordinator._req_to_lane = {"a": 0, "b": 0, "c": 1}
    coordinator._assign_slot("a", 0)
    coordinator._assign_slot("b", 0)
    coordinator._assign_slot("c", 1)

    output = coordinator.schedule()
    plan = get_tt_step_plan(output)

    assert isinstance(plan, TTStepPlan)
    assert plan.is_decode is True
    assert plan.scheduled_rows == (0, 1, 4)
    assert plan.scheduled_req_ids == ("a", "b", "c")
    assert plan.input_rows == tuple(range(8))
    assert plan.batch_size_per_dp == (4, 4)
    assert plan.prefill_empty_slots is None


def test_prefill_step_plan_exposes_empty_slots_without_lane_metadata():
    lane0 = FakeLane()
    lane1 = FakeLane(waiting=1)
    coordinator = _make_coordinator([lane0, lane1], per_lane_max=4)
    lane0.schedule = SchedulerOutput.make_empty
    lane1.schedule = lambda: _scheduled_output(["a"])
    coordinator._req_to_lane = {"a": 1}

    output = coordinator.schedule()
    plan = get_tt_step_plan(output)

    assert isinstance(plan, TTStepPlan)
    assert plan.is_decode is False
    assert plan.scheduled_rows == (4,)
    assert plan.scheduled_req_ids == ("a",)
    assert plan.input_rows == (4,)
    assert plan.batch_size_per_dp == (0, 1)
    assert plan.prefill_empty_slots == (4,)
    assert not hasattr(output, "_tt_lane_step_metadata")


def test_per_lane_vllm_config_uses_per_lane_max_num_seqs():
    # Lanes must be constructed from a config whose max_num_seqs is the
    # *per-lane* cap, so the base scheduler derives max_num_running_reqs ==
    # per_lane at __init__ (Scheduler.__init__:
    # self.max_num_running_reqs = scheduler_config.max_num_seqs). Without this,
    # four lanes built from the global cap (32) would each believe they may run
    # the whole global batch, letting their combined running set reach 128 and
    # overflow the runner's merged persistent batch (req_index >= max_num_reqs).
    coordinator = TTLaneCoordinator.__new__(TTLaneCoordinator)
    coordinator._per_lane_max = 8
    global_config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=32))

    per_lane_config = coordinator._build_per_lane_vllm_config(global_config)

    # The lanes' config carries the per-lane cap...
    assert per_lane_config.scheduler_config.max_num_seqs == 8
    # ...while the shared global config the coordinator/runner read for KV and
    # model sizing is left untouched (copied, not aliased/mutated).
    assert global_config.scheduler_config.max_num_seqs == 32
    assert per_lane_config.scheduler_config is not global_config.scheduler_config
