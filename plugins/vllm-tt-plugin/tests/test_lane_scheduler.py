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
    TTLaneCoordinator,
    TTStepPlan,
    get_tt_step_plan,
    merge_lane_scheduler_outputs,
)
from vllm_tt_plugin.scheduler import TTSchedulingMode

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
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
    assert get_tt_step_plan(output).is_decode is True
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
    assert get_tt_step_plan(output).is_decode is True
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
    assert get_tt_step_plan(output).is_decode is False
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


# --------------------------------------------------------------------------
# merge_lane_scheduler_outputs (per-field stitching of non-empty lane outputs)
# --------------------------------------------------------------------------


def test_merge_combines_nonempty_lane_outputs():
    # Two lanes each scheduling a cached (decode) request, plus new requests,
    # preemption/invalid-spec reported by only one lane, and per-lane common
    # prefix block counts.
    lane0 = SchedulerOutput.make_empty()
    lane0.scheduled_new_reqs = ["new-a"]
    lane0.scheduled_cached_reqs = CachedRequestData(
        req_ids=["a"],
        resumed_req_ids={"a"},
        new_token_ids=[[11]],
        all_token_ids={"a": [1, 11]},
        new_block_ids=[([0],)],
        num_computed_tokens=[1],
        num_output_tokens=[1],
    )
    lane0.num_scheduled_tokens = {"a": 1}
    lane0.total_num_scheduled_tokens = 1
    lane0.num_common_prefix_blocks = [2, 1]
    lane0.finished_req_ids = {"fin-0"}
    lane0.free_encoder_mm_hashes = ["h0"]
    lane0.preempted_req_ids = {"pre-0"}
    lane0.num_invalid_spec_tokens = {"a": 3}

    lane1 = SchedulerOutput.make_empty()
    lane1.scheduled_new_reqs = ["new-b", "new-c"]
    lane1.scheduled_cached_reqs = CachedRequestData(
        req_ids=["b"],
        resumed_req_ids=set(),
        new_token_ids=[[22]],
        all_token_ids={"b": [2, 22]},
        new_block_ids=[([0],)],
        num_computed_tokens=[5],
        num_output_tokens=[2],
    )
    lane1.num_scheduled_tokens = {"b": 1}
    lane1.total_num_scheduled_tokens = 1
    lane1.num_common_prefix_blocks = [1, 3]
    lane1.finished_req_ids = {"fin-1"}
    lane1.free_encoder_mm_hashes = ["h1"]
    # lane1 reports no preemption and no invalid spec tokens.

    merged = merge_lane_scheduler_outputs([lane0, lane1])

    # New requests concatenated in lane order.
    assert merged.scheduled_new_reqs == ["new-a", "new-b", "new-c"]
    # Cached struct-of-arrays concatenated field-by-field in lane order.
    c = merged.scheduled_cached_reqs
    assert c.req_ids == ["a", "b"]
    assert c.new_token_ids == [[11], [22]]
    assert c.new_block_ids == [([0],), ([0],)]
    assert c.num_computed_tokens == [1, 5]
    assert c.num_output_tokens == [1, 2]
    assert c.all_token_ids == {"a": [1, 11], "b": [2, 22]}
    assert c.resumed_req_ids == {"a"}
    # Scheduled-token dicts merged; total recomputed from the merged dict.
    assert merged.num_scheduled_tokens == {"a": 1, "b": 1}
    assert merged.total_num_scheduled_tokens == 2
    # Elementwise max across the (equal-length) per-lane prefix-block lists.
    assert merged.num_common_prefix_blocks == [2, 3]
    # Finished unioned, free-encoder hashes extended.
    assert merged.finished_req_ids == {"fin-0", "fin-1"}
    assert merged.free_encoder_mm_hashes == ["h0", "h1"]
    # Optional fields reported by exactly one lane survive the merge.
    assert merged.preempted_req_ids == {"pre-0"}
    assert merged.num_invalid_spec_tokens == {"a": 3}


def test_merge_preserves_none_for_absent_optional_fields():
    # No lane reports preemption or invalid spec tokens: the merged output keeps
    # None (the base output's "absent" representation), not an empty set/dict.
    lane0 = SchedulerOutput.make_empty()
    lane0.num_scheduled_tokens = {"a": 1}
    lane0.total_num_scheduled_tokens = 1
    lane1 = SchedulerOutput.make_empty()

    merged = merge_lane_scheduler_outputs([lane0, lane1])

    assert merged.preempted_req_ids is None
    assert merged.num_invalid_spec_tokens is None


def test_merge_raises_on_mismatched_common_prefix_block_counts():
    # Lanes share one kv_cache_config, so their num_common_prefix_blocks lists
    # must be equal length (one entry per KV cache group). A mismatch is an
    # upstream bug; the merge must raise, not silently drop the extra groups.
    lane0 = SchedulerOutput.make_empty()
    lane0.num_common_prefix_blocks = [2, 1]
    lane1 = SchedulerOutput.make_empty()
    lane1.num_common_prefix_blocks = [1, 3, 5]

    try:
        merge_lane_scheduler_outputs([lane0, lane1])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on mismatched group counts")


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
