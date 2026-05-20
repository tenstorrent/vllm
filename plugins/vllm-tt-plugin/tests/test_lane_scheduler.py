# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from vllm_tt_plugin.lane_scheduler import TTLaneCoordinator
from vllm_tt_plugin.scheduler import TTScheduler

from vllm.v1.core.sched.output import SchedulerOutput


def test_lane_scheduler_preserves_finished_only_cleanup(
    monkeypatch: pytest.MonkeyPatch,
):
    coordinator = TTLaneCoordinator.__new__(TTLaneCoordinator)
    coordinator.num_lanes = 4
    coordinator._last_lane_metadata = object()
    coordinator._refresh_lane_counts = lambda: None
    coordinator._lane_has_work = lambda lane: False

    sentinel = SchedulerOutput.make_empty()
    sentinel.finished_req_ids = {"finished-req"}
    monkeypatch.setattr(TTScheduler, "schedule", lambda self: sentinel)

    output = TTLaneCoordinator.schedule(coordinator)

    assert output is sentinel
    assert output.finished_req_ids == {"finished-req"}
    assert coordinator._last_lane_metadata is None
