# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only unit tests for the single-process lane-DP stable slot allocator.

These exercise the pure bookkeeping that keeps each request pinned to a stable
device slot for its lifetime (so the on-device per-slot seed RNG state is never
disturbed by admissions/finishes in the same lane). No device required.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm_tt_plugin.model_runner import TTModelRunner

# Unbound methods called against a lightweight stub ``self``.
_sync = TTModelRunner._sync_lane_slot_assignment
_free = TTModelRunner._free_finished_lane_slots
_slots_for = TTModelRunner._lane_slots_for
_scatter = TTModelRunner._scatter_rows_to_slots


def _runner(num_lanes: int, per_lane: int) -> SimpleNamespace:
    r = SimpleNamespace(
        tt_data_parallel_size=num_lanes,
        tt_per_lane_max_num_seqs=per_lane,
        _lane_slot_assignment=[{} for _ in range(num_lanes)],
    )
    # ``_sync_lane_slot_assignment`` delegates freeing to this method, so the
    # stub must expose it as a bound callable.
    r._free_finished_lane_slots = lambda fin, _r=r: _free(_r, fin)
    return r


class TestLaneSlotAssignment:
    def test_new_requests_get_lowest_free_slot(self):
        r = _runner(num_lanes=2, per_lane=4)
        _sync(r, [["a", "b"], ["c"]], set())
        assert _slots_for(r, 0, ["a", "b"]) == [0, 1]
        assert _slots_for(r, 1, ["c"]) == [0]

    def test_existing_requests_keep_slot_on_admission(self):
        r = _runner(num_lanes=1, per_lane=8)
        _sync(r, [["a", "b"]], set())
        # A new request is admitted; existing ones must not move.
        _sync(r, [["a", "b", "c"]], set())
        assert _slots_for(r, 0, ["a", "b", "c"]) == [0, 1, 2]

    def test_finished_slot_is_left_as_gap_then_reused(self):
        r = _runner(num_lanes=1, per_lane=8)
        _sync(r, [["a", "b", "c"]], set())  # slots 0,1,2
        # 'b' finishes: its slot 1 is freed but 'a' and 'c' do not move.
        _sync(r, [["a", "c"]], finished_req_ids={"b"})
        assert _slots_for(r, 0, ["a", "c"]) == [0, 2]
        # A new request reuses the lowest free slot (the gap at 1).
        _sync(r, [["a", "c", "d"]], set())
        assert _slots_for(r, 0, ["a", "c", "d"]) == [0, 2, 1]

    def test_reordered_request_ids_keep_their_slots(self):
        # The lane's request order can change step to step (e.g. global-batch
        # condense re-sorts), but each request must keep its assigned slot.
        r = _runner(num_lanes=1, per_lane=8)
        _sync(r, [["a", "b", "c"]], set())
        assert _slots_for(r, 0, ["c", "a", "b"]) == [2, 0, 1]

    def test_free_on_zero_token_drain_step_does_not_leak(self):
        # A finished request is reported one step after it completes; if the
        # lane has drained by then, that step schedules zero tokens and
        # ``execute_model_lanes`` returns before slot assignment runs. Freeing
        # must still happen via ``_free_finished_lane_slots`` so the slot is
        # reclaimed instead of leaking until the lane overflows capacity.
        r = _runner(num_lanes=1, per_lane=2)
        for i in range(10):
            req = f"req-{i}"
            _sync(r, [[req]], set())  # admit
            _free(r, {req})  # drained: finish reported on a zero-token step
        # No leak: every finished slot was reclaimed, so a fresh request still
        # fits even though far more than ``per_lane`` requests have come and
        # gone through this lane.
        _sync(r, [["fresh"]], set())
        assert _slots_for(r, 0, ["fresh"]) == [0]


class TestScatterRowsToSlots:
    def test_scatter_2d_rows_leaves_padded_gaps(self):
        rows = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
        out = _scatter(None, rows, slots=[0, 2], batch_size=4, pad_value=-1)
        assert out.tolist() == [[10, 11], [-1, -1], [20, 21], [-1, -1]]

    def test_scatter_1d_param_uses_default_for_gaps(self):
        vals = torch.tensor([0.7, 0.9], dtype=torch.float32)
        out = _scatter(None, vals, slots=[1, 3], batch_size=4, pad_value=1.0)
        # float32 round-trip: compare with tolerance, not against float64 literals.
        assert out.tolist() == pytest.approx([1.0, 0.7, 1.0, 0.9])

    def test_scatter_empty_is_all_padding(self):
        rows = torch.zeros((0, 1), dtype=torch.int32)
        out = _scatter(None, rows, slots=[], batch_size=3, pad_value=0)
        assert out.tolist() == [[0], [0], [0]]
