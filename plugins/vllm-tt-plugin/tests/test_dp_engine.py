# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tests for gathered-DP mode negotiation."""

from types import SimpleNamespace

import vllm_tt_plugin.engine as engine_module
from vllm_tt_plugin.engine import TTDPEngineCoreProc
from vllm_tt_plugin.scheduler import TTSchedulingMode


def _core_with_scheduler(scheduler):
    core = TTDPEngineCoreProc.__new__(TTDPEngineCoreProc)
    core.scheduler = scheduler
    core.dp_group = object()
    core.dlog = lambda *args, **kwargs: None
    return core


def test_dp_negotiation_prefers_running_prefill_continuation(monkeypatch):
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
