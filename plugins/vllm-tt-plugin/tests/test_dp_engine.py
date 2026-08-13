# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tests for gathered-DP mode negotiation and step orchestration."""

from concurrent.futures import Future
from types import SimpleNamespace

import torch
import vllm_tt_plugin.engine as engine_module
from vllm_tt_plugin.engine import DPGatherHandle, TTDPEngineCoreProc
from vllm_tt_plugin.scheduler import TTSchedulingMode


def _core_with_scheduler(scheduler):
    core = TTDPEngineCoreProc.__new__(TTDPEngineCoreProc)
    core.scheduler = scheduler
    core.dp_group = object()
    core.dlog = lambda *args, **kwargs: None
    return core


def _handle(**overrides):
    fields = dict(
        future=Future(),
        scheduler_output=SimpleNamespace(total_num_scheduled_tokens=1),
        local_has_requests=True,
        is_decode=True,
        overlap_ok=False,
        any_needs_logprobs=False,
        intermediate_prefill_mask=None,
        req_ids=[],
        req_id_to_index={},
    )
    fields.update(overrides)
    return DPGatherHandle(**fields)


def _step_core(monkeypatch, *, overlap_ok, in_flight):
    """A core whose ``step_dp_with_batch_queue`` collaborators are recorded."""
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=4, pending_structured_output_tokens=False
    )
    scheduler = SimpleNamespace(
        has_requests=lambda: True,
        schedule=lambda: scheduler_output,
        get_grammar_bitmask=lambda _so: None,
        update_from_output=lambda _so, _out: {},
    )
    core = _core_with_scheduler(scheduler)
    core.batch_queue = object()
    core.is_ec_producer = False
    core._dp_in_flight = in_flight
    core._process_aborts_queue = lambda: events.append("aborts")

    events: list = []
    submits: list = []

    monkeypatch.setattr(
        core, "_dp_any_rank_has_scheduler_requests", lambda: True, raising=False
    )
    monkeypatch.setattr(
        core,
        "_dp_negotiate_forced_mode",
        lambda: TTSchedulingMode.DECODE_ONLY,
        raising=False,
    )
    monkeypatch.setattr(
        core,
        "_dp_can_attempt_steady_decode_from_scheduler",
        lambda _so, _go: overlap_ok,
        raising=False,
    )

    def fake_submit(_so, _go, *, overlap_ok, outstanding):
        events.append("submit")
        submits.append(outstanding)
        return _handle()

    def fake_finalize(handle):
        events.append("finalize")
        return SimpleNamespace(handle=handle)

    monkeypatch.setattr(core, "dp_gather_submit", fake_submit, raising=False)
    monkeypatch.setattr(core, "dp_gather_finalize", fake_finalize, raising=False)
    return core, events, submits


def test_step_hands_invalidations_to_the_still_outstanding_submission(monkeypatch):
    """Overlap applies the previous result after this step's states are updated.

    So the requests this step finished or resumed belong to the outstanding
    handle. Giving them to the submission created here would reject a token that
    submission legitimately produces for its own resumed rows.
    """
    in_flight = _handle()
    core, events, submits = _step_core(
        monkeypatch, overlap_ok=True, in_flight=in_flight
    )

    core.step_dp_with_batch_queue()

    assert events == ["submit", "aborts", "finalize"]
    assert submits == [in_flight]


def test_step_drops_invalidations_once_the_previous_result_is_applied(monkeypatch):
    """Without overlap the previous result is applied in scheduler order.

    Nothing is outstanding by the time this step's states are updated, so the
    invalidations must be discarded rather than held for a later step.
    """
    in_flight = _handle()
    core, events, submits = _step_core(
        monkeypatch, overlap_ok=False, in_flight=in_flight
    )

    core.step_dp_with_batch_queue()

    assert events == ["aborts", "finalize", "submit"]
    assert submits == [None]


def test_gather_submit_moves_invalidations_off_the_new_handle(monkeypatch):
    """The runner drains the set on every DP step, including an idle one.

    A rank that scheduled nothing still notes finished requests, and leaving
    them on the runner would filter an unrelated later step's rows.
    """
    core = _core_with_scheduler(SimpleNamespace())
    core.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_rank_local=0, data_parallel_size=1
        )
    )
    core.dp_rank = 0
    core.dp_device_ranks = [0]
    core._dp_gather_forced_mode = TTSchedulingMode.PREFILL_ONLY
    core.model_executor = SimpleNamespace(
        collective_rpc=lambda *args, **kwargs: [
            (None, 0, 0, 0, 0, 1, 0, None, [], {}, {"finished"})
        ]
    )

    monkeypatch.setattr(
        engine_module.dist, "all_reduce", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        engine_module.dist, "gather_object", lambda *a, **k: None, raising=False
    )
    monkeypatch.setattr(
        engine_module, "get_tt_per_lane_max_num_seqs", lambda _cfg: 2, raising=False
    )

    outstanding = _handle()
    new_handle = core.dp_gather_submit(None, None, outstanding=outstanding)

    assert outstanding.invalidated_req_ids == {"finished"}
    assert new_handle.invalidated_req_ids == set()


def test_finalize_filters_rows_with_the_handles_own_invalidations(monkeypatch):
    """The set reaches the runner through the handle, not through runner state."""
    core = _core_with_scheduler(SimpleNamespace())
    core.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1)
    )
    core.dp_rank = 0
    applied: list = []
    core.model_executor = SimpleNamespace(
        collective_rpc=lambda _name, args=None: (applied.append(args), [None])[1]
    )
    monkeypatch.setattr(
        engine_module.dist, "scatter", lambda *a, **k: None, raising=False
    )

    future: Future = Future()
    future.set_result((torch.zeros((1, 2, 1), dtype=torch.int32), [None]))
    handle = _handle(future=future, invalidated_req_ids={"resumed"})

    core.dp_gather_finalize(handle)

    assert applied[0][-1] == {"resumed"}


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


def test_contract_version_is_read_once_and_reduced_permissively(monkeypatch):
    """Ranks holding no model report -1, so the reduction has to be MAX.

    Reading it once per process is what keeps every rank on one value: only
    ``data_parallel_rank_local == 0`` loads a model, so a per-rank read of the
    adapter attribute raises everywhere else.
    """
    core = _core_with_scheduler(SimpleNamespace())
    core.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_rank_local=1, data_parallel_size=2
        )
    )
    core.dp_rank = 1
    core.dp_device_ranks = [0]
    core._dp_gather_forced_mode = TTSchedulingMode.DECODE_ONLY
    core._dp_local_contract_version = None

    rpc_names: list[str] = []

    def collective_rpc(name, args=None, kwargs=None, non_block=False):
        rpc_names.append(name)
        if name == "build_dp_model_input":
            return [(None, 0, 0, 0, 0, 1, 0, None, [], {}, set())]
        if name == "decode_input_update_contract_version":
            return [-1]
        if name == "build_dp_decode_gather_input":
            return [
                {
                    "int_inputs": torch.zeros(2, dtype=torch.int32),
                    "float_inputs": torch.zeros(2),
                }
            ]
        return [None]

    core.model_executor = SimpleNamespace(collective_rpc=collective_rpc)

    reduced: list[torch.Tensor] = []

    def all_reduce(tensor, *, op, group):
        assert op == engine_module.dist.ReduceOp.MAX
        reduced.append(tensor.clone())
        tensor[6] = 1  # a device rank agreed on version 1

    monkeypatch.setattr(engine_module.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(engine_module.dist, "gather", lambda *a, **k: None)
    monkeypatch.setattr(engine_module.dist, "recv", lambda *a, **k: None)
    monkeypatch.setattr(
        engine_module, "get_tt_per_lane_max_num_seqs", lambda _cfg: 2, raising=False
    )

    core.dp_gather_submit(None, None)
    core.dp_gather_submit(None, None)

    assert reduced[0][6].item() == -1
    assert rpc_names.count("decode_input_update_contract_version") == 1
