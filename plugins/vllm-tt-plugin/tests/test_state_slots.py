# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tests for per-request device state slots.

A persistent-batch ROW is not stable for a request: ``_update_states`` evicts a
running request the step does not schedule (every prefill step does) and re-adds it
at whatever row is free, and ``condense`` moves rows down when a request finishes.
Device state indexed by slot (Qwen3.6 GDN recurrent+conv, the per-slot seed RNG, the
decode trace's token/position buffers) does not follow, so
``_alloc_prefill_state_slots`` and ``_decode_state_slot_remap`` say where each
request's state is. No device execution: both are pure index bookkeeping, run here
against a fake runner.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm_tt_plugin.model_input import TTModelInput, TTSamplingParams
from vllm_tt_plugin.model_runner import TTModelRunner

SLOTS = 8


def _runner(slots=SLOTS):
    """Fake runner: the state-slot map, the live-request set and the slot capacity."""
    return SimpleNamespace(
        tt_per_lane_max_num_seqs=slots, _req_state_slot={}, requests={}
    )


def _prefill(runner, row_req_ids):
    out = TTModelRunner._alloc_prefill_state_slots(runner, list(row_req_ids))
    runner.requests.update(dict.fromkeys(row_req_ids))
    return out


def _decode(runner, row_req_ids):
    remap = TTModelRunner._decode_state_slot_remap(runner, list(row_req_ids))
    return None if remap is None else remap.tolist()


def _merge(runner, inputs):
    return TTModelRunner._merge_dp_prefill_slots(runner, inputs)


def _scheduler_output(*, preempted=None, scheduled=("KEEP",)):
    """Just the SchedulerOutput fields ``_update_states`` reads on a quiet step."""
    return SimpleNamespace(
        finished_req_ids=[],
        preempted_req_ids=preempted,
        free_encoder_mm_hashes=[],
        num_scheduled_tokens=dict.fromkeys(scheduled, 1),
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
    )


def _gather(state, remap):
    """What the device does with a remap: row ``i`` reads slot ``remap[i]``."""
    return list(state) if remap is None else [state[s] for s in remap]


def _assert_state_found(runner, state):
    """The invariant: a request's recorded slot is where its state actually sits."""
    for req_id in runner.requests:
        slot = runner._req_state_slot[req_id]
        assert state[slot] == req_id, (
            f"{req_id} thinks its state is in slot {slot}, which holds "
            f"{state[slot]!r} (device state: {state})"
        )


def test_state_follows_the_request_across_row_moves():
    """The full lifecycle: fresh prefill, eviction, return at a new row, re-prefill."""
    r = _runner()
    # Empty server: fresh slots equal fresh rows, so no state has to move.
    assert _prefill(r, ["A"]) == [0]
    assert _decode(r, ["A"]) is None  # identity -> consumers skip the gather

    # THE BUG. A is live but unscheduled, so vLLM gives its row 0 to a new request.
    # A prefill into slot 0 destroys A's recurrent state and A then emits garbage.
    incoming = [f"B{i}" for i in range(7)]
    slots = _prefill(r, incoming)
    assert 0 not in slots, f"prefill took A's live slot: {slots}"
    assert len(set(slots)) == 7 and all(0 < s < SLOTS for s in slots)

    # vLLM re-adds A at the first free row (7). The remap must fetch A's state from
    # the slot it sits in, and every other row from its own.
    rows = incoming + ["A"]
    remap = _decode(r, rows)
    assert remap is not None, "a returning request needs its state moved"
    assert len(remap) == SLOTS and sorted(remap) == list(range(SLOTS)), (
        "must be a permutation"
    )
    assert remap[7] == 0, f"row 7 (A) must read A's slot 0, got {remap[7]}"
    for row, s in enumerate(slots):
        assert remap[row] == s, (
            f"row {row} must read {rows[row]}'s slot {s}, got {remap[row]}"
        )
    # State now sits at each request's row, so the next step is free again.
    assert _decode(r, rows) is None

    # A preempted request is re-prefilled while it still owns its slot: it must keep
    # that slot, not move and leave the old one stranded.
    assert _prefill(r, ["B0"]) == [0], (
        "a re-prefilled live request must keep its own slot"
    )


def test_remap_carries_off_batch_state():
    """A non-identity remap permutes ALL slots, live off-batch holders' included."""
    r = _runner()
    state: list[str | None] = [None] * SLOTS
    for row, slot in enumerate(_prefill(r, ["A", "B"])):
        state[slot] = ["A", "B"][row]
    assert r._req_state_slot == {"A": 0, "B": 1}

    # Only B decodes; pulling it to row 0 pushes live off-batch A out of slot 0.
    remap = _decode(r, ["B"])
    assert remap is not None and remap[0] == 1, f"row 0 must read B's slot 1: {remap}"
    state = _gather(state, remap)
    assert state[0] == "B"
    assert state[1] == "A", "A's state was displaced by the gather"
    _assert_state_found(r, state)

    # A is rescheduled: its recorded slot must be the one it landed in.
    remap = _decode(r, ["B", "A"])
    state = _gather(state, remap)
    _assert_state_found(r, state)
    assert state[:2] == ["B", "A"], f"state must sit at each request's row: {state}"


def _decode_global_slots(local_slots_per_rank, stride=SLOTS):
    """The global slot space, computed the way gathered-DP DECODE computes it
    (``raw_remap + arange(world) * B``). Prefill has to agree with this side."""
    world = len(local_slots_per_rank)
    raw = torch.zeros((world, stride), dtype=torch.int32)
    for rank, slots in enumerate(local_slots_per_rank):
        raw[rank, : len(slots)] = torch.tensor(slots, dtype=torch.int32)
    offsets = torch.arange(world, dtype=torch.int32).unsqueeze(1) * stride
    globalised = raw + offsets
    return [
        int(globalised[rank, i])
        for rank, slots in enumerate(local_slots_per_rank)
        for i in range(len(slots))
    ]


def _sampling_params(rows):
    """Neutral per-row sampling tensors, the shape ``concat_dp_model_inputs`` cats."""
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


def _rank_input(slots, rows=None):
    """One DP rank's prefill input: ``rows`` rows placed at ``slots``."""
    rows = len(slots) if rows is None else rows
    return TTModelInput(
        input_tokens=torch.zeros((rows, 4), dtype=torch.int32),
        input_positions=np.zeros(rows, dtype=np.int32),
        prompt_lens=np.full(rows, 4, dtype=np.int32),
        block_tables=torch.zeros((rows, 1), dtype=torch.int32),
        block_tables_per_group=[torch.zeros((rows, 1), dtype=torch.int32)],
        block_tables_per_layer=None,
        unpadded_batch_size=rows,
        tt_sampling_params=_sampling_params(rows),
        multi_modal_kwargs={},
        perform_device_sampling=True,
        grammar_bitmask=[None],
        logitsprocs_list=[None],
        bad_words_token_ids_list=[{}],
        allowed_token_ids_mask_list=[None],
        generators_list=[{}],
        max_num_logprobs=[None],
        prefill_empty_slots=slots,
    )


def _concat_runner(slots=SLOTS):
    r = _runner(slots)
    r.max_num_blocks_per_req = 1
    r._num_kv_cache_groups = 1
    r.model_config = SimpleNamespace(is_multimodal_model=False)
    r._block_tables_per_layer = lambda per_group: None
    r._merge_dp_prefill_slots = lambda inputs: _merge(r, inputs)
    return r


def test_gathered_dp_prefill_slots_match_the_decode_offsets():
    """The merge helper lifts each rank's local slots into the decode global space."""
    per_rank = [[3], [], [0, 2]]  # rank 0 kept a live slot free; rank 1 idle
    inputs = [_rank_input([3]), None, _rank_input([0, 2])]
    assert _merge(_runner(), inputs) == _decode_global_slots(per_rank)

    # No rank allocated any (stateless model): stay None so submit_prefill keeps its
    # scheduling-order fallback rather than sending an empty list.
    assert _merge(_runner(), [None, None]) is None
    assert _merge(_runner(), [_rank_input(None, rows=1)]) is None


def test_merged_slots_must_line_up_with_the_merged_rows():
    """A rank contributing rows but no slots would silently shift every later rank's
    state into the wrong place."""
    with pytest.raises(AssertionError, match="rank 1 contributes 2 row"):
        _merge(_runner(), [_rank_input([0]), _rank_input(None, rows=2)])
    with pytest.raises(AssertionError, match="rank 0 contributes 3 row"):
        _merge(_runner(), [_rank_input([0, 1], rows=3)])


def test_concat_dp_prefill_carries_the_global_slots():
    """#462 was in the caller: it dropped the merged slots, so every rank but rank 0
    prefilled into rank 0's slots. Test where the bug was."""
    per_rank = [[3], [0, 2]]
    merged = TTModelRunner.concat_dp_model_inputs(
        _concat_runner(),
        [_rank_input(per_rank[0]), _rank_input(per_rank[1])],
        is_decode=False,
        max_blocks_decode_batch=None,
        any_structured_inputs=False,
    )

    assert merged.prefill_empty_slots == _decode_global_slots(per_rank)
    assert len(merged.prefill_empty_slots) == merged.input_tokens.shape[0], (
        "one slot per merged row, in row order"
    )


def test_preemption_releases_its_state_slot():
    """A preempted request re-prefills from scratch, so its slot must be released."""
    r = _runner()
    r.encoder_cache = {}
    r._decode_layout_changed_since_last_decode = False
    r.input_batch = SimpleNamespace(
        req_id_to_index={"KEEP": 0}, refresh_logitsprocs=lambda: None
    )
    r._req_state_slot.update({"P": 0, "KEEP": 1})
    r.requests.update(dict.fromkeys(["P", "KEEP"]))

    TTModelRunner._update_states(r, _scheduler_output(preempted={"P"}))

    assert r._req_state_slot == {"KEEP": 1}, "only the preempted request releases"
    assert "P" in r.requests, "the request is still live, it just re-prefills"

    # Which is the point: the freed slot is available to the incoming prefill.
    assert _prefill(r, ["NEW"]) == [0]


def test_slot_exhaustion_fails_instead_of_guessing():
    """Exhaustion means the map has stopped describing the device, and it is the only
    record of slot ownership. Guessing returns plausible, wrong text."""
    r = _runner(slots=2)
    _prefill(r, ["A", "B"])  # both slots held by live requests
    with pytest.raises(AssertionError, match="no free device state slot"):
        _prefill(r, ["C"])

    # Over-capacity is the scheduler's decision to make, not this function's.
    with pytest.raises(AssertionError, match="exceed the 2 device state slots"):
        _prefill(_runner(slots=2), ["A", "B", "C"])


def test_more_decode_rows_than_slots_raises():
    """Truncating to the slot width would silently drop C's state instead of saying
    the batch cannot be described."""
    r = _runner(slots=2)
    r._req_state_slot.update({"A": 1, "B": 0})
    assert _decode(r, ["A", "B"]) == [1, 0], "at capacity is still fine"
    with pytest.raises(AssertionError, match="3 decode row"):
        _decode(r, ["A", "B", "C"])


def test_a_clean_map_is_silent_and_a_broken_one_raises():
    """A duplicated source slot would make a device gather read one slot twice.
    Refusing sends no gather at all, which corrupts every off-row request instead."""
    # Steady state: everyone already sits at their own row, so nothing moves.
    r = _runner()
    _prefill(r, ["X", "Y"])
    assert _decode(r, ["X", "Y"]) is None

    # A duplicate is an impossible state, and Z's entry says who else is affected.
    r._req_state_slot.update({"X": 3, "Y": 3, "Z": 5})
    with pytest.raises(AssertionError, match="not a permutation") as exc:
        _decode(r, ["X", "Y"])
    assert "duplicated=[3]" in str(exc.value)
    assert "'Z': 5" in str(exc.value), "the whole map is the diagnostic"
    # It fails before writing, so off-batch entries like Z are left alone.
    assert r._req_state_slot == {"X": 3, "Y": 3, "Z": 5}


def test_a_decoding_request_without_a_slot_raises():
    """Inventing ownership records a second request at a slot a live one owns. It is
    the hole both corruptions travel through."""
    r = _runner()
    _prefill(r, ["A"])
    with pytest.raises(AssertionError, match="'GHOST' has no device state slot"):
        _decode(r, ["A", "GHOST"])
