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

from vllm_tt_plugin import model_runner as model_runner_module
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


def test_gathered_dp_prefill_slots_match_the_decode_offsets():
    """Gathered DP decode offsets each rank's remap by ``rank * stride``; prefill has
    to place state in that same global space or the two disagree about where it is."""
    r = _runner()
    ranks = [
        SimpleNamespace(prefill_empty_slots=[3]),  # rank 0 kept a live slot free
        None,  # nothing scheduled
        SimpleNamespace(prefill_empty_slots=[0, 2]),
    ]
    assert _merge(r, ranks) == [3, 2 * SLOTS + 0, 2 * SLOTS + 2]

    # Scheduling order is what #454 removed: rank 2's rows are not slots 0 and 1.
    assert _merge(r, ranks) != [0, 2 * SLOTS + 0, 2 * SLOTS + 1]

    # No rank allocated any (stateless model): stay None so submit_prefill keeps its
    # scheduling-order fallback rather than sending an empty list.
    assert _merge(r, [None, None]) is None
    assert _merge(r, [SimpleNamespace(prefill_empty_slots=None)]) is None


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


def test_slot_exhaustion_evicts_instead_of_killing_the_engine(monkeypatch):
    """Full slots plus a prefill is legitimate: it must not assert in the hot path."""
    warned: list[str] = []
    monkeypatch.setattr(
        model_runner_module.logger,
        "warning",
        lambda msg, *a, **k: warned.append(str(msg)),
    )

    r = _runner(slots=2)
    _prefill(r, ["A", "B"])  # both slots held by live requests
    slots = _prefill(r, ["C"])

    assert slots == [1], f"a slot must still be allocated, got {slots}"
    assert any("slots exhausted" in w for w in warned), warned
    # The victim's entry goes with its slot, so the map stays a bijection.
    assert r._req_state_slot == {"A": 0, "C": 1}


def test_capacity_and_slot_width_are_enforced():
    """Over-capacity prefills stay in range; rows past the slot width are dropped."""
    r = _runner(slots=2)
    slots = _prefill(r, ["A", "B", "C"])
    assert all(0 <= s < 2 for s in slots), f"slots must stay in range: {slots}"

    # Rows beyond the slot width are truncated, so an over-long batch still yields a
    # permutation of the real slots instead of an out-of-range source index.
    r = _runner(slots=2)
    r._req_state_slot.update({"A": 1, "B": 0})
    assert _decode(r, ["A", "B", "C"]) == [1, 0]


def test_non_permutation_is_refused_and_a_clean_map_is_silent(monkeypatch):
    """A duplicated source slot would make a device gather read one slot twice.

    The warning is captured by replacing the module logger; the plugin's logger does not
    propagate to pytest's root handler.
    """
    warned: list[str] = []
    monkeypatch.setattr(
        model_runner_module.logger,
        "warning",
        lambda msg, *a, **k: warned.append(str(msg)),
    )

    # A request the allocator never saw is assumed to sit at its own row, which keeps
    # the remap the identity instead of guessing. Nothing wrong, so nothing logged.
    r = _runner()
    assert _decode(r, ["X", "Y"]) is None
    assert not warned, f"the clean path must not warn: {warned}"

    # Skip the move -- one incoherent response beats an OOB device read -- and say so.
    r._req_state_slot.update({"X": 3, "Y": 3, "Z": 5})
    assert _decode(r, ["X", "Y"]) is None
    assert any("not a permutation" in w for w in warned), warned
    # Nothing moved, so nothing is rewritten -- rewriting the rows would strand Z.
    assert r._req_state_slot == {"X": 3, "Y": 3, "Z": 5}
