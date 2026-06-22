# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only unit tests for ``TTLaneInputBatch`` (single-process lane-DP).

Covers the lane-chunked layout the batch owns on behalf of the model runner:
  1. Placement: requests land at a stable lane-local slot (row = lane*per_lane
     + slot); existing requests never move; freed rows become reusable gaps;
     condense is a no-op.
  2. Merged host sampling: one ``SamplingMetadata`` over the whole slot batch
     samples each live request exactly as a per-request batch-of-1 reference,
     across penalties / logit_bias / min_tokens / min_p / allowed_token_ids /
     bad_words and a CUSTOM logits processor -- including the case where a slot
     is freed and reused within the same step (the builtin-logitsproc
     remove+re-add reconciliation).

No device / ttnn execution required.
"""

from types import SimpleNamespace

import torch
import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.input_batch import InputBatch, TTLaneInputBatch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor, build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState

VOCAB = 64
BLOCK = 16
MAX_MODEL_LEN = 256


class FirstPromptTokenBoost(AdapterLogitsProcessor):
    """Custom logits processor: boost the token equal to the request's first
    prompt token. Intrinsic to the request, so it is batching-invariant."""

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params):
        def boost(prompt_ids, output_ids, logits):
            if prompt_ids:
                logits[prompt_ids[0]] = logits[prompt_ids[0]] + 50.0
            return logits

        return boost


def _make_logitsprocs(max_num_reqs, with_custom=True):
    cfg = SimpleNamespace(
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_reqs),
    )
    return build_logitsprocs(
        cfg,
        torch.device("cpu"),
        is_pin_memory=False,
        is_pooling_model=False,
        custom_logitsprocs=[FirstPromptTokenBoost] if with_custom else [],
    )


def _make_req(req_id, prompt, output, sp_kwargs, seed=None):
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=None,
        sampling_params=SamplingParams(**sp_kwargs),
        generator=gen,
        block_ids=([0],),
        num_computed_tokens=len(prompt),
        output_token_ids=list(output),
    )


def _lane_batch(num_lanes, per_lane, with_custom=True):
    return TTLaneInputBatch(
        num_lanes=num_lanes,
        per_lane=per_lane,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * num_lanes * per_lane,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
        logitsprocs=_make_logitsprocs(num_lanes * per_lane, with_custom=with_custom),
    )


def _add_to_lane(batch, req, lane):
    """Place ``req`` at the lowest free row in ``lane``'s chunk.

    Slot allocation (which lane, which free row) is the coordinator's job
    (see ``TTLaneCoordinator``). This helper stands in for that policy so the
    batch's placement behavior can be tested in isolation, driving the batch
    through its scheduler-owned entry point ``add_request_to_row``.
    """
    base = lane * batch.per_lane
    for slot in range(batch.per_lane):
        row = base + slot
        if batch._req_ids[row] is None:
            return batch.add_request_to_row(req, row)
    raise ValueError(f"lane {lane} has no free slot (capacity {batch.per_lane})")


# --------------------------------------------------------------------------
# Placement / stable slots
# --------------------------------------------------------------------------


def test_requests_land_in_their_lane_chunk():
    b = _lane_batch(num_lanes=2, per_lane=4)  # rows 0..3 lane0, 4..7 lane1
    r0 = _add_to_lane(b, _make_req("a", [1], [], dict(temperature=0.0)), lane=0)
    r1 = _add_to_lane(b, _make_req("b", [1], [], dict(temperature=0.0)), lane=1)
    r2 = _add_to_lane(b, _make_req("c", [1], [], dict(temperature=0.0)), lane=0)
    assert (r0, r2) == (0, 1)  # lane 0 chunk, lowest free slots
    assert r1 == 4  # lane 1 chunk base
    assert b.occupied_rows() == [0, 1, 4]
    assert b.lane_of("b") == 1


def test_existing_requests_keep_row_on_admission_and_removal():
    b = _lane_batch(num_lanes=1, per_lane=8)
    rows = {
        rid: _add_to_lane(b, _make_req(rid, [1], [], dict(temperature=0.0)), 0)
        for rid in ("a", "b", "c")
    }
    assert rows == {"a": 0, "b": 1, "c": 2}
    # Remove the middle request: its row becomes a gap; others do not move.
    assert b.remove_request("b") == 1
    assert b.req_id_to_index["a"] == 0 and b.req_id_to_index["c"] == 2
    assert b.occupied_rows() == [0, 2]
    # A new request reuses the lowest free slot (the gap at row 1).
    assert _add_to_lane(b, _make_req("d", [1], [], dict(temperature=0.0)), 0) == 1


def test_condense_is_noop():
    b = _lane_batch(num_lanes=1, per_lane=4)
    _add_to_lane(b, _make_req("a", [1], [], dict(temperature=0.0)), 0)
    _add_to_lane(b, _make_req("b", [1], [], dict(temperature=0.0)), 0)
    b.remove_request("a")  # gap at row 0
    b.condense([0])  # must not move "b" down into row 0
    assert b.req_id_to_index["b"] == 1


# --------------------------------------------------------------------------
# State update via apply_step_plan (moved here from the lane executor)
# --------------------------------------------------------------------------


def _new_req(req_id, prompt):
    """A scheduled_new_reqs entry (the fields build_cached_request_state reads)."""
    return SimpleNamespace(
        req_id=req_id,
        sampling_params=SamplingParams(temperature=0.0),
        prompt_token_ids=list(prompt),
        mm_features=None,
        num_computed_tokens=len(prompt),
        block_ids=([0],),
        lora_request=None,
        prompt_embeds=None,
    )


def _step_output(new_reqs=(), finished=(), plan_rows=None):
    from vllm.v1.core.sched.output import SchedulerOutput

    out = SchedulerOutput.make_empty()
    out.scheduled_new_reqs = list(new_reqs)
    out.finished_req_ids = set(finished)
    return out


def test_apply_step_plan_places_new_requests_and_reports_layout_change():
    from vllm_tt_plugin.lane_scheduler import TTStepPlan

    b = _lane_batch(num_lanes=2, per_lane=2)  # rows 0,1 lane0; 2,3 lane1
    requests: dict = {}
    out = _step_output(new_reqs=[_new_req("a", [1, 2]), _new_req("b", [3])])
    plan = TTStepPlan(
        is_decode=False,
        capacity=4,
        scheduled_req_ids=("a", "b"),
        scheduled_rows=(0, 2),
        input_rows=(0, 2),
        req_id_to_row={"a": 0, "b": 2},
        batch_size_per_dp=(1, 1),
        prefill_empty_slots=(0, 2),
    )

    changed = b.apply_step_plan(out, plan, requests, encoder_cache={})

    assert changed is True
    assert set(requests) == {"a", "b"}
    assert b.req_id_to_index == {"a": 0, "b": 2}
    assert b.occupied_rows() == [0, 2]


def test_apply_step_plan_finished_request_releases_slot():
    from vllm_tt_plugin.lane_scheduler import TTStepPlan

    b = _lane_batch(num_lanes=1, per_lane=4)
    requests: dict = {}
    empty_plan = TTStepPlan(
        is_decode=False,
        capacity=4,
        scheduled_req_ids=("a",),
        scheduled_rows=(0,),
        input_rows=(0,),
        req_id_to_row={"a": 0},
        batch_size_per_dp=(1,),
        prefill_empty_slots=(0,),
    )
    b.apply_step_plan(
        _step_output(new_reqs=[_new_req("a", [1])]), empty_plan, requests, {}
    )
    assert "a" in requests and b.occupied_rows() == [0]

    # A later step finishes "a": its row is freed and the request map cleaned.
    changed = b.apply_step_plan(
        _step_output(finished=["a"]),
        TTStepPlan(
            is_decode=True,
            capacity=4,
            scheduled_req_ids=(),
            scheduled_rows=(),
            input_rows=(),
            req_id_to_row={},
            batch_size_per_dp=(0,),
            prefill_empty_slots=None,
        ),
        requests,
        {},
    )
    assert changed is True
    assert "a" not in requests
    assert b.occupied_rows() == []


def test_lane_full_raises():
    b = _lane_batch(num_lanes=1, per_lane=2)
    _add_to_lane(b, _make_req("a", [1], [], dict(temperature=0.0)), 0)
    _add_to_lane(b, _make_req("b", [1], [], dict(temperature=0.0)), 0)
    try:
        _add_to_lane(b, _make_req("c", [1], [], dict(temperature=0.0)), 0)
    except ValueError as e:
        assert "no free slot" in str(e)
    else:
        raise AssertionError("expected ValueError on full lane")


# --------------------------------------------------------------------------
# Merged host sampling == per-request reference
# --------------------------------------------------------------------------


def _ref_sampling_metadata(batch, n):
    """Reference builder over a front-packed batch-of-n (mirrors the merged
    builder but on a plain front-packed InputBatch)."""
    s = batch.sampling
    temperature = s.temperature[:n]
    all_greedy = bool((temperature == 0.0).all())
    all_random = bool((temperature != 0.0).all())
    presence, frequency, repetition = (
        s.presence_penalty[:n],
        s.frequency_penalty[:n],
        s.repetition_penalty[:n],
    )
    no_penalties = bool(
        (presence == 0.0).all()
        and (frequency == 0.0).all()
        and (repetition == 1.0).all()
    )
    if not no_penalties:
        prompt_token_ids = batch.make_prompt_token_ids_tensor().to(torch.int64)
        prompt_token_ids = prompt_token_ids.masked_fill(prompt_token_ids == -1, VOCAB)
        out = batch.make_output_token_ids_tensor()
        output_token_ids = [[t for t in row.tolist() if t != -1] for row in out]
    else:
        prompt_token_ids = None
        output_token_ids = [[] for _ in range(n)]
    allowed = s.allowed_token_ids_mask
    if allowed is not None:
        allowed = allowed[:n]
    return SamplingMetadata(
        temperature=temperature if not all_greedy else None,
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=s.top_p[:n],
        top_k=s.top_k[:n],
        generators=dict(s.generators),
        max_num_logprobs=batch.max_num_logprobs,
        no_penalties=no_penalties,
        prompt_token_ids=prompt_token_ids,
        frequency_penalties=frequency,
        presence_penalties=presence,
        repetition_penalties=repetition,
        output_token_ids=output_token_ids,
        allowed_token_ids_mask=allowed,
        bad_words_token_ids=dict(s.bad_words_token_ids),
        logitsprocs=s.logitsprocs,
    )


def _plain_batch_of_one(req, with_custom=True):
    b = InputBatch(
        max_num_reqs=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
        logitsprocs=_make_logitsprocs(1, with_custom=with_custom),
    )
    b.add_request(req)
    b.refresh_logitsprocs()
    return b


def _feature_specs():
    # Heterogeneous greedy requests exercising every host sampling feature.
    return [
        dict(
            req_id="r0",
            prompt=[5, 1, 2],
            output=[5, 5, 7],
            sp=dict(temperature=0.0, presence_penalty=1.5),
        ),
        dict(
            req_id="r1",
            prompt=[9, 3],
            output=[],
            sp=dict(temperature=0.0, logit_bias={10: 80.0}),
        ),
        dict(
            req_id="r2",
            prompt=[2, 2, 2],
            output=[2, 2],
            sp=dict(temperature=0.0, repetition_penalty=2.0, frequency_penalty=1.0),
        ),
        dict(
            req_id="r3",
            prompt=[40],
            output=[],
            sp=dict(temperature=0.0, min_p=0.2, allowed_token_ids=[11, 12, 13]),
        ),
        dict(
            req_id="r4",
            prompt=[7, 8],
            output=[15],
            sp=dict(temperature=0.0, max_tokens=64, min_tokens=20),
        ),
        dict(req_id="r5", prompt=[33, 1], output=[], sp=dict(temperature=0.0)),
    ]


def test_merged_lane_sampling_equals_per_request():
    # Place 6 heterogeneous requests across 2 lanes of 4 (rows 0,1,2 and 4,5,6;
    # rows 3 and 7 are gaps), then sample the whole slot batch in one call.
    torch.manual_seed(1234)
    specs = _feature_specs()
    placement = [(0, "r0"), (0, "r1"), (0, "r2"), (1, "r3"), (1, "r4"), (1, "r5")]
    spec_by_id = {s["req_id"]: s for s in specs}

    batch = _lane_batch(num_lanes=2, per_lane=4)
    rows = {}
    for lane, rid in placement:
        s = spec_by_id[rid]
        rows[rid] = _add_to_lane(
            batch, _make_req(rid, s["prompt"], s["output"], s["sp"]), lane
        )
    batch.refresh_logitsprocs()

    n_slots = batch.max_num_reqs
    logits = torch.randn(n_slots, VOCAB)
    sampler = Sampler()

    merged = (
        sampler(
            logits=logits.clone(),
            sampling_metadata=batch.build_merged_sampling_metadata(),
        )
        .sampled_token_ids.reshape(-1)
        .tolist()
    )

    # Each live request's merged token must equal its batch-of-1 reference at
    # the same logits row.
    for rid, row in rows.items():
        s = spec_by_id[rid]
        ref_batch = _plain_batch_of_one(
            _make_req(rid, s["prompt"], s["output"], s["sp"])
        )
        ref = (
            sampler(
                logits=logits[row : row + 1].clone(),
                sampling_metadata=_ref_sampling_metadata(ref_batch, 1),
            )
            .sampled_token_ids.reshape(-1)
            .tolist()[0]
        )
        assert merged[row] == ref, f"{rid}@row{row}: merged={merged[row]} ref={ref}"


def test_merged_sampling_correct_after_free_and_reuse_same_step():
    # Free a min_p slot and reuse it in the SAME step. Without reconciling the
    # logitsproc remove+re-add, the reused request's min_p would be cleared and
    # it would sample differently from its reference.
    torch.manual_seed(7)
    batch = _lane_batch(num_lanes=1, per_lane=4, with_custom=False)
    a = _make_req("a", [1], [], dict(temperature=0.0, min_p=0.5))
    keep = _make_req("b", [2], [], dict(temperature=0.0, min_p=0.3))
    _add_to_lane(batch, a, 0)  # row 0
    _add_to_lane(batch, keep, 0)  # row 1
    batch.refresh_logitsprocs()

    # Same step: remove "a" (frees row 0), admit "c" (min_p) -> reuses row 0.
    batch.remove_request("a")
    c = _make_req("c", [3], [], dict(temperature=0.0, min_p=0.7))
    row_c = _add_to_lane(batch, c, 0)
    assert row_c == 0  # reused the freed gap
    batch.refresh_logitsprocs()

    logits = torch.randn(batch.max_num_reqs, VOCAB)
    sampler = Sampler()
    merged = (
        sampler(
            logits=logits.clone(),
            sampling_metadata=batch.build_merged_sampling_metadata(),
        )
        .sampled_token_ids.reshape(-1)
        .tolist()
    )

    for rid, row, sp in (
        ("c", 0, dict(temperature=0.0, min_p=0.7)),
        ("b", 1, dict(temperature=0.0, min_p=0.3)),
    ):
        ref_batch = _plain_batch_of_one(
            _make_req(rid, [int(rid != "b") + 2], [], sp), with_custom=False
        )
        ref = (
            sampler(
                logits=logits[row : row + 1].clone(),
                sampling_metadata=_ref_sampling_metadata(ref_batch, 1),
            )
            .sampled_token_ids.reshape(-1)
            .tolist()[0]
        )
        assert merged[row] == ref, f"{rid}@row{row}: merged={merged[row]} ref={ref}"


def test_max_num_logprobs_over_gappy_layout():
    b = _lane_batch(num_lanes=2, per_lane=4, with_custom=False)
    assert b.max_num_logprobs is None  # empty
    _add_to_lane(b, _make_req("a", [1], [], dict(temperature=0.0)), 0)
    assert b.max_num_logprobs is None  # no logprobs requested
    _add_to_lane(b, _make_req("b", [1], [], dict(temperature=0.0, logprobs=5)), 1)
    assert b.max_num_logprobs == 5  # found despite the gappy (row 0 + row 4) layout


def test_merged_sampling_metadata_filters_generators_to_scheduled_rows():
    b = _lane_batch(num_lanes=2, per_lane=4, with_custom=False)
    row0 = _add_to_lane(b, _make_req("a", [1], [], dict(temperature=0.7), seed=11), 0)
    row4 = _add_to_lane(b, _make_req("b", [1], [], dict(temperature=0.7), seed=22), 1)
    assert (row0, row4) == (0, 4)

    metadata = b.build_merged_sampling_metadata(scheduled_rows=[row4])

    assert set(metadata.generators) == {row4}
    assert metadata.generators[row4] is b.sampling.generators[row4]


def test_scheduled_seeded_row_isolated_from_unscheduled_random_row():
    """A scheduled seeded request samples identically whether or not an
    unscheduled (slot-occupying, not in ``scheduled_rows``) random request
    shares the merged batch, and the unscheduled request's RNG is not advanced.

    Guards the concern that filtering generators to the scheduled rows makes
    ``len(generators) != batch_size``, so the sampler takes the global
    ``exponential_`` path over every row. That is safe here because the merged
    batch is ALWAYS the full, constant-size slot grid (decode = full logits,
    prefill = scattered onto the full grid), so each row's randomness is
    independent and seeded rows are overwritten by their own filtered generator
    -- an unscheduled row can neither change the global RNG draw count nor
    another row's value.
    """
    torch.manual_seed(99)
    seed = 4321
    sampler = Sampler()

    # Reference: the seeded request sampled alone (batch of 1).
    ref_batch = _plain_batch_of_one(
        _make_req("a", [1], [], dict(temperature=0.8, top_k=5), seed=seed),
        with_custom=False,
    )
    logits_a = torch.randn(1, VOCAB)
    ref = (
        sampler(
            logits=logits_a.clone(),
            sampling_metadata=_ref_sampling_metadata(ref_batch, 1),
        )
        .sampled_token_ids.reshape(-1)
        .tolist()[0]
    )

    # Lane: seeded "a" at row 0 (scheduled) + unscheduled random "b" at row 4.
    b = _lane_batch(num_lanes=2, per_lane=4, with_custom=False)
    row_a = _add_to_lane(
        b, _make_req("a", [1], [], dict(temperature=0.8, top_k=5), seed=seed), 0
    )
    row_b = _add_to_lane(b, _make_req("b", [2], [], dict(temperature=0.9), seed=777), 1)
    b.refresh_logitsprocs()
    assert (row_a, row_b) == (0, 4)

    logits = torch.randn(b.max_num_reqs, VOCAB)
    logits[row_a] = logits_a[0]  # same logits row as the reference

    gen_b_before = b.sampling.generators[row_b].get_state().clone()
    merged = (
        sampler(
            logits=logits.clone(),
            sampling_metadata=b.build_merged_sampling_metadata(scheduled_rows=[row_a]),
        )
        .sampled_token_ids.reshape(-1)
        .tolist()
    )

    # The scheduled seeded request matches its batch-of-1 reference: neither the
    # unscheduled row nor the global exponential draw perturbs it.
    assert merged[row_a] == ref
    # The unscheduled request's generator was not advanced (it is excluded).
    assert torch.equal(b.sampling.generators[row_b].get_state(), gen_b_before)


# --------------------------------------------------------------------------
# Runner lane-mode selection (which persistent batch initialize_kv_cache builds)
# --------------------------------------------------------------------------


def _runner_with(data_parallel_size, tt_data_parallel_size):
    from vllm_tt_plugin.model_runner import TTModelRunner

    r = TTModelRunner.__new__(TTModelRunner)
    r.parallel_config = SimpleNamespace(data_parallel_size=data_parallel_size)
    r.tt_data_parallel_size = tt_data_parallel_size
    return r


def test_runner_is_lane_mode_property():
    # Lane mode: vLLM sees one engine (data_parallel_size == 1) but the TT
    # backend runs >1 in-process lane -> build a TTLaneInputBatch.
    assert _runner_with(data_parallel_size=1, tt_data_parallel_size=4)._is_lane_mode
    # Non-DP: one engine, one lane -> plain InputBatch.
    assert not _runner_with(data_parallel_size=1, tt_data_parallel_size=1)._is_lane_mode
    # Gathered multi-process DP: each rank is its own engine -> plain InputBatch.
    assert not _runner_with(data_parallel_size=4, tt_data_parallel_size=4)._is_lane_mode
