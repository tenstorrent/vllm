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


# --------------------------------------------------------------------------
# Placement / stable slots
# --------------------------------------------------------------------------


def test_requests_land_in_their_lane_chunk():
    b = _lane_batch(num_lanes=2, per_lane=4)  # rows 0..3 lane0, 4..7 lane1
    r0 = b.add_request_to_lane(_make_req("a", [1], [], dict(temperature=0.0)), lane=0)
    r1 = b.add_request_to_lane(_make_req("b", [1], [], dict(temperature=0.0)), lane=1)
    r2 = b.add_request_to_lane(_make_req("c", [1], [], dict(temperature=0.0)), lane=0)
    assert (r0, r2) == (0, 1)  # lane 0 chunk, lowest free slots
    assert r1 == 4  # lane 1 chunk base
    assert b.occupied_rows() == [0, 1, 4]
    assert b.lane_of("b") == 1


def test_existing_requests_keep_row_on_admission_and_removal():
    b = _lane_batch(num_lanes=1, per_lane=8)
    rows = {
        rid: b.add_request_to_lane(_make_req(rid, [1], [], dict(temperature=0.0)), 0)
        for rid in ("a", "b", "c")
    }
    assert rows == {"a": 0, "b": 1, "c": 2}
    # Remove the middle request: its row becomes a gap; others do not move.
    assert b.remove_request("b") == 1
    assert b.req_id_to_index["a"] == 0 and b.req_id_to_index["c"] == 2
    assert b.occupied_rows() == [0, 2]
    # A new request reuses the lowest free slot (the gap at row 1).
    assert b.add_request_to_lane(_make_req("d", [1], [], dict(temperature=0.0)), 0) == 1


def test_condense_is_noop():
    b = _lane_batch(num_lanes=1, per_lane=4)
    b.add_request_to_lane(_make_req("a", [1], [], dict(temperature=0.0)), 0)
    b.add_request_to_lane(_make_req("b", [1], [], dict(temperature=0.0)), 0)
    b.remove_request("a")  # gap at row 0
    b.condense([0])  # must not move "b" down into row 0
    assert b.req_id_to_index["b"] == 1
    assert b.occupied_rows() == [1]


def test_lane_full_raises():
    b = _lane_batch(num_lanes=1, per_lane=2)
    b.add_request_to_lane(_make_req("a", [1], [], dict(temperature=0.0)), 0)
    b.add_request_to_lane(_make_req("b", [1], [], dict(temperature=0.0)), 0)
    try:
        b.add_request_to_lane(_make_req("c", [1], [], dict(temperature=0.0)), 0)
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
        rows[rid] = batch.add_request_to_lane(
            _make_req(rid, s["prompt"], s["output"], s["sp"]), lane
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
    batch.add_request_to_lane(a, 0)  # row 0
    batch.add_request_to_lane(keep, 0)  # row 1
    batch.refresh_logitsprocs()

    # Same step: remove "a" (frees row 0), admit "c" (min_p) -> reuses row 0.
    batch.remove_request("a")
    c = _make_req("c", [3], [], dict(temperature=0.0, min_p=0.7))
    row_c = batch.add_request_to_lane(c, 0)
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
    b.add_request_to_lane(_make_req("a", [1], [], dict(temperature=0.0)), 0)
    assert b.max_num_logprobs is None  # no logprobs requested
    b.add_request_to_lane(_make_req("b", [1], [], dict(temperature=0.0, logprobs=5)), 1)
    assert b.max_num_logprobs == 5  # found despite the gappy (row 0 + row 4) layout


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
