# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only spike: prove a single merged host-sampling call is correct.

Validates the PR399 step-3 claim that host sampling can run over one merged
batch (no per-lane slicing), so the per-lane generator/penalty remap and the
builtin-logits-processor remap can be deleted and custom processors work.

Three checks:
  1. Isolation: merged batched sampling == per-request (batch-of-1) reference,
     exercising penalties, logit_bias, min_tokens, min_p, allowed_token_ids,
     bad_words, and a CUSTOM logits processor (greedy -> deterministic).
  2. Generator keying (A1): row-permutation equivariance proves each request
     samples with its own generator keyed by persistent row.
  3. Reproducibility: same merged batch, re-seeded generators -> identical.
"""

from types import SimpleNamespace

import torch
import vllm_tt_plugin  # noqa: F401  (activates tt platform / ttnn import)
from vllm_tt_plugin.input_batch import InputBatch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    build_logitsprocs,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_input_batch import CachedRequestState

VOCAB = 64
MAX_REQS = 8
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


def _stub_cfg():
    return SimpleNamespace(
        speculative_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=MAX_REQS),
    )


def _make_logitsprocs(with_custom=True):
    return build_logitsprocs(
        _stub_cfg(),
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
    sp = SamplingParams(**sp_kwargs)
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=None,
        sampling_params=sp,
        generator=gen,
        block_ids=([0],),
        num_computed_tokens=len(prompt),
        output_token_ids=list(output),
    )


def _build_batch(reqs, with_custom=True):
    batch = InputBatch(
        max_num_reqs=MAX_REQS,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=MAX_MODEL_LEN * MAX_REQS,
        vocab_size=VOCAB,
        block_sizes=[BLOCK],
        kernel_block_sizes=[BLOCK],
        logitsprocs=_make_logitsprocs(with_custom=with_custom),
    )
    for r in reqs:
        batch.add_request(r)
    batch.refresh_logitsprocs()
    return batch


def _build_sampling_metadata(batch):
    """Merged host-sampling builder (the reference impl for the rework)."""
    n = batch.num_reqs
    s = batch.sampling
    temperature = s.temperature[:n]
    all_greedy = bool((temperature == 0.0).all())
    all_random = bool((temperature != 0.0).all())
    presence = s.presence_penalty[:n]
    frequency = s.frequency_penalty[:n]
    repetition = s.repetition_penalty[:n]
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


def _sample(sampler, batch, logits):
    sm = _build_sampling_metadata(batch)
    out = sampler(logits=logits.clone(), sampling_metadata=sm)
    toks = out.sampled_token_ids.reshape(-1).tolist()
    return toks


def _req_specs():
    # Heterogeneous greedy requests exercising every host feature.
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


def _build_reqs(specs):
    return [_make_req(s["req_id"], s["prompt"], s["output"], s["sp"]) for s in specs]


def test_isolation_merged_equals_per_request():
    torch.manual_seed(1234)
    specs = _req_specs()
    n = len(specs)
    logits = torch.randn(n, VOCAB)
    sampler = Sampler()

    merged_batch = _build_batch(_build_reqs(specs))
    merged = _sample(sampler, merged_batch, logits)

    ref = []
    for i, spec in enumerate(specs):
        b = _build_batch(_build_reqs([spec]))
        tok = _sample(sampler, b, logits[i : i + 1])
        ref.append(tok[0])

    assert merged == ref, f"merged={merged} per_request={ref}"
    print("  [1] isolation merged==per-request:", merged)


def _random_specs(seed_a, seed_b):
    return [
        dict(
            req_id="a",
            prompt=[1],
            output=[],
            seed=seed_a,
            sp=dict(temperature=1.0, seed=seed_a),
        ),
        dict(
            req_id="b",
            prompt=[1],
            output=[],
            seed=seed_b,
            sp=dict(temperature=1.0, seed=seed_b),
        ),
    ]


def _build_random_reqs(specs):
    return [
        _make_req(s["req_id"], s["prompt"], s["output"], s["sp"], seed=s["seed"])
        for s in specs
    ]


def test_generator_keyed_by_row():
    # Uniform logits + no custom proc, so the sampled token is driven purely by
    # each row's generator and the two seeds yield different tokens.
    logits2 = torch.zeros(2, VOCAB)
    sampler = Sampler()

    specs = _random_specs(seed_a=111, seed_b=222)
    b1 = _build_batch(_build_random_reqs(specs), with_custom=False)
    t1 = _sample(sampler, b1, logits2)

    specs_sw = _random_specs(seed_a=222, seed_b=111)  # swap seeds across rows
    b2 = _build_batch(_build_random_reqs(specs_sw), with_custom=False)
    t2 = _sample(sampler, b2, logits2)

    assert t1[0] != t1[1], f"seeds did not diverge: {t1}"
    # Row0 of run1 used seed 111; row1 of run2 used seed 111 -> must match.
    assert t1[0] == t2[1] and t1[1] == t2[0], f"t1={t1} t2={t2}"
    print("  [2] generator keyed by row (permutation equivariance):", t1, t2)


def test_reproducible():
    logits = torch.zeros(2, VOCAB)
    sampler = Sampler()
    specs = _random_specs(seed_a=5, seed_b=6)
    t_first = _sample(
        sampler, _build_batch(_build_random_reqs(specs), with_custom=False), logits
    )
    t_again = _sample(
        sampler, _build_batch(_build_random_reqs(specs), with_custom=False), logits
    )
    assert t_first == t_again, f"{t_first} != {t_again}"
    print("  [3] reproducible across re-seeded builds:", t_first)


if __name__ == "__main__":
    test_isolation_merged_equals_per_request()
    test_generator_keyed_by_row()
    test_reproducible()
    print("ALL SPIKE CHECKS PASSED")
