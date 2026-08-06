# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm_tt_plugin.platform import TTPlatform

from vllm.sampling_params import SamplingParams, StructuredOutputsParams


@pytest.fixture(autouse=True)
def block_model_contract(monkeypatch):
    monkeypatch.setattr(TTPlatform, "block_output_size", 256)
    monkeypatch.setattr(TTPlatform, "block_model_max_len", 262144)


def _validate(params, prompt_len=32):
    TTPlatform.validate_request(
        prompt={"prompt_token_ids": [2] * prompt_len},
        params=params,
        processed_inputs={"prompt_token_ids": [2] * prompt_len},
    )


def test_block_model_accepts_default_sampling_and_stop_trimming():
    _validate(
        SamplingParams(
            max_tokens=300,
            ignore_eos=True,
            stop=["END"],
        )
    )


def test_block_model_accepts_exact_prompt_canvas_boundary():
    _validate(SamplingParams(max_tokens=256), prompt_len=261888)


def test_block_model_rejects_prompt_that_cannot_fit_a_canvas():
    with pytest.raises(ValueError, match="physical 256-token canvases"):
        _validate(SamplingParams(max_tokens=1), prompt_len=261889)


def test_block_model_rejects_rounded_physical_capacity_overrun():
    with pytest.raises(
        ValueError,
        match=r"prompt length 261844.*max_tokens=300.*512 physical",
    ):
        _validate(SamplingParams(max_tokens=300), prompt_len=261844)


def test_block_model_accepts_two_physical_canvases_at_short_prompt():
    _validate(SamplingParams(max_tokens=300), prompt_len=32)


def test_block_model_resolves_unbounded_max_tokens_before_dispatch():
    _validate(SamplingParams(max_tokens=None), prompt_len=261888)
    with pytest.raises(ValueError, match="max_tokens=257.*512 physical"):
        _validate(SamplingParams(max_tokens=None), prompt_len=261887)


def test_api_config_initializes_contract_from_model_capability():
    class BlockModel:
        model_capabilities = {"output_tokens_per_step": 256}

    TTPlatform._set_block_output_contract(
        BlockModel,
        SimpleNamespace(max_model_len=262144),
        is_diffusion_gemma=True,
    )

    assert TTPlatform.block_output_size == 256
    assert TTPlatform.block_model_max_len == 262144


def test_block_model_accepts_neutral_temperature():
    _validate(SamplingParams(max_tokens=256, temperature=1.0))


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"n": 2}, "n"),
        ({"logprobs": 0}, "logprobs"),
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": 0.5}, "temperature"),
        ({"top_p": 0.9}, "top_p"),
        ({"top_k": 10}, "top_k"),
        ({"min_p": 0.1}, "min_p"),
        ({"seed": 42}, "seed"),
        ({"presence_penalty": 0.5}, "presence_penalty"),
        ({"frequency_penalty": 0.5}, "frequency_penalty"),
        ({"repetition_penalty": 1.1}, "repetition_penalty"),
        ({"bad_words": ["forbidden"]}, "bad_words"),
        (
            {"structured_outputs": StructuredOutputsParams(json_object=True)},
            "structured_outputs",
        ),
        ({"logit_bias": {2: -1.0}}, "logit_bias"),
        ({"allowed_token_ids": [2, 3]}, "allowed_token_ids"),
        ({"min_tokens": 1}, "min_tokens"),
    ],
)
def test_block_model_rejects_unsupported_request_sampling(kwargs, parameter):
    with pytest.raises(ValueError, match=parameter):
        _validate(SamplingParams(**kwargs))
