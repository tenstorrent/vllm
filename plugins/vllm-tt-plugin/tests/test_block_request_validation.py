# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys

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
    with pytest.raises(ValueError, match="reserve one full 256-token.*261888"):
        _validate(SamplingParams(max_tokens=1), prompt_len=261889)


@pytest.mark.parametrize(
    ("cli_args", "max_model_len"),
    [
        (
            [
                "--model",
                "google/diffusiongemma-26B-A4B-it",
                "--max-model-len",
                "262144",
            ],
            262144,
        ),
        (
            [
                "--model=google/diffusiongemma-26B-A4B-it",
                "--max_model_len=131072",
            ],
            131072,
        ),
    ],
)
def test_api_pre_registration_initializes_exact_block_boundary(
    monkeypatch, cli_args, max_model_len
):
    monkeypatch.setattr(sys, "argv", ["vllm", "serve", *cli_args])

    TTPlatform.pre_register_and_update()

    assert TTPlatform.block_output_size == 256
    assert TTPlatform.block_model_max_len == max_model_len
    _validate(
        SamplingParams(max_tokens=256),
        prompt_len=max_model_len - TTPlatform.block_output_size,
    )
    with pytest.raises(ValueError, match="reserve one full 256-token"):
        _validate(
            SamplingParams(max_tokens=1),
            prompt_len=max_model_len - TTPlatform.block_output_size + 1,
        )


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"n": 2}, "n"),
        ({"logprobs": 0}, "logprobs"),
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
