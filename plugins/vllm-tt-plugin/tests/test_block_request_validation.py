# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
from vllm_tt_plugin import config as tt_config
from vllm_tt_plugin.platform import TTPlatform

from vllm.sampling_params import SamplingParams, StructuredOutputsParams


@pytest.fixture(autouse=True)
def block_model_contract(monkeypatch):
    monkeypatch.setattr(TTPlatform, "output_tokens_per_step", 256, raising=False)
    monkeypatch.setattr(TTPlatform, "block_output_size", 256, raising=False)
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


@pytest.mark.parametrize("prompt_len", [32, 1000, 261887, 261888])
def test_block_model_accepts_omitted_max_tokens_with_whole_canvas_capacity(
    prompt_len,
):
    _validate(SamplingParams(max_tokens=None), prompt_len=prompt_len)


def test_block_model_clamps_transport_default_to_whole_canvas_capacity(monkeypatch):
    import vllm.entrypoints.utils as entrypoint_utils

    platform = TTPlatform()
    monkeypatch.setattr(entrypoint_utils, "current_platform", platform)

    assert platform.get_max_output_tokens(32) == 261888
    assert platform.get_max_output_tokens(1000) == 261120
    assert platform.get_max_output_tokens(261887) == 256
    assert platform.get_max_output_tokens(261888) == 256
    assert entrypoint_utils.get_max_tokens(262144, None, 32, {}) == 261888
    assert entrypoint_utils.get_max_tokens(262144, 257, 261887, {}) == 257


def test_block_model_rejects_omitted_max_tokens_when_no_canvas_fits():
    with pytest.raises(
        ValueError,
        match=r"resolved default output capacity.*cannot fit one.*shorter prompt",
    ):
        _validate(SamplingParams(max_tokens=None), prompt_len=261889)


def test_block_model_explicit_capacity_error_identifies_user_value():
    with pytest.raises(
        ValueError,
        match=r"user-specified max_tokens=257.*Reduce max_tokens",
    ):
        _validate(SamplingParams(max_tokens=257), prompt_len=261887)


def test_block_model_accepts_neutral_temperature():
    _validate(SamplingParams(max_tokens=256, temperature=1.0))


def test_block_model_rejects_zero_temperature_as_unwired_transport_control():
    with pytest.raises(ValueError) as exc_info:
        _validate(SamplingParams(max_tokens=256, temperature=0.0))

    message = str(exc_info.value)
    assert "temperature=0.0" in message
    assert "accepted transport value: 1.0" in message
    assert "model-owned Gumbel sampler" in message
    assert "0.8-to-0.4" in message
    assert "greedy" not in message.lower()


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


def _public_config(
    *,
    max_model_len=262144,
    max_num_seqs=1,
    enable_chunked_prefill=True,
):
    scheduler_config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=4096,
        enable_chunked_prefill=enable_chunked_prefill,
        long_prefill_token_threshold=128,
        async_scheduling=False,
        scheduler_cls=None,
        disable_chunked_mm_input=False,
        verify_max_model_len=lambda _max_model_len: None,
    )
    model_config = SimpleNamespace(
        max_logprobs=20,
        max_model_len=max_model_len,
        model="test-model",
        hf_config=SimpleNamespace(
            architectures=["FutureBlockModel"],
            model_type="gemma4",
        ),
        get_sliding_window=lambda: None,
    )
    return SimpleNamespace(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            worker_cls="auto",
            data_parallel_size=1,
        ),
        speculative_config=None,
        lora_config=None,
        additional_config={},
    )


def _patch_public_config_dependencies(monkeypatch, model_class):
    import vllm.model_executor.model_loader.utils as loader_utils
    from vllm.model_executor.models.registry import ModelRegistry

    monkeypatch.setattr(
        "vllm_tt_plugin.platform.register_tt_models", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        ModelRegistry,
        "get_supported_archs",
        lambda: {
            "TTFutureBlockModel",
            "TTDiffusionGemmaForBlockDiffusion",
        },
    )
    monkeypatch.setattr(
        loader_utils,
        "get_model_architecture",
        lambda _model_config: (model_class, "FutureBlockModel"),
    )


def test_public_config_normalizes_generic_block_capability_and_disables_chunking(
    monkeypatch,
):
    class BlockModel:
        model_capabilities = {"output_tokens_per_step": 256}

    config = _public_config()
    _patch_public_config_dependencies(monkeypatch, BlockModel)

    TTPlatform.check_and_update_config(config)

    assert tt_config.get_tt_output_tokens_per_step(config) == 256
    assert TTPlatform.output_tokens_per_step == 256
    assert TTPlatform.block_model_max_len == 262144
    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.long_prefill_token_threshold == 0
    assert config.scheduler_config.max_num_batched_tokens == 262144


def test_public_config_rejects_block_width_larger_than_model_context(monkeypatch):
    class BlockModel:
        model_capabilities = {"output_tokens_per_step": 256}

    config = _public_config(max_model_len=255)
    _patch_public_config_dependencies(monkeypatch, BlockModel)

    with pytest.raises(
        ValueError,
        match=r"max_model_len=255.*output_tokens_per_step=256",
    ):
        TTPlatform.check_and_update_config(config)


def test_public_config_applies_single_sequence_limit_to_generic_block_model(
    monkeypatch,
):
    class BlockModel:
        model_capabilities = {"output_tokens_per_step": 256}

    config = _public_config(max_num_seqs=2)
    _patch_public_config_dependencies(monkeypatch, BlockModel)

    with pytest.raises(ValueError, match=r"Block-output models.*max-num-seqs 1"):
        TTPlatform.check_and_update_config(config)


def test_public_config_defaults_autoregressive_model_to_one_token(monkeypatch):
    class AutoregressiveModel:
        model_capabilities = {}

    config = _public_config(enable_chunked_prefill=True)
    _patch_public_config_dependencies(monkeypatch, AutoregressiveModel)

    TTPlatform.check_and_update_config(config)

    assert tt_config.get_tt_output_tokens_per_step(config) == 1
    assert TTPlatform.output_tokens_per_step == 1
    assert TTPlatform.block_model_max_len is None
    assert config.scheduler_config.enable_chunked_prefill is True


def test_public_config_requires_diffusion_gemma_to_declare_block_width(monkeypatch):
    class MissingCapabilityModel:
        model_capabilities = {}

    config = _public_config()
    config.model_config.hf_config.architectures = ["DiffusionGemmaForBlockDiffusion"]
    _patch_public_config_dependencies(monkeypatch, MissingCapabilityModel)

    with pytest.raises(
        ValueError,
        match=r"DiffusionGemma must declare output_tokens_per_step > 1",
    ):
        TTPlatform.check_and_update_config(config)
