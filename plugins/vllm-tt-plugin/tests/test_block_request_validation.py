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
    # Past the last whole-canvas boundary the clamp must not collapse to 0:
    # max_tokens=0 would fail SamplingParams validation with a confusing
    # message. The min() then lands on max_model_len - prompt_len and
    # validate_request produces the canonical capacity error.
    assert platform.get_max_output_tokens(261889) == 256
    assert platform.get_max_output_tokens(262000) == 256
    assert entrypoint_utils.get_max_tokens(262144, None, 32, {}) == 261888
    assert entrypoint_utils.get_max_tokens(262144, 257, 261887, {}) == 257
    assert entrypoint_utils.get_max_tokens(262144, None, 262000, {}) == 144


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


_NEUTRAL_SAMPLING_VALUES = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "seed": None,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 0.0),
        ("temperature", 0.5),
        ("top_p", 0.9),
        ("top_k", 10),
        ("min_p", 0.1),
        ("seed", 42),
        ("presence_penalty", 0.5),
        ("frequency_penalty", 0.5),
        ("repetition_penalty", 1.1),
    ],
)
def test_block_model_accepts_and_ignores_sampling_knobs(field, value):
    """Sampling knobs must not fail the request; the model-owned sampler
    ignores them, so they are neutralized in place instead."""
    params = SamplingParams(max_tokens=256, **{field: value})

    _validate(params)

    assert getattr(params, field) == _NEUTRAL_SAMPLING_VALUES[field]


def test_block_model_ignores_typical_client_default_sampling_bundle():
    """A stock OpenAI-client request (or checkpoint generation-config
    injection) sends several non-neutral knobs at once; the request must
    succeed and every knob must be neutralized."""
    params = SamplingParams(max_tokens=256, temperature=0.7, top_p=0.95, top_k=64)

    _validate(params)

    assert params.temperature == 1.0
    assert params.top_p == 1.0
    assert params.top_k == 0


@pytest.mark.parametrize(
    ("kwargs", "parameter"),
    [
        ({"n": 2}, "n"),
        ({"logprobs": 0}, "logprobs"),
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
def test_block_model_rejects_response_contract_parameters(kwargs, parameter):
    """Parameters that change the response contract or force host-side
    logits processing are still rejected with a clear error."""
    with pytest.raises(ValueError, match=parameter):
        _validate(SamplingParams(**kwargs))


def _public_config(
    *,
    max_model_len=262144,
    max_num_seqs=1,
    enable_chunked_prefill=True,
    sample_on_device_mode=None,
    enable_prefix_caching=False,
    data_parallel_size=1,
    logits_processors=None,
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
        generation_config="auto",
        logits_processors=logits_processors,
    )
    additional_config = (
        {"tt": {"sample_on_device_mode": sample_on_device_mode}}
        if sample_on_device_mode
        else {}
    )
    return SimpleNamespace(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            worker_cls="auto",
            data_parallel_size=data_parallel_size,
        ),
        speculative_config=None,
        lora_config=None,
        additional_config=additional_config,
    )


class _DeviceSamplingBlockModel:
    model_capabilities = {
        "output_tokens_per_step": 256,
        "supports_sample_on_device": True,
    }


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
    config = _public_config(sample_on_device_mode="all")
    _patch_public_config_dependencies(monkeypatch, _DeviceSamplingBlockModel)

    TTPlatform.check_and_update_config(config)

    assert tt_config.get_tt_output_tokens_per_step(config) == 256
    assert TTPlatform.output_tokens_per_step == 256
    assert TTPlatform.block_model_max_len == 262144
    assert config.scheduler_config.enable_chunked_prefill is False
    assert config.scheduler_config.long_prefill_token_threshold == 0
    assert config.scheduler_config.max_num_batched_tokens == 262144
    # The model-owned sampler contract also neutralizes checkpoint
    # generation_config defaults that would otherwise be injected into
    # bare requests (DiffusionGemma ships max_new_tokens=256).
    assert config.model_config.generation_config == "vllm"


def test_public_config_requires_device_sampling_for_block_model(monkeypatch):
    config = _public_config(sample_on_device_mode=None)
    _patch_public_config_dependencies(monkeypatch, _DeviceSamplingBlockModel)

    with pytest.raises(ValueError, match=r'require.*sample_on_device_mode="all"'):
        TTPlatform.check_and_update_config(config)


def test_public_config_rejects_logits_processors_for_block_model(monkeypatch):
    config = _public_config(
        sample_on_device_mode="all", logits_processors=["custom.Processor"]
    )
    _patch_public_config_dependencies(monkeypatch, _DeviceSamplingBlockModel)

    with pytest.raises(ValueError, match=r"logits-processors"):
        TTPlatform.check_and_update_config(config)


def test_public_config_rejects_data_parallel_block_model(monkeypatch):
    config = _public_config(sample_on_device_mode="all", data_parallel_size=2)
    config.scheduler_config.max_num_seqs = 1
    _patch_public_config_dependencies(monkeypatch, _DeviceSamplingBlockModel)

    with pytest.raises(ValueError, match=r"do not yet support data parallelism"):
        TTPlatform.check_and_update_config(config)


def test_public_config_rejects_prefix_caching_block_model(monkeypatch):
    class PrefixCachingBlockModel:
        model_capabilities = {
            "output_tokens_per_step": 256,
            "supports_sample_on_device": True,
            "supports_prefix_caching": True,
        }

    config = _public_config(sample_on_device_mode="all", enable_prefix_caching=True)
    _patch_public_config_dependencies(monkeypatch, PrefixCachingBlockModel)

    with pytest.raises(ValueError, match=r"prefix caching"):
        TTPlatform.check_and_update_config(config)


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
    # Autoregressive models keep vLLM's default generation-config handling.
    assert config.model_config.generation_config == "auto"


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


def test_block_model_clamped_transport_default_gets_canonical_error():
    """A prompt past the last whole-canvas boundary resolves to a small
    positive default max_tokens; validation must produce the capacity error
    (not a max_tokens=0 crash, not advice to reduce max_tokens)."""
    with pytest.raises(ValueError) as exc_info:
        _validate(SamplingParams(max_tokens=144), prompt_len=262000)

    message = str(exc_info.value)
    assert "physical 256-token canvases" in message
    assert "shorter prompt" in message
    assert "Reduce max_tokens" not in message
