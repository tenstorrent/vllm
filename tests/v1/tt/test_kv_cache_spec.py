# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``TTWorker.get_kv_cache_spec``.

The worker offers an opt-in hook on the registered TT model class so hybrid
attention models (e.g. Gemma3/4 mixed sliding+full) can declare per-layer
KV cache specs for upstream's hybrid kv cache manager. Models without the
hook fall back to the legacy single-spec behaviour.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)


@pytest.fixture
def worker():
    """Construct a TTWorker without invoking its heavy ``__init__``.

    The hook resolution path only reads config attributes, so we can build
    a synthetic instance and set just those attributes directly.
    """
    from vllm_tt_plugin.worker import TTWorker

    w = TTWorker.__new__(TTWorker)
    w.vllm_config = MagicMock()
    w.model_config = MagicMock()
    w.parallel_config = MagicMock()
    w.cache_config = MagicMock()

    w.model_config.architecture = "TestArch"
    w.model_config.use_mla = False
    w.model_config.dtype = torch.bfloat16
    w.model_config.get_num_kv_heads.return_value = 8
    w.model_config.get_head_size.return_value = 128
    w.model_config.get_sliding_window.return_value = None
    w.cache_config.cache_dtype = "auto"
    w.cache_config.block_size = 64
    # Default to non-DP so the hybrid-under-DP guard doesn't fire on
    # the legacy / single-DP test paths.
    w.parallel_config.data_parallel_size = 1
    return w


class _ModelWithoutHook:
    pass


class _ModelHookReturningNone:
    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        return None


class _ModelHookReturningSpecs:
    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        common = dict(
            block_size=64, num_kv_heads=4, head_size=128, dtype=torch.bfloat16
        )
        return {
            "layer.0.full": FullAttentionSpec(**common),
            "layer.1.sliding": SlidingWindowSpec(**common, sliding_window=1024),
        }


class _ModelHookReturningGarbage:
    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        return ["not", "a", "dict"]


@patch("vllm.model_executor.models.registry.ModelRegistry.resolve_model_cls")
def test_no_hook_falls_back_to_single_spec(resolve, worker):
    resolve.return_value = (_ModelWithoutHook, "TestArch")

    spec = worker.get_kv_cache_spec()

    assert list(spec.keys()) == ["foo"]
    assert isinstance(spec["foo"], FullAttentionSpec)
    assert spec["foo"].num_kv_heads == 8
    assert spec["foo"].head_size == 128
    assert spec["foo"].block_size == 64


@patch("vllm.model_executor.models.registry.ModelRegistry.resolve_model_cls")
def test_hook_returning_none_falls_back(resolve, worker):
    resolve.return_value = (_ModelHookReturningNone, "TestArch")

    spec = worker.get_kv_cache_spec()

    assert list(spec.keys()) == ["foo"]


@patch("vllm.model_executor.models.registry.ModelRegistry.resolve_model_cls")
def test_hook_returning_specs_used_directly(resolve, worker):
    resolve.return_value = (_ModelHookReturningSpecs, "TestArch")

    spec = worker.get_kv_cache_spec()

    assert set(spec.keys()) == {"layer.0.full", "layer.1.sliding"}
    assert isinstance(spec["layer.0.full"], FullAttentionSpec)
    assert isinstance(spec["layer.1.sliding"], SlidingWindowSpec)
    assert spec["layer.1.sliding"].sliding_window == 1024


@patch("vllm.model_executor.models.registry.ModelRegistry.resolve_model_cls")
def test_hook_returning_invalid_type_raises(resolve, worker):
    resolve.return_value = (_ModelHookReturningGarbage, "TestArch")

    with pytest.raises(TypeError, match="get_kv_cache_spec"):
        worker.get_kv_cache_spec()


@patch("vllm.model_executor.models.registry.ModelRegistry.resolve_model_cls")
def test_default_spec_uses_mla_when_set(resolve, worker):
    resolve.return_value = (_ModelWithoutHook, "TestArch")
    worker.model_config.use_mla = True

    spec = worker.get_kv_cache_spec()

    assert isinstance(spec["foo"], MLAAttentionSpec)


@patch("vllm.model_executor.models.registry.ModelRegistry.resolve_model_cls")
def test_default_spec_passes_sliding_window(resolve, worker):
    resolve.return_value = (_ModelWithoutHook, "TestArch")
    worker.model_config.get_sliding_window.return_value = 4096

    spec = worker.get_kv_cache_spec()

    assert isinstance(spec["foo"], FullAttentionSpec)
    assert spec["foo"].sliding_window == 4096


@patch("vllm.model_executor.models.registry.ModelRegistry.resolve_model_cls")
def test_hybrid_under_dp_returns_full_spec(resolve, worker):
    """Hybrid spec (mixed FullAttention + SlidingWindow) under DP > 1
    flows through unchanged: the DP gather/merge path now packs every
    kv_cache_group's block table into the gather payload and rebuilds
    them on the driver, so per-rank routing survives the merge."""
    resolve.return_value = (_ModelHookReturningSpecs, "TestArch")
    worker.parallel_config.data_parallel_size = 2

    spec = worker.get_kv_cache_spec()

    assert {type(s) for s in spec.values()} == {FullAttentionSpec, SlidingWindowSpec}


@patch("vllm.model_executor.models.registry.ModelRegistry.resolve_model_cls")
def test_uniform_spec_under_dp_allowed(resolve, worker):
    """A spec hook returning all-FullAttention layers (uniform) is fine
    under DP — only mixed-type hybrids are gated."""

    class _UniformHook:
        @classmethod
        def get_kv_cache_spec(cls, vllm_config):
            common = dict(
                block_size=64, num_kv_heads=4, head_size=128, dtype=torch.bfloat16
            )
            return {
                "model.layers.0.self_attn": FullAttentionSpec(**common),
                "model.layers.1.self_attn": FullAttentionSpec(**common),
            }

    resolve.return_value = (_UniformHook, "TestArch")
    worker.parallel_config.data_parallel_size = 4

    spec = worker.get_kv_cache_spec()

    assert len(spec) == 2
    assert all(isinstance(v, FullAttentionSpec) for v in spec.values())
