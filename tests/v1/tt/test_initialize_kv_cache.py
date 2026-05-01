# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``TTModelRunner.initialize_kv_cache`` and helpers.

Phase 2 of kv-cache-groups: the runner now refactors group handling into
helpers, validates AttentionSpec on every group, stores ``kv_cache_config``
for downstream phases, and explicitly errors on multi-group configs (the
end-to-end hybrid path lands in Phase 3 + 7).
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
    SlidingWindowSpec,
)


def _full_spec(block_size=64, num_kv_heads=8, head_size=128):
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
    )


def _sliding_spec(block_size=64, num_kv_heads=8, head_size=128, sliding_window=1024):
    return SlidingWindowSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=torch.bfloat16,
        sliding_window=sliding_window,
    )


def _config(groups, num_blocks=2048, tensors=None):
    if tensors is None:
        tensors = [
            KVCacheTensor(size=1, shared_by=[name])
            for g in groups
            for name in g.layer_names
        ]
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=groups,
    )


@pytest.fixture
def runner():
    """Synthetic TTModelRunner skipping the real ``__init__``."""
    from vllm.v1.worker.tt_model_runner import TTModelRunner

    r = TTModelRunner.__new__(TTModelRunner)
    r.model_config = MagicMock()
    r.parallel_config = MagicMock()
    r.cache_config = MagicMock()
    r.device_config = MagicMock()
    r.scheduler_config = MagicMock()
    r.model = MagicMock()
    r._host_logitsprocs = MagicMock()
    r.vocab_size = 32000

    r.model_config.max_model_len = 8192
    r.cache_config.block_size = 64
    r.parallel_config.data_parallel_size = 1
    r.parallel_config.data_parallel_rank_local = 0
    r.device_config.num_devices = 2
    r.scheduler_config.max_num_seqs = 32
    r.scheduler_config.max_num_batched_tokens = 8192

    # Default: single-attention model, 32 layers
    r.model_config.get_num_layers_by_block_type.return_value = 32
    r.model.allocate_kv_cache.return_value = "kv-caches-sentinel"
    return r


def test_validate_groups_empty_raises(runner):
    with pytest.raises(ValueError, match="no groups"):
        runner._validate_kv_cache_groups([])


def test_validate_groups_non_attention_raises(runner):
    mamba_group = KVCacheGroupSpec(
        layer_names=["m.0"],
        kv_cache_spec=MambaSpec(
            shapes=((2, 4),),
            dtypes=(torch.bfloat16,),
            block_size=-1,
        ),
    )
    with pytest.raises(TypeError, match="Expected AttentionSpec"):
        runner._validate_kv_cache_groups([mamba_group])


def test_kv_cache_shape_folds_in_tp(runner):
    runner.device_config.num_devices = 4  # 4 devices, 1 DP rank
    spec = _full_spec(num_kv_heads=8, head_size=64, block_size=32)
    shape = runner._kv_cache_shape(spec, num_blocks=128)
    # tp = min(4, 8) = 4 → num_kv_heads_per_device = 8 // 4 = 2
    assert shape == (128, 2, 32, 64)


def test_kv_cache_shape_when_kv_heads_below_tp(runner):
    """When kv heads < tp, each device carries one head (existing behavior)."""
    runner.device_config.num_devices = 8  # tp = 8 but only 2 kv heads
    spec = _full_spec(num_kv_heads=2)
    shape = runner._kv_cache_shape(spec, num_blocks=64)
    # min(8, 2) = 2 → num_kv_heads_per_device = 2 // 2 = 1
    assert shape[1] == 1


def test_initialize_single_group_calls_model_allocate(runner):
    runner.model.allocate_kv_cache_per_layer.return_value = "kv-caches-sentinel"
    spec = _full_spec(num_kv_heads=8, head_size=128, block_size=64)
    config = _config([KVCacheGroupSpec(layer_names=["l.0"], kv_cache_spec=spec)])

    runner.initialize_kv_cache(config)

    runner.model.allocate_kv_cache_per_layer.assert_called_once()
    (per_layer_specs,), _ = runner.model.allocate_kv_cache_per_layer.call_args
    expected_shape = (2048, 8 // min(2, 8), 64, 128)
    assert len(per_layer_specs) == 32
    assert all(s == (expected_shape, torch.bfloat16) for s in per_layer_specs)
    assert runner.kv_caches == "kv-caches-sentinel"
    assert runner.kv_cache_config is config


def test_initialize_multi_group_assigns_specs_per_layer(runner):
    """Hybrid configuration: 2 full-attention layers + 4 sliding-window
    layers should produce a per-layer spec list where each layer's shape
    matches its group's spec.
    """
    runner.model_config.get_num_layers_by_block_type.return_value = 6
    runner.model.allocate_kv_cache_per_layer.return_value = "kv-caches-sentinel"
    full = _full_spec(num_kv_heads=4, head_size=128, block_size=64)
    sliding = _sliding_spec(
        num_kv_heads=4, head_size=128, block_size=64, sliding_window=1024
    )
    config = _config(
        [
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn", "model.layers.5.self_attn"],
                kv_cache_spec=full,
            ),
            KVCacheGroupSpec(
                layer_names=[
                    "model.layers.1.self_attn",
                    "model.layers.2.self_attn",
                    "model.layers.3.self_attn",
                    "model.layers.4.self_attn",
                ],
                kv_cache_spec=sliding,
            ),
        ],
        num_blocks=512,
    )

    runner.initialize_kv_cache(config)

    (per_layer_specs,), _ = runner.model.allocate_kv_cache_per_layer.call_args
    assert len(per_layer_specs) == 6
    full_shape = (512, 4 // min(2, 4), 64, 128)
    sliding_shape = (512, 4 // min(2, 4), 64, 128)
    # Layer indices 0 and 5 are full attention; 1-4 are sliding window.
    # (Shapes happen to match since both groups share kv-heads/head-size in
    # this fixture; the spec types differ, which is the structural change
    # that propagates downstream once forward routing lands.)
    assert per_layer_specs[0][0] == full_shape
    assert per_layer_specs[5][0] == full_shape
    assert per_layer_specs[1][0] == sliding_shape
    assert per_layer_specs[4][0] == sliding_shape


def test_initialize_stores_config_even_when_not_dp_rank_zero(runner):
    """Non-driver DP ranks skip allocation but should still build the input
    batch so they can participate in input prep / DP gather."""
    runner.parallel_config.data_parallel_rank_local = 1
    spec = _full_spec()
    config = _config([KVCacheGroupSpec(layer_names=["l.0"], kv_cache_spec=spec)])

    runner.initialize_kv_cache(config)

    runner.model.allocate_kv_cache.assert_not_called()
    assert runner.kv_cache_config is config
    assert runner.input_batch is not None


def test_build_per_layer_specs_rejects_unparseable_layer_name(runner):
    runner.model_config.get_num_layers_by_block_type.return_value = 2
    full = _full_spec()
    sliding = _sliding_spec()
    config = _config(
        [
            KVCacheGroupSpec(layer_names=["foo"], kv_cache_spec=full),
            KVCacheGroupSpec(layer_names=["bar"], kv_cache_spec=sliding),
        ]
    )
    with pytest.raises(ValueError, match="parse a layer index"):
        runner._build_per_layer_specs(config, num_layers=2)


def test_build_per_layer_specs_rejects_missing_layers(runner):
    """If the spec hook didn't return a spec for every attention layer,
    we should fail loudly in the multi-group path rather than silently
    allocating None for the missing layers."""
    runner.model_config.get_num_layers_by_block_type.return_value = 4
    config = _config(
        [
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"], kv_cache_spec=_full_spec()
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.1.self_attn"], kv_cache_spec=_sliding_spec()
            ),
        ]
    )
    with pytest.raises(ValueError, match=r"layer indices \[2, 3\]"):
        runner._build_per_layer_specs(config, num_layers=4)


def test_build_per_layer_specs_rejects_duplicate_layer(runner):
    config = _config(
        [
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"], kv_cache_spec=_full_spec()
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"], kv_cache_spec=_sliding_spec()
            ),
        ]
    )
    with pytest.raises(ValueError, match="already.*assigned"):
        runner._build_per_layer_specs(config, num_layers=2)


def test_build_per_layer_specs_rejects_out_of_range_index(runner):
    """Layer index from a group's layer name must be within
    [0, num_layers); otherwise we'd silently corrupt the per-layer list.
    Forces the multi-group path so the parsing branch runs."""
    config = _config(
        [
            KVCacheGroupSpec(
                layer_names=["model.layers.0.self_attn"], kv_cache_spec=_full_spec()
            ),
            KVCacheGroupSpec(
                layer_names=["model.layers.5.self_attn"], kv_cache_spec=_sliding_spec()
            ),
        ]
    )
    with pytest.raises(ValueError, match="out of range"):
        runner._build_per_layer_specs(config, num_layers=2)


def test_initialize_creates_one_block_table_per_group(runner, monkeypatch):
    """For hybrid models, InputBatch must receive block_sizes with one
    entry per kv_cache_group so its MultiGroupBlockTable produces the
    matching number of per-group block tables."""
    from vllm.v1.worker import tt_model_runner as runner_module

    captured = {}

    def fake_input_batch(**kw):
        captured["block_sizes"] = kw["block_sizes"]
        captured["kernel_block_sizes"] = kw["kernel_block_sizes"]
        return MagicMock(name="fake-input-batch")

    monkeypatch.setattr(runner_module, "InputBatch", fake_input_batch)

    runner.model_config.get_num_layers_by_block_type.return_value = 6
    runner.model.allocate_kv_cache_per_layer.return_value = "kv-caches-sentinel"
    full = _full_spec()
    sliding = _sliding_spec()
    config = _config(
        [
            KVCacheGroupSpec(
                layer_names=[f"model.layers.{i}.self_attn" for i in (0, 5)],
                kv_cache_spec=full,
            ),
            KVCacheGroupSpec(
                layer_names=[f"model.layers.{i}.self_attn" for i in (1, 2, 3, 4)],
                kv_cache_spec=sliding,
            ),
        ]
    )

    runner.initialize_kv_cache(config)

    # Two groups → two block tables, both with the same block_size.
    assert captured["block_sizes"] == [64, 64]
    assert captured["kernel_block_sizes"] == [64, 64]


def test_initialize_single_group_keeps_one_block_table(runner, monkeypatch):
    """Single-group (uniform attention) is the legacy contract: exactly
    one block table in the input batch — confirms Phase 4 didn't change
    the wire shape for uniform models."""
    from vllm.v1.worker import tt_model_runner as runner_module

    captured = {}

    def fake_input_batch(**kw):
        captured["block_sizes"] = kw["block_sizes"]
        return MagicMock(name="fake-input-batch")

    monkeypatch.setattr(runner_module, "InputBatch", fake_input_batch)

    config = _config([KVCacheGroupSpec(layer_names=["l.0"], kv_cache_spec=_full_spec())])
    runner.initialize_kv_cache(config)

    assert captured["block_sizes"] == [64]


def test_initialize_mixed_block_sizes_rejected(runner):
    """The persistent input batch currently only supports a single
    block_size; uneven block sizes across groups must error early."""
    g1 = KVCacheGroupSpec(layer_names=["l.0"], kv_cache_spec=_full_spec(block_size=32))
    g2 = KVCacheGroupSpec(layer_names=["l.1"], kv_cache_spec=_full_spec(block_size=64))
    config = _config([g1, g2])

    with pytest.raises(AssertionError, match="block size"):
        runner.initialize_kv_cache(config)
