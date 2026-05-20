# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``get_num_available_blocks_tt``.

The TT backend skips KV cache memory profiling and instead returns a
hard-coded per-model token budget that gets installed as
``num_gpu_blocks_override``. For hybrid attention models (Gemma3/4,
GPT-OSS, ...) the budget needs extra headroom for the sliding-window
groups; this test fixes the formula so the heuristic stays right as we
add hybrid models.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def cfg():
    """Minimal VllmConfig stand-in for the heuristic.

    Mocks ttnn.get_arch_name so the wormhole check passes; otherwise we'd
    need a real device to land on the per-SKU branches.
    """
    c = MagicMock()
    c.model_config.model = "unknown-model-falls-into-default-branch"
    c.model_config.get_sliding_window.return_value = None
    c.parallel_config.data_parallel_size = 1
    c.device_config.num_devices = 1
    c.scheduler_config.max_num_seqs = 32
    c.cache_config.block_size = 64
    return c


def test_default_branch_no_sliding(cfg):
    from vllm_tt_plugin.worker import get_num_available_blocks_tt

    with patch("vllm_tt_plugin.worker.ttnn.get_arch_name", return_value="wormhole_b0"):
        n = get_num_available_blocks_tt(cfg)

    # Default branch: max_tokens_all_users = 131072, plus block_size*batch
    # padding (64*32 = 2048). num_blocks = ceil(133120 / 64) = 2080.
    assert n == 2080


def test_lane_mode_uses_global_batch_padding(cfg):
    """Single-process lane mode must size KV padding for all concurrent
    requests, not just one lane's local ``max_num_seqs``."""
    from vllm_tt_plugin.worker import get_num_available_blocks_tt

    cfg.scheduler_config.max_num_seqs = 8
    cfg.plugin_config = {"tt": {"tt_data_parallel_size": 4}}

    with patch("vllm_tt_plugin.worker.ttnn.get_arch_name", return_value="wormhole_b0"):
        n = get_num_available_blocks_tt(cfg)

    # Default tokens (131072) + global batch padding (64 * (8 * 4) = 2048)
    # = 133120 tokens -> ceil/64 = 2080.
    assert n == 2080


def test_sliding_window_adds_headroom(cfg):
    """Hybrid models declare a sliding_window; the heuristic should add
    additional headroom proportional to sliding_window × max_batch ×
    a per-buffer group multiplier, otherwise hybrid prefill would run
    out of blocks at full batch."""
    from vllm_tt_plugin.worker import get_num_available_blocks_tt

    cfg.model_config.get_sliding_window.return_value = 1024

    with patch("vllm_tt_plugin.worker.ttnn.get_arch_name", return_value="wormhole_b0"):
        n = get_num_available_blocks_tt(cfg)

    # Default tokens (131072) + batch padding (64*32=2048) +
    # sliding overhead (1024 * 32 * 8 = 262144) = 395264 tokens →
    # ceil(395264 / 64) = 6176 blocks.
    assert n == 6176


def test_n150_branch_unchanged_for_uniform_model(cfg):
    """N150 Llama-3.1-8B (uniform attention) keeps its existing budget;
    sliding_window is None so no headroom is added."""
    from vllm_tt_plugin.worker import get_num_available_blocks_tt

    cfg.model_config.model = "/path/to/Llama-3.1-8B-Instruct"
    cfg.device_config.num_devices = 1

    with patch("vllm_tt_plugin.worker.ttnn.get_arch_name", return_value="wormhole_b0"):
        n = get_num_available_blocks_tt(cfg)

    # Llama8B-N150 branch: 32768 + 64*32 padding = 34816 → ceil/64 = 544.
    assert n == 544


def test_per_model_branch_with_sliding_window(cfg):
    """Per-model SKU branches (e.g. gemma-3-4b on N300) still get sliding
    headroom on top of the per-SKU base."""
    from vllm_tt_plugin.worker import get_num_available_blocks_tt

    cfg.model_config.model = "/path/to/gemma-3-4b-it"
    cfg.model_config.get_sliding_window.return_value = 1024
    cfg.device_config.num_devices = 2

    with patch("vllm_tt_plugin.worker.ttnn.get_arch_name", return_value="wormhole_b0"):
        n = get_num_available_blocks_tt(cfg)

    # gemma-3-4b N300 branch: 65536 base + 64*32 padding + 1024*32*8 sliding
    # = 65536 + 2048 + 262144 = 329728 → ceil/64 = 5152
    assert n == 5152
