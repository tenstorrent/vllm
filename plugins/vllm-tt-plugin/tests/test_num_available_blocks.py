# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``get_num_available_blocks_tt``.

The TT backend skips KV cache memory profiling and instead returns a
hard-coded per-model token budget that gets installed as
``num_gpu_blocks_override``. For hybrid attention models (Gemma3/4,
GPT-OSS, ...) the budget needs extra headroom for the sliding-window
groups; these tests pin the formula so the heuristic stays right.

``get_model_architecture`` is patched in every test. With a ``MagicMock``
config the real call raises ``ValueError`` (the registry returns an empty
unpack at ``resolve_model_cls``), which the production ``except
AttributeError`` clause does not catch. Patching it lets each test pick a
deterministic budget source:

* ``_fallback_arch`` makes the call raise ``AttributeError`` so the
  function uses the 131072 fallback.
* ``_model_budget`` returns a stub class whose ``get_max_tokens_all_users``
  yields a fixed per-model budget, isolating this function from the
  per-SKU tables that live on the model class.

Sliding-window headroom is gated by ``_HYBRID_KV_CACHE_GROUPS_ENABLED``
(currently ``False`` while hybrid KV groups are disabled model-side), so
the sliding tests patch it ``True`` to exercise the headroom formula.
"""

from unittest.mock import MagicMock, patch

import pytest
from vllm_tt_plugin import config as tt_config


@pytest.fixture
def cfg():
    """Minimal VllmConfig stand-in for the heuristic.

    Mocks ttnn.get_arch_name so the wormhole check passes; otherwise we'd
    need a real device to land on the per-SKU branches.

    additional_config and plugin_config are set to empty dicts so that
    get_tt_config's isinstance(config, dict) guard doesn't raise on
    the auto-generated MagicMock attributes.
    """
    c = MagicMock()
    c.model_config.model = "unknown-model-falls-into-default-branch"
    c.model_config.get_sliding_window.return_value = None
    c.parallel_config.data_parallel_size = 1
    c.device_config.num_devices = 1
    c.scheduler_config.max_num_seqs = 32
    c.cache_config.block_size = 64
    c.additional_config = {}
    c.plugin_config = {}
    return c


def _fallback_arch():
    """Force get_num_available_blocks_tt onto the 131072 fallback branch."""
    return patch(
        "vllm_tt_plugin.worker.get_model_architecture",
        side_effect=AttributeError,
    )


def _model_budget(max_tokens: int):
    """Patch the model class to return a fixed ``max_tokens_all_users``."""
    fake_model_class = MagicMock()
    fake_model_class.get_max_tokens_all_users.return_value = max_tokens
    return patch(
        "vllm_tt_plugin.worker.get_model_architecture",
        return_value=(fake_model_class, "arch"),
    )


def test_default_branch_no_sliding(cfg):
    from vllm_tt_plugin.worker import get_num_available_blocks_tt

    with (
        patch("vllm_tt_plugin.worker.ttnn.get_arch_name", return_value="wormhole_b0"),
        _fallback_arch(),
    ):
        n = get_num_available_blocks_tt(cfg)

    # Default branch: max_tokens_all_users = 131072, plus block_size*batch
    # padding (64*32 = 2048). num_blocks = ceil(133120 / 64) = 2080.
    assert n == 2080


def test_lane_mode_kv_shape_matches_per_lane_gathered_dp(cfg):
    """Enabling single-process lanes must not change ``num_blocks``.

    ``num_blocks`` is applied to each submesh KV cache un-divided, so the
    model -- plus its on-disk tensor cache -- must see the identical KV shape
    regardless of parallelism mode. A submesh serves only its *per-lane* slice
    of requests, so the batch padding uses ``max_num_seqs // lanes``. Padding
    with the global batch size would inflate ``num_blocks`` and give the model
    a different KV shape from a gathered-DP rank with the same per-lane
    capacity."""
    from vllm_tt_plugin.worker import get_num_available_blocks_tt

    cfg.scheduler_config.max_num_seqs = 32
    # Lane count is read from the resolved-lane-count key on additional_config.
    cfg.additional_config = {tt_config._RESOLVED_LANE_COUNT_KEY: 4}

    with (
        patch("vllm_tt_plugin.worker.ttnn.get_arch_name", return_value="wormhole_b0"),
        _fallback_arch(),
    ):
        n = get_num_available_blocks_tt(cfg)

    # Per-lane batch is 32 // 4 = 8.
    # Default tokens (131072) + batch padding (64 * 8 = 512) = 131584 tokens
    # -> ceil/64 = 2056.
    assert n == 2056


def test_sliding_window_adds_headroom(cfg):
    """Hybrid models declare a sliding_window; with hybrid KV groups
    enabled the heuristic adds headroom proportional to
    sliding_window x max_batch x a per-buffer group multiplier, otherwise
    hybrid prefill would run out of blocks at full batch."""
    from vllm_tt_plugin.worker import get_num_available_blocks_tt

    cfg.model_config.get_sliding_window.return_value = 1024

    with (
        patch("vllm_tt_plugin.worker.ttnn.get_arch_name", return_value="wormhole_b0"),
        patch("vllm_tt_plugin.worker._HYBRID_KV_CACHE_GROUPS_ENABLED", True),
        _fallback_arch(),
    ):
        n = get_num_available_blocks_tt(cfg)

    # Default tokens (131072) + batch padding (64*32=2048) +
    # sliding overhead (1024 * 32 * 8 = 262144) = 395264 tokens ->
    # ceil(395264 / 64) = 6176 blocks.
    assert n == 6176


def test_n150_branch_unchanged_for_uniform_model(cfg):
    """N150 Llama-3.1-8B (uniform attention) keeps its existing budget;
    sliding_window is None so no headroom is added."""
    from vllm_tt_plugin.worker import get_num_available_blocks_tt

    cfg.model_config.model = "/path/to/Llama-3.1-8B-Instruct"
    cfg.device_config.num_devices = 1

    with (
        patch("vllm_tt_plugin.worker.ttnn.get_arch_name", return_value="wormhole_b0"),
        _model_budget(32768),
    ):
        n = get_num_available_blocks_tt(cfg)

    # Llama8B-N150 branch: 32768 + 64*32 padding = 34816 -> ceil/64 = 544.
    assert n == 544


def test_per_model_branch_with_sliding_window(cfg):
    """Per-model SKU branches (e.g. gemma-3-4b on N300) still get sliding
    headroom on top of the per-SKU base when hybrid KV groups are on."""
    from vllm_tt_plugin.worker import get_num_available_blocks_tt

    cfg.model_config.model = "/path/to/gemma-3-4b-it"
    cfg.model_config.get_sliding_window.return_value = 1024
    cfg.device_config.num_devices = 2

    with (
        patch("vllm_tt_plugin.worker.ttnn.get_arch_name", return_value="wormhole_b0"),
        patch("vllm_tt_plugin.worker._HYBRID_KV_CACHE_GROUPS_ENABLED", True),
        _model_budget(65536),
    ):
        n = get_num_available_blocks_tt(cfg)

    # gemma-3-4b N300 branch: 65536 base + 64*32 padding + 1024*32*8 sliding
    # = 65536 + 2048 + 262144 = 329728 -> ceil/64 = 5152
    assert n == 5152
