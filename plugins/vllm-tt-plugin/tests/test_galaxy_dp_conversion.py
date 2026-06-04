# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host tests for the transparent gathered-DP -> single-process lane conversion.

Galaxy-generator models (llama3_70b_galaxy, qwen3_32b_galaxy) are served on a
single device mesh, so ``--data_parallel_size N`` is folded into ``N``
in-process TT lanes by ``platform._convert_galaxy_gather_dp_to_lanes``.
"""

from types import SimpleNamespace

from vllm_tt_plugin import config as tt_config
from vllm_tt_plugin import platform as tt_platform


def _vllm_config(*, data_parallel_size, max_num_seqs):
    return SimpleNamespace(
        additional_config={"tt": {}},
        plugin_config={},
        parallel_config=SimpleNamespace(
            data_parallel_size=data_parallel_size,
            data_parallel_size_local=data_parallel_size,
            data_parallel_rank=0,
            data_parallel_rank_local=None,
            data_parallel_index=0,
            data_parallel_external_lb=False,
            data_parallel_hybrid_lb=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def test_galaxy_gather_dp_converted_to_lanes(monkeypatch):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "llama3_70b_galaxy")
    config = _vllm_config(data_parallel_size=4, max_num_seqs=8)

    tt_platform._convert_galaxy_gather_dp_to_lanes(config)

    # Gathered DP collapsed to a single in-process engine with 4 lanes.
    assert config.parallel_config.data_parallel_size == 1
    assert tt_config.get_tt_data_parallel_size(config) == 4
    assert tt_config.uses_tt_lane_coordinator(config)
    # Global capacity scaled so per-lane capacity stays the requested 8.
    assert config.scheduler_config.max_num_seqs == 32
    assert tt_config.get_tt_per_lane_max_num_seqs(config) == 8


def test_qwen3_galaxy_gather_dp_converted_to_lanes(monkeypatch):
    monkeypatch.setenv("TT_QWEN3_TEXT_VER", "qwen3_32b_galaxy")
    config = _vllm_config(data_parallel_size=2, max_num_seqs=16)

    tt_platform._convert_galaxy_gather_dp_to_lanes(config)

    assert config.parallel_config.data_parallel_size == 1
    assert tt_config.get_tt_data_parallel_size(config) == 2
    assert config.scheduler_config.max_num_seqs == 32


def test_collapse_resets_derived_parallel_fields(monkeypatch):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "llama3_70b_galaxy")
    config = _vllm_config(data_parallel_size=4, max_num_seqs=8)
    config.parallel_config.data_parallel_size_local = 4
    config.parallel_config.data_parallel_external_lb = True

    tt_platform._convert_galaxy_gather_dp_to_lanes(config)

    pc = config.parallel_config
    assert pc.data_parallel_size_local == 1
    assert pc.data_parallel_rank == 0
    # Local rank must collapse to 0, not None: the TT plugin gates mesh open,
    # model load, and KV-cache allocation on ``data_parallel_rank_local == 0``,
    # which is the value a genuine single-process run resolves to.
    assert pc.data_parallel_rank_local == 0
    assert pc.data_parallel_index == 0
    assert pc.data_parallel_external_lb is False
    assert pc.data_parallel_hybrid_lb is False


def test_no_conversion_for_non_galaxy_model(monkeypatch):
    monkeypatch.delenv("TT_LLAMA_TEXT_VER", raising=False)
    monkeypatch.delenv("TT_QWEN3_TEXT_VER", raising=False)
    config = _vllm_config(data_parallel_size=4, max_num_seqs=8)

    tt_platform._convert_galaxy_gather_dp_to_lanes(config)

    # Left as gathered multi-process DP, untouched.
    assert config.parallel_config.data_parallel_size == 4
    assert config.scheduler_config.max_num_seqs == 8
    assert tt_config._RESOLVED_LANE_COUNT_KEY not in config.additional_config


def test_conversion_is_noop_without_dp(monkeypatch):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "llama3_70b_galaxy")
    config = _vllm_config(data_parallel_size=1, max_num_seqs=8)

    tt_platform._convert_galaxy_gather_dp_to_lanes(config)

    assert config.parallel_config.data_parallel_size == 1
    assert config.scheduler_config.max_num_seqs == 8
    assert tt_config._RESOLVED_LANE_COUNT_KEY not in config.additional_config


def test_conversion_is_idempotent(monkeypatch):
    monkeypatch.setenv("TT_LLAMA_TEXT_VER", "llama3_70b_galaxy")
    config = _vllm_config(data_parallel_size=4, max_num_seqs=8)

    tt_platform._convert_galaxy_gather_dp_to_lanes(config)
    tt_platform._convert_galaxy_gather_dp_to_lanes(config)

    # Second call short-circuits (data_parallel_size already 1); no double scale.
    assert config.parallel_config.data_parallel_size == 1
    assert config.scheduler_config.max_num_seqs == 32
    assert tt_config.get_tt_data_parallel_size(config) == 4
