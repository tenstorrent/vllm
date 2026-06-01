# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from vllm_tt_plugin import config as tt_config


def _vllm_config(
    *,
    data_parallel_size: int = 1,
    max_num_seqs: int = 8,
    tt_data_parallel_size: int | None = None,
):
    additional_config = {"tt": {}}
    if tt_data_parallel_size is not None:
        additional_config["tt"]["tt_data_parallel_size"] = tt_data_parallel_size

    return SimpleNamespace(
        additional_config=additional_config,
        plugin_config={},
        parallel_config=SimpleNamespace(data_parallel_size=data_parallel_size),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def test_get_tt_per_lane_max_num_seqs_derives_lane_capacity_from_global_cap():
    config = _vllm_config(max_num_seqs=32, tt_data_parallel_size=4)

    assert tt_config.get_tt_per_lane_max_num_seqs(config) == 8


def test_get_tt_per_lane_max_num_seqs_requires_divisible_global_cap():
    config = _vllm_config(max_num_seqs=30, tt_data_parallel_size=4)

    with pytest.raises(ValueError, match="max_num_seqs.*divisible"):
        tt_config.get_tt_per_lane_max_num_seqs(config)


def test_get_tt_max_batch_size_uses_global_cap_for_single_process_lanes():
    config = _vllm_config(max_num_seqs=32, tt_data_parallel_size=4)

    assert tt_config.get_tt_max_batch_size(config) == 32


def test_get_tt_max_batch_size_keeps_gathered_dp_contract():
    config = _vllm_config(data_parallel_size=4, max_num_seqs=8)

    assert tt_config.get_tt_max_batch_size(config) == 32


def test_uses_tt_lane_coordinator_only_for_single_process_lanes():
    assert tt_config.uses_tt_lane_coordinator(
        _vllm_config(data_parallel_size=1, tt_data_parallel_size=4)
    )
    assert not tt_config.uses_tt_lane_coordinator(
        _vllm_config(data_parallel_size=4, tt_data_parallel_size=4)
    )
    assert not tt_config.uses_tt_lane_coordinator(_vllm_config(data_parallel_size=1))


def test_get_tt_data_parallel_size_rejects_zero_lane_count():
    config = _vllm_config(tt_data_parallel_size=0)

    with pytest.raises(ValueError, match="tt_data_parallel_size must be >= 1"):
        tt_config.get_tt_data_parallel_size(config)


def test_gathered_dp_ignores_conflicting_tt_lane_count_for_batch_sizing():
    config = _vllm_config(
        data_parallel_size=4,
        max_num_seqs=8,
        tt_data_parallel_size=2,
    )

    assert tt_config.get_tt_data_parallel_size(config) == 4
    assert tt_config.get_tt_max_batch_size(config) == 32
    assert tt_config.get_tt_per_lane_max_num_seqs(config) == 8
