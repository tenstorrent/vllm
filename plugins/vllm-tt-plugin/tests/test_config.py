# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from vllm_tt_plugin import config as tt_config


def _vllm_config(
    *,
    data_parallel_size: int = 1,
    max_num_seqs: int = 8,
    lane_count: int | None = None,
):
    additional_config: dict = {}
    if lane_count is not None:
        additional_config[tt_config._RESOLVED_LANE_COUNT_KEY] = lane_count

    return SimpleNamespace(
        additional_config=additional_config,
        parallel_config=SimpleNamespace(data_parallel_size=data_parallel_size),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def test_get_tt_per_lane_max_num_seqs_derives_lane_capacity_from_global_cap():
    config = _vllm_config(max_num_seqs=32, lane_count=4)

    assert tt_config.get_tt_per_lane_max_num_seqs(config) == 8


def test_get_tt_per_lane_max_num_seqs_requires_divisible_global_cap():
    config = _vllm_config(max_num_seqs=30, lane_count=4)

    with pytest.raises(ValueError, match="max_num_seqs.*divisible"):
        tt_config.get_tt_per_lane_max_num_seqs(config)


def test_get_tt_max_batch_size_uses_global_cap_for_single_process_lanes():
    config = _vllm_config(max_num_seqs=32, lane_count=4)

    assert tt_config.get_tt_max_batch_size(config) == 32


def test_get_tt_max_batch_size_keeps_gathered_dp_contract():
    config = _vllm_config(data_parallel_size=4, max_num_seqs=8)

    assert tt_config.get_tt_max_batch_size(config) == 32


def test_uses_tt_lane_coordinator_only_for_single_process_lanes():
    assert tt_config.uses_tt_lane_coordinator(
        _vllm_config(data_parallel_size=1, lane_count=4)
    )
    assert not tt_config.uses_tt_lane_coordinator(
        _vllm_config(data_parallel_size=4, lane_count=4)
    )
    assert not tt_config.uses_tt_lane_coordinator(_vllm_config(data_parallel_size=1))


def test_store_tt_lane_count_round_trips_through_get():
    config = _vllm_config(data_parallel_size=1)

    tt_config.store_tt_lane_count(config, 4)

    # Stored as an internal top-level key, not in the user "tt" namespace.
    assert config.additional_config[tt_config._RESOLVED_LANE_COUNT_KEY] == 4
    assert "tt" not in config.additional_config
    assert tt_config.get_tt_data_parallel_size(config) == 4


def test_store_tt_lane_count_creates_additional_config_when_missing():
    config = SimpleNamespace(
        additional_config=None,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )

    tt_config.store_tt_lane_count(config, 2)

    assert config.additional_config[tt_config._RESOLVED_LANE_COUNT_KEY] == 2


def test_store_tt_lane_count_rejects_zero():
    config = _vllm_config(data_parallel_size=1)

    with pytest.raises(ValueError, match="lane count must be >= 1"):
        tt_config.store_tt_lane_count(config, 0)
