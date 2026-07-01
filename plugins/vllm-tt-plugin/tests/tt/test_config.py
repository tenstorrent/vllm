# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from vllm_tt_plugin import config as tt_config


def _vllm_config(
    additional_config=None,
    data_parallel_size=1,
    max_num_seqs=8,
):
    return SimpleNamespace(
        additional_config=additional_config or {},
        parallel_config=SimpleNamespace(data_parallel_size=data_parallel_size),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )


def test_get_tt_config_reads_additional_config():
    config = _vllm_config(
        additional_config={"tt": {"sample_on_device_mode": "all"}},
    )

    assert tt_config.get_tt_config(config) == {"sample_on_device_mode": "all"}


def test_get_tt_per_lane_max_num_seqs_derives_lane_capacity_from_global_cap():
    config = _vllm_config(
        additional_config={tt_config._RESOLVED_LANE_COUNT_KEY: 4},
        max_num_seqs=32,
    )

    assert tt_config.get_tt_per_lane_max_num_seqs(config) == 8


def test_get_tt_per_lane_max_num_seqs_requires_divisible_global_cap():
    config = _vllm_config(
        additional_config={tt_config._RESOLVED_LANE_COUNT_KEY: 4},
        max_num_seqs=30,
    )

    with pytest.raises(ValueError, match="max_num_seqs.*divisible"):
        tt_config.get_tt_per_lane_max_num_seqs(config)
