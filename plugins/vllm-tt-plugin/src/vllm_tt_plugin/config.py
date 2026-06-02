# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def get_tt_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Return TT plugin config from the generic plugin namespace."""
    return dict(vllm_config.plugin_config.get("tt", {}))


def uses_tt_gathered_dp(vllm_config: "VllmConfig") -> bool:
    """Returns whether TT gathered-DP mode is enabled."""
    tt_data_parallel_size = get_tt_data_parallel_size(vllm_config)
    return tt_data_parallel_size is not None


def get_tt_data_parallel_size(vllm_config: "VllmConfig") -> int | None:
    """Returns the optional TT-specific gathered-DP size override."""
    tt_config = get_tt_config(vllm_config)
    raw = tt_config.get("tt_data_parallel_size")
    if raw is not None and (not isinstance(raw, int) or raw <= 0):
        raise ValueError(
            "TT plugin config key 'tt_data_parallel_size' must be a positive integer"
        )
    return raw


def should_open_mesh_for_rank(
    local_dp_rank: int | None, gathered_dp_mode: bool
) -> bool:
    """Returns whether this worker rank should own a TT mesh device."""
    rank = 0 if local_dp_rank is None else local_dp_rank
    return (not gathered_dp_mode) or rank == 0
