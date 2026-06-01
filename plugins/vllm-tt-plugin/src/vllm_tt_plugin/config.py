# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def get_tt_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Return TT plugin config from the generic plugin namespace."""
    return dict(vllm_config.plugin_config.get("tt", {}))


def should_open_mesh_for_rank(local_dp_rank: int | None, full_dp_mode: bool) -> bool:
    """Return whether this worker rank should own a TT mesh device."""
    rank = 0 if local_dp_rank is None else local_dp_rank
    return full_dp_mode or rank == 0
