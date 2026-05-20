# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def get_tt_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Return TT plugin config from the generic plugin namespace."""
    return dict(vllm_config.plugin_config.get("tt", {}))


def get_tt_data_parallel_size(vllm_config: "VllmConfig") -> int:
    """Effective TT lane count for batching, KV sizing, and merged execution.

    When vLLM ``data_parallel_size > 1``, gathered-DP uses one engine per rank
    and this returns ``data_parallel_size``. When ``data_parallel_size == 1``,
    optional ``tt.tt_data_parallel_size`` enables in-process lanes.
    """
    parallel_config = vllm_config.parallel_config
    vllm_dp = parallel_config.data_parallel_size
    tt_config = get_tt_config(vllm_config)
    configured = tt_config.get("tt_data_parallel_size")

    if vllm_dp > 1:
        if configured is not None and int(configured) != vllm_dp:
            logger.warning(
                "Ignoring tt_data_parallel_size=%s because "
                "data_parallel_size=%d (gathered-DP uses vLLM DP size).",
                configured,
                vllm_dp,
            )
        return vllm_dp

    if configured is None:
        return 1
    lanes = int(configured)
    if lanes < 1:
        raise ValueError(f"tt_data_parallel_size must be >= 1, got {lanes}")
    return lanes


def uses_tt_lane_coordinator(vllm_config: "VllmConfig") -> bool:
    return (
        vllm_config.parallel_config.data_parallel_size == 1
        and get_tt_data_parallel_size(vllm_config) > 1
    )
