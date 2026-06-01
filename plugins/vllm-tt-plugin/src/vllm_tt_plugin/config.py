# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

_warned_plugin_config = False


def _extract_tt_config(
    config: dict[str, Any], config_name: str
) -> tuple[dict[str, Any], bool]:
    if not isinstance(config, dict):
        raise ValueError(f"{config_name} must be a JSON object")
    if "tt" not in config:
        return {}, False
    tt_config = config["tt"]
    if not isinstance(tt_config, dict):
        raise ValueError(f"{config_name}['tt'] must be a JSON object")
    return tt_config, True


def _warn_plugin_config() -> None:
    global _warned_plugin_config
    if _warned_plugin_config:
        return
    logger.warning(
        "TT config passed through --plugin-config is deprecated. "
        "Use --additional-config '{\"tt\": {...}}' instead."
    )
    _warned_plugin_config = True


def get_tt_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Return TT config from vLLM's generic additional config namespace."""
    additional_config, has_additional_config = _extract_tt_config(
        getattr(vllm_config, "additional_config", {}) or {}, "additional_config"
    )
    plugin_config, has_plugin_config = _extract_tt_config(
        getattr(vllm_config, "plugin_config", {}) or {}, "plugin_config"
    )

    if has_plugin_config:
        _warn_plugin_config()

    if has_additional_config and has_plugin_config:
        raise ValueError(
            "Only one of additional_config or plugin_config may contain TT config. "
            "Prefer additional_config. plugin_config is deprecated."
        )

    return dict(additional_config if has_additional_config else plugin_config)


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


def get_tt_max_batch_size(vllm_config: "VllmConfig") -> int:
    """Return the global TT batch capacity for model/KV sizing.

    Gathered multi-process DP keeps the historical contract: each rank receives
    ``max_num_seqs`` requests and the TT model is initialized for the gathered
    DP batch. Single-process lane mode is different: vLLM sees one engine, so
    ``max_num_seqs`` is already the global engine capacity and lanes are only an
    internal partition.
    """
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    if vllm_config.parallel_config.data_parallel_size > 1:
        return max_num_seqs * get_tt_data_parallel_size(vllm_config)
    return max_num_seqs


def get_tt_per_lane_max_num_seqs(vllm_config: "VllmConfig") -> int:
    """Return the per-lane/per-rank scheduling and wire-format capacity."""
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    if not uses_tt_lane_coordinator(vllm_config):
        return max_num_seqs

    lanes = get_tt_data_parallel_size(vllm_config)
    if max_num_seqs % lanes != 0:
        raise ValueError(
            "max_num_seqs must be divisible by tt_data_parallel_size in "
            "single-process TT lane mode; got "
            f"max_num_seqs={max_num_seqs}, tt_data_parallel_size={lanes}."
        )
    per_lane = max_num_seqs // lanes
    if per_lane < 1:
        raise ValueError(
            "max_num_seqs must provide at least one request per TT lane; got "
            f"max_num_seqs={max_num_seqs}, tt_data_parallel_size={lanes}."
        )
    return per_lane


def uses_tt_lane_coordinator(vllm_config: "VllmConfig") -> bool:
    return (
        vllm_config.parallel_config.data_parallel_size == 1
        and get_tt_data_parallel_size(vllm_config) > 1
    )
