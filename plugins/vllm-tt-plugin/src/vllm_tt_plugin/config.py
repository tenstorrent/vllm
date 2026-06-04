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
