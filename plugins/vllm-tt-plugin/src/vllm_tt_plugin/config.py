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

    if (
        has_additional_config
        and has_plugin_config
        and additional_config != plugin_config
    ):
        raise ValueError(
            "TT config in additional_config and plugin_config differs. "
            "Pass only additional_config. plugin_config is deprecated."
        )

    return dict(additional_config if has_additional_config else plugin_config)
