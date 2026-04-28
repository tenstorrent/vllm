# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def get_tt_config(vllm_config: "VllmConfig") -> dict[str, Any]:
    """Return TT plugin config from the generic plugin namespace."""
    return dict(vllm_config.plugin_config.get("tt", {}))
