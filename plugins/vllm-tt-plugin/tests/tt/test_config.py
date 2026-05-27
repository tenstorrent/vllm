# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from vllm_tt_plugin import config as tt_config


@pytest.fixture(autouse=True)
def reset_warning_state():
    old_warned_plugin_config = tt_config._warned_plugin_config
    tt_config._warned_plugin_config = False
    yield
    tt_config._warned_plugin_config = old_warned_plugin_config


def _vllm_config(additional_config=None, plugin_config=None):
    return SimpleNamespace(
        additional_config=additional_config or {},
        plugin_config=plugin_config or {},
    )


def test_get_tt_config_prefers_additional_config():
    config = _vllm_config(
        additional_config={"tt": {"sample_on_device_mode": "all"}},
    )

    assert tt_config.get_tt_config(config) == {"sample_on_device_mode": "all"}


def test_get_tt_config_accepts_plugin_config_with_warning(caplog):
    config = _vllm_config(
        plugin_config={"tt": {"sample_on_device_mode": "decode_only"}},
    )

    assert tt_config.get_tt_config(config) == {"sample_on_device_mode": "decode_only"}
    assert "--plugin-config is deprecated" in caplog.text


def test_get_tt_config_rejects_both_config_sources():
    config = _vllm_config(
        additional_config={"tt": {"trace_mode": "all"}},
        plugin_config={"tt": {"trace_mode": "none"}},
    )

    with pytest.raises(
        ValueError, match="Only one of additional_config or plugin_config"
    ):
        tt_config.get_tt_config(config)
