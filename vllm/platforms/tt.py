# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility shim for the TT platform plugin.

The TT backend implementation lives in the out-of-tree Phase 1 plugin package
under ``plugins/vllm-tt-plugin``. This module keeps older imports such as
``vllm.platforms.tt.TTPlatform`` working while vLLM core still has a built-in
TT platform fallback path.
"""

from importlib import import_module

_tt_platform = import_module("vllm_tt_plugin.platform")

TTPlatform = _tt_platform.TTPlatform
_should_pre_register_tt_test_models_from_cli = (
    _tt_platform._should_pre_register_tt_test_models_from_cli
)
register_tt_models = _tt_platform.register_tt_models
register_tt_test_models = _tt_platform.register_tt_test_models

TT_SCHEDULER_CLS = "vllm_tt_plugin.scheduler.TTScheduler"
TT_WORKER_CLS = "vllm_tt_plugin.worker.TTWorker"

__all__ = [
    "TTPlatform",
    "TT_SCHEDULER_CLS",
    "TT_WORKER_CLS",
    "_should_pre_register_tt_test_models_from_cli",
    "register_tt_models",
    "register_tt_test_models",
]
