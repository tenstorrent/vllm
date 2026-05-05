# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test config for the TT-side unit tests.

These tests mock TTNN internally and don't need a working tt-metal C++
extension. We unconditionally short-circuit ``ttnn`` to a sentinel module
in ``sys.modules`` *before* any tests are collected so importing
``vllm_tt_plugin.model_runner`` and friends — which do
``import ttnn`` at module top — doesn't blow up on a broken local build.
A real CI environment with a working ttnn isn't affected because we
overwrite ``sys.modules['ttnn']`` regardless: the tests don't exercise
any real ttnn behaviour, only TT-runner Python logic.
"""

import sys
from unittest.mock import MagicMock

_ttnn_mock = MagicMock(name="ttnn-test-mock")
sys.modules["ttnn"] = _ttnn_mock
sys.modules["ttnn._ttnn"] = MagicMock(name="ttnn._ttnn-test-mock")
