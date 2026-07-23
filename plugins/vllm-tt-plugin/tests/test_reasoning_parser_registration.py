# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning import ReasoningParserManager
from vllm_tt_plugin.entrypoints import _register_tt_reasoning_parsers


def test_gemma_family_reasoning_parser_aliases(monkeypatch):
    registrations = []

    def record_registration(name, module_path, class_name):
        registrations.append((name, module_path, class_name))

    monkeypatch.setattr(
        ReasoningParserManager,
        "register_lazy_module",
        staticmethod(record_registration),
    )

    _register_tt_reasoning_parsers()

    assert registrations == [
        (
            "gemma4",
            "vllm_tt_plugin.gemma4_reasoning_parser",
            "Gemma4ReasoningParser",
        ),
        (
            "diffusion_gemma",
            "vllm_tt_plugin.gemma4_reasoning_parser",
            "Gemma4ReasoningParser",
        ),
    ]
