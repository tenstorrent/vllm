# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the async-decode grammar payload wiring.

`_reorder_grammar_bitmask` is covered in test_reorder_grammar_bitmask.py. These
tests cover the layer above it: that `complete_non_dp_decode_step` builds the
grammar payload from the *submit-time* request-to-slot mapping captured in the
step context, not the runner's live mapping (which may have changed while the
async decode was in flight).
"""

from importlib import import_module
from types import SimpleNamespace

import torch

async_decode_module = import_module("vllm_tt_plugin.async_decode")
TTAsyncDecodeController = async_decode_module.TTAsyncDecodeController
SubmittedStepContext = async_decode_module.SubmittedStepContext


class TestAsyncGrammarWiring:
    def _controller_with_runner(self, live_req_id_to_index: dict[str, int]):
        controller = object.__new__(TTAsyncDecodeController)
        runner = SimpleNamespace()
        runner.input_batch = SimpleNamespace(req_id_to_index=live_req_id_to_index)
        captured: dict[str, object] = {}

        def _fake_get_output_tokens(**kwargs):
            captured["grammar_outputs"] = kwargs["grammar_outputs"]
            return [torch.zeros((1, 1), dtype=torch.int32)], [None]

        runner._get_output_tokens = _fake_get_output_tokens
        controller.runner = runner
        controller.captured = captured
        # finalize_decode is the device-touching step; stub it out.
        controller.finalize_decode = lambda submission: SimpleNamespace(
            tt_out=object(), tt_log_probs=None
        )
        return controller

    def _submission(self):
        return SimpleNamespace(
            sampling_params=None,
            model_sampling_params=None,
            batch_size_per_dp=[1],
            perform_device_sampling=False,
            device_sampling_deferred=False,
        )

    def test_grammar_payload_uses_captured_mapping(self):
        # Live mapping differs from the one captured at submit time.
        controller = self._controller_with_runner({"req_a": 1, "req_b": 0})
        captured_map = {"req_a": 0, "req_b": 1}
        context = SubmittedStepContext(
            req_ids=["req_a", "req_b"],
            req_id_to_index=captured_map,
            request_states=(),
        )
        grammar_output = SimpleNamespace(structured_output_request_ids=["req_a"])

        controller.complete_non_dp_decode_step(
            submission=self._submission(),
            model_input=SimpleNamespace(),
            grammar_output=grammar_output,
            context=context,
        )

        grammar_outputs = controller.captured["grammar_outputs"]
        assert grammar_outputs == [(grammar_output, captured_map)]
        # Guard against regressing to the live mapping.
        assert grammar_outputs[0][1] is captured_map

    def test_grammar_payload_is_none_without_grammar(self):
        controller = self._controller_with_runner({"req_a": 0})
        context = SubmittedStepContext(
            req_ids=["req_a"],
            req_id_to_index={"req_a": 0},
            request_states=(),
        )

        controller.complete_non_dp_decode_step(
            submission=self._submission(),
            model_input=SimpleNamespace(),
            grammar_output=None,
            context=context,
        )

        assert controller.captured["grammar_outputs"] == [None]
