# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for TT structured-output grammar bitmask reordering.

These tests use synthetic data only (no device required). They cover the
async-scheduling case where grammar output must be interpreted with the
submit-time request-to-slot mapping, not the runner's current live mapping.
"""

from importlib import import_module
from types import SimpleNamespace

import numpy as np
import torch

model_runner_module = import_module("vllm_tt_plugin.model_runner")
TTModelRunner = model_runner_module.TTModelRunner


class TestReorderGrammarBitmask:
    def _runner_with_live_mapping(self, req_id_to_index: dict[str, int]):
        runner = object.__new__(TTModelRunner)
        runner.input_batch = SimpleNamespace(req_id_to_index=req_id_to_index)
        return runner

    def test_uses_captured_mapping_when_provided(self):
        runner = self._runner_with_live_mapping(
            {
                "req_a": 1,
                "req_b": 0,
            }
        )
        captured_req_id_to_index = {
            "req_a": 0,
            "req_b": 1,
        }
        grammar_output = SimpleNamespace(
            structured_output_request_ids=["req_a", "req_b"],
            grammar_bitmask=np.array(
                [
                    [0x11111111, 0x22222222],
                    [0x33333333, 0x44444444],
                ],
                dtype=np.int32,
            ),
        )

        reordered = runner._reorder_grammar_bitmask(
            (grammar_output, captured_req_id_to_index),
            batch_length=2,
        )

        assert reordered is not None
        expected = torch.tensor(
            [
                [0x11111111, 0x22222222],
                [0x33333333, 0x44444444],
            ],
            dtype=torch.int32,
        )
        assert torch.equal(reordered, expected)

    def test_falls_back_to_live_mapping_without_capture(self):
        runner = self._runner_with_live_mapping(
            {
                "req_a": 1,
                "req_b": 0,
            }
        )
        grammar_output = SimpleNamespace(
            structured_output_request_ids=["req_a", "req_b"],
            grammar_bitmask=np.array(
                [
                    [0x11111111, 0x22222222],
                    [0x33333333, 0x44444444],
                ],
                dtype=np.int32,
            ),
        )

        reordered = runner._reorder_grammar_bitmask(
            grammar_output,
            batch_length=2,
        )

        assert reordered is not None
        expected = torch.tensor(
            [
                [0x33333333, 0x44444444],
                [0x11111111, 0x22222222],
            ],
            dtype=torch.int32,
        )
        assert torch.equal(reordered, expected)

    def test_unstructured_rows_allow_all_tokens(self):
        runner = self._runner_with_live_mapping(
            {
                "plain_req": 0,
                "structured_req": 1,
            }
        )
        grammar_output = SimpleNamespace(
            structured_output_request_ids=["structured_req"],
            grammar_bitmask=np.array([[0x12345678, 0x7FFFFFFF]], dtype=np.int32),
        )

        reordered = runner._reorder_grammar_bitmask(grammar_output, batch_length=2)

        assert reordered is not None
        assert torch.equal(
            reordered[0],
            torch.full((2,), -1, dtype=torch.int32),
        )
        assert torch.equal(
            reordered[1],
            torch.tensor([0x12345678, 0x7FFFFFFF], dtype=torch.int32),
        )
