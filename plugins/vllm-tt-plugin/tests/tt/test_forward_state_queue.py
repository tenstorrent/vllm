# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the non-DP forward-state queue and empty-batch path.

The split decode protocol is two calls: ``execute_model`` pushes a forward
state onto ``_forward_state_queue`` and ``sample_tokens`` pops it. These host
tests cover the queue's FIFO contract and the zero-batch short-circuit, neither
of which touches the device.
"""

from collections import deque
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

model_runner_module = import_module("vllm_tt_plugin.model_runner")
TTModelRunner = model_runner_module.TTModelRunner
TTPendingDecodeState = model_runner_module.TTPendingDecodeState


class TestEmptyForwardOutput:
    def test_zero_batch_short_circuits_to_empty_output(self):
        runner = object.__new__(TTModelRunner)
        runner.vocab_size = 8
        model_input = SimpleNamespace(
            unpadded_batch_size=[0],
            tt_sampling_params=SimpleNamespace(),
            perform_device_sampling=False,
            prompt_lens=None,  # decode
        )

        out = runner.execute_forward_with_model_input(model_input)

        # No device forward ran: an empty (0-row) payload is returned with the
        # decode shape and the caller's batch sizing preserved.
        assert out.tt_out.shape == (0, 1, runner.vocab_size)
        assert out.tt_log_probs is None
        assert out.batch_size_per_dp == [0]
        assert out.is_decode is True
        assert out.device_sampling_deferred is False


class TestForwardStateQueueFifo:
    def _runner_with_recording_sampler(self):
        runner = object.__new__(TTModelRunner)
        runner._forward_state_queue = deque()
        sampled: list = []

        def _fake_sample_forward_output(forward_output, grammar):
            sampled.append(forward_output)
            return [torch.zeros((1, 1), dtype=torch.int32)], [None]

        runner.sample_forward_output = _fake_sample_forward_output
        runner.apply_and_build_runner_output = lambda *args, **kwargs: SimpleNamespace()
        runner.sampled = sampled
        return runner

    def test_sample_tokens_pops_in_submit_order(self):
        runner = self._runner_with_recording_sampler()
        first = SimpleNamespace(label="first")
        second = SimpleNamespace(label="second")
        # execute_model would append in submit order.
        runner._forward_state_queue.append(first)
        runner._forward_state_queue.append(second)

        runner.sample_tokens(None)
        runner.sample_tokens(None)

        assert [fo.label for fo in runner.sampled] == ["first", "second"]
        assert not runner._forward_state_queue

    def test_sample_tokens_asserts_on_empty_queue(self):
        runner = self._runner_with_recording_sampler()
        with pytest.raises(AssertionError):
            runner.sample_tokens(None)
