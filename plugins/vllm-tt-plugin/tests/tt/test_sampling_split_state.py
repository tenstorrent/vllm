# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side tests for sampling-split state handoff."""

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch

model_runner_module = import_module("vllm_tt_plugin.model_runner")
worker_module = import_module("vllm_tt_plugin.worker")

TTModelRunner = model_runner_module.TTModelRunner
TTSamplingParams = model_runner_module.TTSamplingParams
TTWorker = worker_module.TTWorker


class _RecordingBatch:
    def __init__(self, generator: torch.Generator):
        self.sampling = SimpleNamespace(generators={0: generator})
        self.advance_count = 0
        self.pop_count = 0

    def advance_generators(self) -> None:
        self.advance_count += 1
        for generator in self.sampling.generators.values():
            torch.rand(1, generator=generator)

    def pop_slot_remap(self) -> torch.Tensor:
        self.pop_count += 1
        return torch.arange(2, dtype=torch.int32)


def _sampling_params(batch: int = 2) -> TTSamplingParams:
    return TTSamplingParams(
        temperature=torch.ones(batch),
        top_k=torch.ones(batch, dtype=torch.int32),
        top_p=torch.ones(batch),
        presence_penalty=torch.zeros(batch),
        frequency_penalty=torch.zeros(batch),
        repetition_penalty=torch.ones(batch),
        seed=torch.zeros(batch, dtype=torch.int32),
        num_logprobs=torch.full((batch,), -2, dtype=torch.int32),
        enable_log_probs=torch.zeros(batch, dtype=torch.bool),
    )


def _decode_model_input(generator: torch.Generator):
    return SimpleNamespace(
        input_tokens=torch.ones((2, 1), dtype=torch.int32),
        input_positions=torch.arange(2, dtype=torch.int32),
        block_tables_per_group=[torch.zeros((2, 1), dtype=torch.int32)],
        unpadded_batch_size=2,
        tt_sampling_params=_sampling_params(),
        max_num_logprobs=[None],
        allowed_token_ids_mask_list=[None],
        bad_words_token_ids_list=[{}],
        logitsprocs_list=[None],
        generators_list=[{}],
        slot_remap=torch.tensor([1, 0], dtype=torch.int32),
    )


class TestDPSamplingSplitState:
    def _runner(self, batch: _RecordingBatch):
        runner = object.__new__(TTModelRunner)
        runner.scheduler_config = SimpleNamespace(max_num_seqs=2)
        runner._num_kv_cache_groups = 1
        runner.input_batch = batch
        return runner

    def test_dp_host_fallback_captures_and_advances_generators(self):
        generator = torch.Generator().manual_seed(123)
        batch = _RecordingBatch(generator)
        runner = self._runner(batch)

        result = runner.build_dp_decode_gather_input(
            _decode_model_input(generator),
            max_blocks_decode_batch=1,
            any_penalties_inputs=False,
            use_device_sampling=False,
        )

        assert result["host_only_sample_params"]["generators"] == {0: generator}
        assert batch.advance_count == 1

    def test_dp_host_fallback_does_not_consume_slot_remap(self):
        generator = torch.Generator().manual_seed(123)
        batch = _RecordingBatch(generator)
        runner = self._runner(batch)

        runner.build_dp_decode_gather_input(
            _decode_model_input(generator),
            max_blocks_decode_batch=1,
            any_penalties_inputs=False,
            use_device_sampling=False,
        )

        assert batch.pop_count == 0

    def test_dp_device_sampling_consumes_slot_remap_after_packing_it(self):
        generator = torch.Generator().manual_seed(123)
        batch = _RecordingBatch(generator)
        runner = self._runner(batch)

        result = runner.build_dp_decode_gather_input(
            _decode_model_input(generator),
            max_blocks_decode_batch=1,
            any_penalties_inputs=False,
            use_device_sampling=True,
        )

        assert result["int_inputs"][-2:].tolist() == [1, 0]
        assert batch.pop_count == 1


class TestDeviceSamplingLogprobsFallback:
    def test_topk_device_sampling_rejects_requests_beyond_device_topk(self):
        runner = object.__new__(TTModelRunner)
        runner.sample_on_device_mode = "decode_only"
        runner.device_config = SimpleNamespace(num_devices=8)
        runner.parallel_config = SimpleNamespace(data_parallel_size=1)
        runner.supports_topk_logprobs = True
        runner.model_config = SimpleNamespace(logits_processors=[])
        runner.input_batch = SimpleNamespace(
            no_allowed_token_ids=True,
            max_num_logprobs=model_runner_module.MAX_K + 1,
            sampling=SimpleNamespace(
                bad_words_token_ids={},
                has_active_logitsprocs=lambda: False,
            ),
        )

        assert not runner.check_perform_device_sampling(
            is_decode=True,
            has_structured_outputs=False,
        )


class TestDeviceSamplingContract:
    def test_decode_sampling_does_not_filter_unsupported_kwargs(self):
        class ModelWithoutBitmask:
            def sample_decode_on_device(
                self,
                tt_logits,
                sampling_params,
                reset_batch=False,
                prompt_tokens=None,
                output_tokens=None,
                slot_remap=None,
                enable_trace=False,
            ):
                return torch.ones((1, 1), dtype=torch.int32)

        runner = object.__new__(TTModelRunner)
        runner.model = ModelWithoutBitmask()
        runner.trace_mode = "all"
        runner._device_sampling_bitmask = lambda **_: torch.zeros(
            (1, 1), dtype=torch.int32
        )
        model_input = SimpleNamespace(
            reset_batch=False,
            prompt_tokens=torch.zeros((1, 1), dtype=torch.int32),
            output_tokens=torch.zeros((1, 1), dtype=torch.int32),
            slot_remap=None,
        )

        with pytest.raises(TypeError, match="bitmask"):
            runner._sample_deferred_device_output(
                tt_out=torch.zeros((1, 1), dtype=torch.float32),
                model_sampling_params=SimpleNamespace(),
                model_input=model_input,
                batch_size_per_dp=[1],
                is_decode=True,
                grammar_outputs=[None],
            )


class TestWorkerUtilityHooks:
    def test_execute_dummy_batch_is_supported_noop(self):
        worker = object.__new__(TTWorker)

        assert worker.execute_dummy_batch() is None
