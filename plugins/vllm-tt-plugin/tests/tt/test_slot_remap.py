# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for InputBatch slot-remap reset on slot reuse.

Pure host tests (no device). ``_slot_remap`` carries the condense remap that
the device sampler uses to reindex per-slot seed state. When a recycled slot is
handed to a new request, the previous occupant's remap must be cleared so the
new request's freshly reset seed state is not reindexed from a stale slot.
"""

from importlib import import_module

import torch

from vllm.sampling_params import SamplingParams

input_batch_module = import_module("vllm_tt_plugin.input_batch")
model_runner_module = import_module("vllm_tt_plugin.model_runner")
InputBatch = input_batch_module.InputBatch
CachedRequestState = input_batch_module.CachedRequestState
_slot_remap_for_model_input = model_runner_module._slot_remap_for_model_input


def _make_input_batch(max_num_reqs: int = 4) -> "InputBatch":
    return InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=32,
        max_num_batched_tokens=32,
        vocab_size=64,
        block_sizes=[16],
        kernel_block_sizes=[16],
    )


def _make_request(req_id: str) -> "CachedRequestState":
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1, 2, 3],
        mm_features=[],
        sampling_params=SamplingParams(temperature=1.0, seed=123),
        generator=None,
        block_ids=([0],),
        num_computed_tokens=0,
        output_token_ids=[],
    )


class TestSlotRemap:
    def test_reset_clears_only_target_slot(self):
        batch = _make_input_batch(max_num_reqs=4)
        # Simulate a condense that moved data into slots 0..3.
        batch._slot_remap = torch.tensor([3, 2, 1, 0], dtype=torch.int32)

        batch.reset_slot_remap_for_new_request(2)

        # Slot 2 is reset to identity; the rest of the remap is untouched.
        assert torch.equal(
            batch._slot_remap, torch.tensor([3, 2, 2, 0], dtype=torch.int32)
        )

    def test_pop_returns_remap_and_resets_to_identity(self):
        batch = _make_input_batch(max_num_reqs=4)
        batch._slot_remap = torch.tensor([3, 2, 1, 0], dtype=torch.int32)

        popped = batch.pop_slot_remap()

        assert torch.equal(popped, torch.tensor([3, 2, 1, 0], dtype=torch.int32))
        assert torch.equal(batch._slot_remap, torch.arange(4, dtype=torch.int32))

    def test_add_request_resets_reused_slot(self):
        batch = _make_input_batch(max_num_reqs=4)
        # Populate slots 0..1 with initial occupants.
        batch.add_request(_make_request("req_0"), req_index=0)
        batch.add_request(_make_request("req_1"), req_index=1)
        # A later condense leaves a non-identity remap on the live slots.
        batch._slot_remap = torch.tensor([3, 2, 1, 0], dtype=torch.int32)

        # Slot 1's occupant finishes and a new request reuses the slot.
        batch.add_request(_make_request("new_req"), req_index=1)

        # Only slot 1's stale remap is cleared; other slots are preserved so a
        # genuine condense remap for them still reaches the device sampler.
        popped = batch.pop_slot_remap()
        assert popped[1].item() == 1
        assert torch.equal(popped, torch.tensor([3, 1, 1, 0], dtype=torch.int32))

    def test_model_input_can_peek_remap_without_consuming(self):
        batch = _make_input_batch(max_num_reqs=4)
        batch._slot_remap = torch.tensor([3, 2, 1, 0], dtype=torch.int32)

        remap = _slot_remap_for_model_input(
            batch,
            is_prompt=False,
            consume=False,
        )

        assert remap is not None
        assert torch.equal(remap, torch.tensor([3, 2, 1, 0], dtype=torch.int32))
        assert torch.equal(
            batch._slot_remap,
            torch.tensor([3, 2, 1, 0], dtype=torch.int32),
        )

    def test_model_input_consumes_remap_only_when_requested(self):
        batch = _make_input_batch(max_num_reqs=4)
        batch._slot_remap = torch.tensor([3, 2, 1, 0], dtype=torch.int32)

        remap = _slot_remap_for_model_input(
            batch,
            is_prompt=False,
            consume=True,
        )

        assert remap is not None
        assert torch.equal(remap, torch.tensor([3, 2, 1, 0], dtype=torch.int32))
        assert torch.equal(batch._slot_remap, torch.arange(4, dtype=torch.int32))
