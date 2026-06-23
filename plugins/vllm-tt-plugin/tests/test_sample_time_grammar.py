# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-only tests for the grammar-at-sample-time seam.

The non-DP/lane-DP forward runs without the grammar bitmask and defers sampling
to ``sample_tokens``; the bitmask is reordered into the live batch/lane layout
and attached at that point. These tests pin the seam that makes that safe: FIFO
pairing of each deferred forward with its grammar, the lane-vs-non-DP reorder
dispatch, and grammar being applied before sampling in every finisher. No device
execution; the index logic runs on plain tensors with fake collaborators.
Requires the ttnn-enabled environment because importing the plugin modules pulls
in ttnn.
"""

from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
from vllm_tt_plugin.model_input import TTModelInput
from vllm_tt_plugin.model_runner import TTModelRunner, _SyncForward

from vllm.v1.core.sched.output import GrammarOutput

VOCAB_WORDS = 2  # int32 words per grammar bitmask row


def _model_input(**overrides) -> TTModelInput:
    """A minimal real ``TTModelInput`` so ``dataclasses.replace`` works."""
    base = dict(
        input_tokens=torch.zeros((1, 1), dtype=torch.int32),
        input_positions=torch.zeros((1,), dtype=torch.int32),
        prompt_lens=None,
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        block_tables_per_group=[torch.zeros((1, 1), dtype=torch.int32)],
        block_tables_per_layer=None,
        unpadded_batch_size=1,
        tt_sampling_params=object(),
        multi_modal_kwargs={},
        perform_device_sampling=False,
        grammar_bitmask=[None],
        logitsprocs_list=[None],
        bad_words_token_ids_list=[{}],
        allowed_token_ids_mask_list=[None],
        generators_list=[{}],
        max_num_logprobs=[None],
    )
    base.update(overrides)
    return TTModelInput(**base)


def _grammar(num_structured: int) -> GrammarOutput:
    bitmask = np.arange(num_structured * VOCAB_WORDS, dtype=np.int32).reshape(
        num_structured, VOCAB_WORDS
    )
    req_ids = [f"req-{i}" for i in range(num_structured)]
    return GrammarOutput(structured_output_request_ids=req_ids, grammar_bitmask=bitmask)


# --------------------------------------------------------------------------
# sample_tokens FIFO pairing
# --------------------------------------------------------------------------


def test_sample_tokens_pops_fifo_and_passes_grammar_through():
    """Each deferred forward must sample with the grammar of its own step."""
    from collections import deque

    seen: list[tuple[str, object]] = []

    def finisher(tag, grammar_output):
        seen.append((tag, grammar_output))
        return tag

    runner = SimpleNamespace(_pending_samples=deque())
    runner._pending_samples.append(partial(finisher, "first"))
    runner._pending_samples.append(partial(finisher, "second"))

    out0 = TTModelRunner.sample_tokens(runner, "grammar-A")
    out1 = TTModelRunner.sample_tokens(runner, "grammar-B")

    assert out0 == "first"
    assert out1 == "second"
    assert seen == [("first", "grammar-A"), ("second", "grammar-B")]
    assert len(runner._pending_samples) == 0


# --------------------------------------------------------------------------
# _reorder_grammar_bitmask dispatch (lane vs non-DP vs none)
# --------------------------------------------------------------------------


def test_reorder_grammar_bitmask_none_returns_none():
    runner = SimpleNamespace()
    result = TTModelRunner._reorder_grammar_bitmask(
        runner, None, _model_input(), lane_total=None
    )
    assert result is None


def test_reorder_grammar_bitmask_lane_path_delegates_to_slot_reorder():
    """``lane_total`` selects the full-slot lane reorder, not the non-DP one."""
    calls: list[tuple] = []
    sentinel = torch.tensor([[7, 7]], dtype=torch.int32)

    def slot_grammar_bitmask(grammar_output, batch_length):
        calls.append((grammar_output, batch_length))
        return sentinel

    runner = SimpleNamespace(
        lane_batch=SimpleNamespace(slot_grammar_bitmask=slot_grammar_bitmask)
    )
    grammar = _grammar(1)

    result = TTModelRunner._reorder_grammar_bitmask(
        runner, grammar, _model_input(), lane_total=32
    )

    assert result is sentinel
    assert calls == [(grammar, 32)]


def test_reorder_grammar_bitmask_non_dp_path_uses_front_packed_reorder():
    """Non-DP (``lane_total=None``) reorders against the front-packed batch."""
    # Two requests, only req-1 structured; batch_length comes from input_tokens.
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_id_to_index={"req-0": 0, "req-1": 1}, num_reqs=2
        )
    )
    grammar = GrammarOutput(
        structured_output_request_ids=["req-1"],
        grammar_bitmask=np.array([[5, 6]], dtype=np.int32),
    )
    model_input = _model_input(input_tokens=torch.zeros((2, 1), dtype=torch.int32))

    result = TTModelRunner._reorder_grammar_bitmask(
        runner, grammar, model_input, lane_total=None
    )

    assert result.shape == (2, VOCAB_WORDS)
    # req-1 (batch index 1) gets its scheduler bitmask row; req-0 is left
    # all-ones (-1 in int32 = every token allowed).
    assert result[1].tolist() == [5, 6]
    assert torch.all(result[0] == -1)


# --------------------------------------------------------------------------
# _apply_grammar_to_input (attach only when a bitmask exists)
# --------------------------------------------------------------------------


def test_apply_grammar_to_input_returns_input_unchanged_when_no_bitmask():
    runner = SimpleNamespace(_reorder_grammar_bitmask=lambda *a, **k: None)
    model_input = _model_input()

    result = TTModelRunner._apply_grammar_to_input(
        runner, model_input, None, lane_total=None
    )

    assert result is model_input


def test_apply_grammar_to_input_wraps_bitmask_in_single_element_list():
    bitmask = torch.tensor([[1, 2]], dtype=torch.int32)
    runner = SimpleNamespace(_reorder_grammar_bitmask=lambda *a, **k: bitmask)
    model_input = _model_input()

    result = TTModelRunner._apply_grammar_to_input(
        runner, model_input, _grammar(1), lane_total=None
    )

    assert result is not model_input  # replace() returns a new frozen instance
    assert len(result.grammar_bitmask) == 1
    assert torch.equal(result.grammar_bitmask[0], bitmask)


# --------------------------------------------------------------------------
# _finish_async_decode (reorder on engine thread, attach to wrapper)
# --------------------------------------------------------------------------


def test_finish_async_decode_skips_bitmask_when_grammar_absent():
    calls: list = []
    wrapper = SimpleNamespace(set_grammar_bitmask=lambda bm: calls.append(bm))
    runner = SimpleNamespace(_reorder_grammar_bitmask=lambda *a, **k: None)

    result = TTModelRunner._finish_async_decode(
        runner, None, wrapper=wrapper, model_input=_model_input(), lane_total=32
    )

    assert result is wrapper
    assert calls == []  # no bitmask set on the wrapper


def test_finish_async_decode_sets_bitmask_when_grammar_present():
    calls: list = []
    bitmask = torch.tensor([[3, 4]], dtype=torch.int32)
    wrapper = SimpleNamespace(set_grammar_bitmask=lambda bm: calls.append(bm))
    runner = SimpleNamespace(_reorder_grammar_bitmask=lambda *a, **k: bitmask)

    result = TTModelRunner._finish_async_decode(
        runner, _grammar(1), wrapper=wrapper, model_input=_model_input(), lane_total=32
    )

    assert result is wrapper
    assert len(calls) == 1 and torch.equal(calls[0], bitmask)


# --------------------------------------------------------------------------
# _finish_nondp_sync / _finish_lane_sync (grammar applied before sampling)
# --------------------------------------------------------------------------


def _sync_forward(model_input: TTModelInput) -> _SyncForward:
    return _SyncForward(
        tt_out=object(),
        tt_log_probs=None,
        sampling_params=object(),
        model_input=model_input,
        batch_size_per_dp=[1],
        perform_device_sampling=False,
        is_decode=True,
    )


def test_finish_nondp_sync_applies_grammar_before_sampling():
    order: list[str] = []

    def apply_grammar(model_input, grammar_output, *, lane_total):
        order.append("grammar")
        assert lane_total is None
        return replace(model_input, reset_batch=True)  # mark it was applied

    def sample_sync(fwd):
        order.append("sample")
        # Grammar must already be on the input the sampler runs against.
        assert fwd.model_input.reset_batch is True
        return [torch.tensor([[42]], dtype=torch.int32)], []

    def build_output(sampled, logprobs, **kwargs):
        order.append("build")
        return SimpleNamespace(sampled=sampled, logprobs=logprobs)

    runner = SimpleNamespace(
        _apply_grammar_to_input=apply_grammar,
        _sample_sync_forward=sample_sync,
        apply_and_build_runner_output=build_output,
    )
    fwd = _sync_forward(_model_input())

    output = TTModelRunner._finish_nondp_sync(runner, _grammar(1), fwd=fwd)

    assert order == ["grammar", "sample", "build"]
    assert output.sampled.tolist() == [[42]]


def test_finish_nondp_sync_with_no_forward_builds_empty_output():
    captured: dict = {}

    def build_output(sampled, logprobs, **kwargs):
        captured["sampled"] = sampled
        captured["logprobs"] = logprobs
        return "empty"

    runner = SimpleNamespace(apply_and_build_runner_output=build_output)

    output = TTModelRunner._finish_nondp_sync(runner, None, fwd=None)

    assert output == "empty"
    assert captured["sampled"].numel() == 0
    assert captured["logprobs"] is None


def test_finish_lane_sync_builds_req_ids_in_scheduled_row_order():
    order: list[str] = []

    def apply_grammar(model_input, grammar_output, *, lane_total):
        order.append("grammar")
        assert lane_total == 32
        return model_input

    def extract_output(
        runner, tt_out, tt_log_probs, model_input, scheduled_rows, *, is_decode
    ):
        order.append("extract")
        return torch.tensor([[11], [12]], dtype=torch.int32), None

    captured: dict = {}

    def build_output(sampled, logprobs, *, req_ids):
        order.append("build")
        captured["req_ids"] = req_ids
        return SimpleNamespace(sampled=sampled)

    # Row index -> req_id; scheduled_rows pick a subset out of slot order.
    lane_req_ids = {4: "req-d", 1: "req-a"}
    runner = SimpleNamespace(
        _apply_grammar_to_input=apply_grammar,
        lane_batch=SimpleNamespace(extract_output=extract_output, req_ids=lane_req_ids),
        apply_and_build_runner_output=build_output,
    )

    output = TTModelRunner._finish_lane_sync(
        runner,
        _grammar(1),
        tt_out=object(),
        tt_log_probs=None,
        model_input=_model_input(),
        scheduled_rows=[4, 1],
        is_decode=True,
        lane_total=32,
    )

    assert order == ["grammar", "extract", "build"]
    # req_ids follow scheduled_rows order, not slot order.
    assert captured["req_ids"] == ["req-d", "req-a"]
    assert output.sampled.tolist() == [[11], [12]]
