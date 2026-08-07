# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Streaming/non-streaming parity for the Gemma-family reasoning parser.

DiffusionGemma commits one whole canvas per step, so streaming deltas carry
hundreds of tokens at once — including the literal ``<|channel>thought\\n``
prefix inside the first delta rather than as its own chunk.
"""

from vllm_tt_plugin.gemma4_reasoning_parser import Gemma4ReasoningParser


class _VocabOnlyTokenizer:
    def get_vocab(self):
        return {
            "<|channel>": 1,
            "<channel|>": 2,
            "<|turn>": 3,
            "<|tool_call>": 4,
            "<|tool_response>": 5,
        }


def _parser() -> Gemma4ReasoningParser:
    return Gemma4ReasoningParser(_VocabOnlyTokenizer())


def test_streaming_whole_canvas_first_delta_strips_channel_and_thought():
    parser = _parser()
    delta_text = "<|channel>thought\nStep 1: consider the problem."
    delta_ids = [1] + [100] * 8

    result = parser.extract_reasoning_streaming(
        previous_text="",
        current_text=delta_text,
        delta_text=delta_text,
        previous_token_ids=[],
        current_token_ids=delta_ids,
        delta_token_ids=delta_ids,
    )

    assert result is not None
    assert result.reasoning == "Step 1: consider the problem."


def test_streaming_whole_canvas_matches_non_streaming_extraction():
    streamed = _parser().extract_reasoning_streaming(
        previous_text="",
        current_text="<|channel>thought\nBecause 6*7.",
        delta_text="<|channel>thought\nBecause 6*7.",
        previous_token_ids=[],
        current_token_ids=[1, 100, 101],
        delta_token_ids=[1, 100, 101],
    )
    reasoning, content = _parser().extract_reasoning(
        "<|channel>thought\nBecause 6*7.<channel|>The answer is 42.",
        request=None,
    )

    assert reasoning == "Because 6*7."
    assert content == "The answer is 42."
    assert streamed is not None and streamed.reasoning == reasoning


def test_streaming_token_by_token_prefix_strip_unchanged():
    parser = _parser()

    first = parser.extract_reasoning_streaming(
        "", "<|channel>", "<|channel>", [], [1], [1]
    )
    assert first is None  # lone start marker is skipped

    second = parser.extract_reasoning_streaming(
        "<|channel>", "<|channel>thought\n", "thought\n", [1], [1, 100], [100]
    )
    assert second is None or not second.reasoning  # thought label buffered

    third = parser.extract_reasoning_streaming(
        "<|channel>thought\n",
        "<|channel>thought\nHi",
        "Hi",
        [1, 100],
        [1, 100, 101],
        [101],
    )
    assert third is not None
    assert third.reasoning == "Hi"
