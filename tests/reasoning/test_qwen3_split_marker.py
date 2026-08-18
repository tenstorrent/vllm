"""Reproduce the streaming </think> leak with a split delta boundary.

test_streaming_json_object_no_reasoning_leak fails because the literal </think>
tag reaches `content`, so the accumulated content is not valid JSON. The parser
is pure Python, so the boundary case can be reproduced with no device and no
model: feed deltas that split "</think>" across a chunk boundary, exactly as the
detokenizer's stop-string buffering does.

Run: pytest test_qwen3_split_marker.py -v
"""

import pytest

from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser


class _Tok:
    """Minimal tokenizer stub: the parser only needs a vocab lookup."""

    def __init__(self):
        self.vocab = {"<think>": 1, "</think>": 2, "<tool_call>": 3, "</tool_call>": 4}

    def get_vocab(self):
        return self.vocab


def _parser():
    p = Qwen3ReasoningParser(_Tok())
    return p


def _drive(parser, deltas, ids_for):
    """Feed deltas in order, accumulating reasoning/content like the server does."""
    prev_text = ""
    prev_ids: list[int] = []
    reasoning, content = "", ""
    for d in deltas:
        ids = ids_for(d)
        msg = parser.extract_reasoning_streaming(
            previous_text=prev_text,
            current_text=prev_text + d,
            delta_text=d,
            previous_token_ids=list(prev_ids),
            current_token_ids=list(prev_ids) + ids,
            delta_token_ids=ids,
        )
        if msg is not None:
            if getattr(msg, "reasoning", None):
                reasoning += msg.reasoning
            if getattr(msg, "content", None):
                content += msg.content
        prev_text += d
        prev_ids += ids
    return reasoning, content


def test_marker_whole_in_one_delta_is_clean():
    """Baseline: when </think> arrives intact, content is clean JSON."""
    p = _parser()
    deltas = ["thinking hard", '</think>{"a": 1}']
    ids_for = lambda d: [2] if "</think>" in d else [9]  # noqa: E731
    reasoning, content = _drive(p, deltas, ids_for)
    assert "</think>" not in content, content
    assert content == '{"a": 1}', content


def test_marker_split_across_deltas_must_not_leak():
    """The failing case: detokenizer splits "</think>" across two deltas.

    The tag's head arrives in one chunk and its tail in the next, with the end
    TOKEN ID already consumed. The tail must not be emitted as content.
    """
    p = _parser()
    # "</thi" then "nk>" -- the id lands with the first fragment
    deltas = ["thinking hard</thi", 'nk>{"a": 1}']
    ids_for = lambda d: [2] if d.endswith("</thi") else [9]  # noqa: E731
    reasoning, content = _drive(p, deltas, ids_for)
    assert "nk>" not in content, "leaked tag tail into content: %r" % content
    assert "</think>" not in content, content
    assert content == '{"a": 1}', "content must be clean JSON, got %r" % content


def test_marker_split_three_ways_must_not_leak():
    p = _parser()
    deltas = ["reasoning", "</", "think>", '{"b": 2}']
    ids_for = lambda d: [2] if d == "</" else [9]  # noqa: E731
    reasoning, content = _drive(p, deltas, ids_for)
    assert "think>" not in content, "leaked into content: %r" % content
    assert content == '{"b": 2}', content


def test_content_after_marker_still_flows():
    """Regression guard: normal content after the marker keeps streaming."""
    p = _parser()
    deltas = ["r", '</think>{"a"', ": 1}"]
    ids_for = lambda d: [2] if "</think>" in d else [9]  # noqa: E731
    reasoning, content = _drive(p, deltas, ids_for)
    assert content == '{"a": 1}', content
