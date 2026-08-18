"""Device-free reproduction of the streaming reasoning->content boundary bug.

Traces taken verbatim from Qwen3-32B on wh-glx6u-08 with --reasoning-parser
qwen3. The recorded CONTENT deltas were:

    ['r.', '\\n</think', '>{', '"l', 'ocatio', 'n"']

i.e. the tail of the reasoning ('r.') and the marker itself both leaked into
content, so response_format=json_object content did not parse as JSON.

The cause is that the token id for </think> lands while the text for it -- and
for the reasoning before it -- is still buffered in the detokenizer. Keying the
boundary off `end_token_id in previous_token_ids` therefore flips to content
mode early. These tests encode that lag explicitly: the end token id is placed
in an EARLIER delta than the text it renders to.

Run: pytest test_qwen3_boundary.py -v
"""

import pytest

from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser

END_ID = 2
TOOL_ID = 3
PLAIN = 9


class _Tok:
    def __init__(self):
        self.vocab = {"<think>": 1, "</think>": END_ID,
                      "<tool_call>": TOOL_ID, "</tool_call>": 4}

    def get_vocab(self):
        return self.vocab


def _drive(deltas_with_ids):
    """Feed (text, ids) deltas; return accumulated (reasoning, content)."""
    parser = Qwen3ReasoningParser(_Tok())
    prev_text, prev_ids = "", []
    reasoning, content = "", ""
    for text, ids in deltas_with_ids:
        msg = parser.extract_reasoning_streaming(
            previous_text=prev_text,
            current_text=prev_text + text,
            delta_text=text,
            previous_token_ids=list(prev_ids),
            current_token_ids=list(prev_ids) + list(ids),
            delta_token_ids=list(ids),
        )
        if msg is not None:
            if getattr(msg, "reasoning", None):
                reasoning += msg.reasoning
            if getattr(msg, "content", None):
                content += msg.content
        prev_text += text
        prev_ids += list(ids)
    return reasoning, content


def test_id_lag_does_not_leak_reasoning_or_marker():
    """The exact hardware trace: id for </think> arrives before its text."""
    deltas = [
        ("Let me check the weathe", [PLAIN]),
        ("r.", [END_ID]),          # <-- id lands here, text still buffered
        ("\n</think", [PLAIN]),
        ('>{"location":"Boston"}', [PLAIN]),
    ]
    reasoning, content = _drive(deltas)

    assert "</think" not in content, "marker leaked into content: %r" % content
    assert "r." not in content, "reasoning tail leaked into content: %r" % content
    assert content == '{"location":"Boston"}', content
    assert reasoning == "Let me check the weather.\n", repr(reasoning)


def test_marker_split_across_deltas():
    deltas = [
        ("thinking hard</thi", [END_ID]),
        ('nk>{"a": 1}', [PLAIN]),
    ]
    reasoning, content = _drive(deltas)
    assert "nk>" not in content, content
    assert content == '{"a": 1}', content
    assert reasoning == "thinking hard", repr(reasoning)


def test_marker_whole_in_one_delta():
    deltas = [
        ("reasoning", [PLAIN]),
        ('</think>{"a": 1}', [END_ID]),
    ]
    reasoning, content = _drive(deltas)
    assert content == '{"a": 1}', content
    assert reasoning == "reasoning", repr(reasoning)


def test_content_keeps_streaming_after_marker():
    deltas = [
        ("r", [PLAIN]),
        ('</think>{"a"', [END_ID]),
        (": 1}", [PLAIN]),
    ]
    reasoning, content = _drive(deltas)
    assert content == '{"a": 1}', content


def test_tool_call_tag_ends_reasoning_and_stays_in_content():
    deltas = [
        ("I should call it", [PLAIN]),
        ('<tool_call>{"name": "x"}', [TOOL_ID]),
    ]
    reasoning, content = _drive(deltas)
    assert content == '<tool_call>{"name": "x"}', content
    assert reasoning == "I should call it", repr(reasoning)


def test_no_marker_is_all_reasoning():
    deltas = [("still thinking", [PLAIN]), (" and thinking", [PLAIN])]
    reasoning, content = _drive(deltas)
    assert content == "", content
    assert reasoning == "still thinking and thinking", repr(reasoning)


@pytest.mark.parametrize("split_at", list(range(1, 8)))
def test_every_split_point_of_the_marker(split_at):
    """</think> chunked at every possible boundary must never leak."""
    marker = "</think>"
    deltas = [
        ("why" + marker[:split_at], [END_ID]),
        (marker[split_at:] + '{"k": 1}', [PLAIN]),
    ]
    reasoning, content = _drive(deltas)
    assert content == '{"k": 1}', "split_at=%d content=%r" % (split_at, content)
    assert reasoning == "why", "split_at=%d reasoning=%r" % (split_at, reasoning)
