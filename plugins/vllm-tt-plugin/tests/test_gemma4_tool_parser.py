# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the Gemma4 tool-call parser.

The parser is stateless for ``extract_tool_calls`` and does not touch the
tokenizer there, so the tests instantiate it with ``tokenizer=None`` and pass
``request=None`` (unused by extraction).
"""

import json

import pytest
from vllm_tt_plugin.gemma4_tool_parser import Gemma4ToolParser


@pytest.fixture
def parser() -> Gemma4ToolParser:
    return Gemma4ToolParser(tokenizer=None)


def test_no_tool_call_returns_content(parser: Gemma4ToolParser):
    out = "just a normal answer, no tools here"
    result = parser.extract_tool_calls(out, request=None)
    assert result.tools_called is False
    assert result.tool_calls == []
    assert result.content == out


def test_single_call_string_arg(parser: Gemma4ToolParser):
    out = '<|tool_call>call:get_weather{location:<|"|>Paris, FR<|"|>}<tool_call|>'
    result = parser.extract_tool_calls(out, request=None)
    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"location": "Paris, FR"}


def _args(parser: Gemma4ToolParser, out: str) -> dict:
    result = parser.extract_tool_calls(out, request=None)
    return json.loads(result.tool_calls[0].function.arguments)


def test_mixed_scalar_args(parser: Gemma4ToolParser):
    out = '<|tool_call>call:f{flag:true,n:3,r:2.5,s:<|"|>hi<|"|>,z:false}<tool_call|>'
    assert _args(parser, out) == {
        "flag": True,
        "n": 3,
        "r": 2.5,
        "s": "hi",
        "z": False,
    }


def test_nested_object_and_arrays(parser: Gemma4ToolParser):
    out = (
        '<|tool_call>call:g{a:{x:1,y:<|"|>q<|"|>},'
        'xs:[1,2,3],ss:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>'
    )
    assert _args(parser, out) == {
        "a": {"x": 1, "y": "q"},
        "xs": [1, 2, 3],
        "ss": ["a", "b"],
    }


def test_empty_args(parser: Gemma4ToolParser):
    out = "<|tool_call>call:ping{}<tool_call|>"
    result = parser.extract_tool_calls(out, request=None)
    assert result.tool_calls[0].function.name == "ping"
    assert json.loads(result.tool_calls[0].function.arguments) == {}


def test_string_value_with_delimiters(parser: Gemma4ToolParser):
    # The quote token makes the string atomic, so embedded {}/[]/, are literal.
    out = '<|tool_call>call:h{q:<|"|>a{b},c[d]<|"|>}<tool_call|>'
    assert _args(parser, out) == {"q": "a{b},c[d]"}


def test_multiple_calls_and_leading_content(parser: Gemma4ToolParser):
    out = (
        "Sure!<|tool_call>call:a{x:1}<tool_call|>"
        '<|tool_call>call:b{y:<|"|>v<|"|>}<tool_call|>'
    )
    result = parser.extract_tool_calls(out, request=None)
    assert result.tools_called is True
    assert result.content == "Sure!"
    assert [c.function.name for c in result.tool_calls] == ["a", "b"]
    assert json.loads(result.tool_calls[0].function.arguments) == {"x": 1}
    assert json.loads(result.tool_calls[1].function.arguments) == {"y": "v"}


def test_incomplete_block_is_ignored(parser: Gemma4ToolParser):
    # No closing <tool_call|>: not emitted as a completed call.
    out = "<|tool_call>call:a{x:1"
    result = parser.extract_tool_calls(out, request=None)
    assert result.tools_called is False


def test_streaming_assembles_name_and_args(parser: Gemma4ToolParser):
    full = (
        "<|tool_call>call:get_weather"
        '{location:<|"|>Paris<|"|>,units:<|"|>c<|"|>}<tool_call|>'
    )
    # Chunk on boundaries that do not split a special token.
    chunks = [
        "<|tool_call>",
        "call:get_weather{location:",
        '<|"|>Paris<|"|>',
        ",units:",
        '<|"|>c<|"|>',
        "}",
        "<tool_call|>",
    ]
    assert "".join(chunks) == full

    name = None
    args_acc = ""
    prev = ""
    for chunk in chunks:
        cur = prev + chunk
        delta = parser.extract_tool_calls_streaming(
            prev, cur, chunk, [], [], [], request=None
        )
        if delta is not None and delta.tool_calls:
            fn = delta.tool_calls[0].function or {}
            if fn.get("name"):
                name = fn["name"]
            if fn.get("arguments"):
                args_acc += fn["arguments"]
        prev = cur

    assert name == "get_weather"
    assert json.loads(args_acc) == {"location": "Paris", "units": "c"}
