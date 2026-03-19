# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai_harmony import Message

from vllm.entrypoints import harmony_utils
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    serialize_message,
    serialize_messages,
)


def test_serialize_message() -> None:
    dict_value = {"a": 1, "b": "2"}
    assert serialize_message(dict_value) == dict_value

    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 1"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_message(msg) == msg_value


def test_serialize_messages() -> None:
    assert serialize_messages(None) is None
    assert serialize_messages([]) is None

    dict_value = {"a": 3, "b": "4"}
    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 2"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_messages([msg, dict_value]) == [msg_value, dict_value]


def test_harmony_default_stop_tokens_respect_ignore_eos() -> None:
    default_sampling_params = {"stop_token_ids": [11, 12]}

    chat_params = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        ignore_eos=True,
        stop_token_ids=[99],
    ).to_sampling_params(
        max_tokens=16,
        logits_processor_pattern=None,
        default_sampling_params=default_sampling_params,
    )
    assert set(chat_params.stop_token_ids) == {99}

    completion_params = CompletionRequest(
        prompt="hi",
        ignore_eos=True,
        stop_token_ids=[77],
    ).to_sampling_params(
        max_tokens=16,
        logits_processor_pattern=None,
        default_sampling_params=default_sampling_params,
    )
    assert set(completion_params.stop_token_ids) == {77}


def test_parse_output_into_messages_can_skip_assistant_action_stop(
    monkeypatch,
) -> None:
    processed_tokens: list[int] = []

    class DummyParser:
        messages: list[object] = []
        current_content = ""

        def process(self, token_id: int) -> None:
            processed_tokens.append(token_id)

    monkeypatch.setattr(
        harmony_utils,
        "get_streamable_parser_for_assistant",
        lambda: DummyParser(),
    )
    monkeypatch.setattr(
        harmony_utils,
        "get_stop_tokens_for_assistant_actions",
        lambda: [2],
    )

    harmony_utils.parse_output_into_messages([1, 2, 3])
    assert processed_tokens == [1, 2]

    processed_tokens.clear()
    harmony_utils.parse_output_into_messages(
        [1, 2, 3],
        stop_on_assistant_action=False,
    )
    assert processed_tokens == [1, 2, 3]
