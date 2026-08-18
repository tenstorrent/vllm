# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3/Qwen3.5 model family.

    The Qwen3 model family uses <think>...</think> tokens to denote reasoning
    text. Starting with Qwen3.5, the chat template places <think> in the
    prompt so only </think> appears in the generated output. The model
    provides a strict switch to disable reasoning output via the
    'enable_thinking=False' parameter.

    When thinking is disabled, the template places <think>\\n\\n</think>\\n\\n
    in the prompt. The serving layer detects this via prompt_is_reasoning_end
    and routes deltas as content without calling the streaming parser.

    NOTE: Models up to the 2507 release (e.g., Qwen/Qwen3-235B-A22B-Instruct-2507)
    use an older chat template where the model generates <think> itself.
    This parser handles both styles: if <think> appears in the generated output
    it is stripped before extraction (non-streaming) or skipped (streaming).

    NOTE: Qwen3.5 models may emit <tool_call> inside the thinking block
    without closing </think> first. <tool_call> is treated as an implicit
    end of reasoning, matching the approach in KimiK2ReasoningParser.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Qwen3 defaults to thinking enabled; only treat output as
        # pure content when the user explicitly disables it.
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)

        self._tool_call_tag = "<tool_call>"
        self._tool_call_token_id = self.vocab.get(self._tool_call_tag)
        self._tool_call_end_tag = "</tool_call>"
        self._tool_call_end_token_id = self.vocab.get(self._tool_call_end_tag)

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        tool_call_token_id = self._tool_call_token_id
        tool_call_end_token_id = self._tool_call_end_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            token_id = input_ids[i]
            if token_id == start_token_id:
                # Found <think> before </think> or <tool_call>
                return False
            if token_id == end_token_id:
                return True
            if tool_call_token_id is not None and token_id == tool_call_token_id:
                # Only treat as implicit reasoning end if this <tool_call>
                # is NOT followed by </tool_call>.  Paired occurrences are
                # template examples in the prompt, not model output.
                if tool_call_end_token_id is not None and any(
                    input_ids[j] == tool_call_end_token_id
                    for j in range(i + 1, len(input_ids))
                ):
                    continue
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if super().is_reasoning_end_streaming(input_ids, delta_ids):
            return True
        if self._tool_call_token_id is not None:
            return self._tool_call_token_id in delta_ids
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        """
        result = super().extract_content_ids(input_ids)
        if result:
            return result
        # Fall back: content starts at <tool_call> (implicit reasoning end).
        if (
            self._tool_call_token_id is not None
            and self._tool_call_token_id in input_ids
        ):
            tool_call_index = (
                len(input_ids) - 1 - input_ids[::-1].index(self._tool_call_token_id)
            )
            return input_ids[tool_call_index:]
        return []

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        The <think> token is placed in the prompt by the chat template,
        so typically only </think> appears in the generated output.
        If <think> is present (e.g. from a different template), it is
        stripped before extraction.

        When thinking is explicitly disabled and no </think> appears,
        returns (None, model_output) — all output is content.
        Otherwise (thinking enabled, default), a missing </think> means
        the output was truncated and everything is reasoning:
        returns (model_output, None).

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Strip <think> if present in the generated output.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.end_token in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            return reasoning, content or None

        if not self.thinking_enabled:
            # Thinking explicitly disabled — treat everything as content.
            return None, model_output

        # No </think> — check for implicit reasoning end via <tool_call>.
        tool_call_index = model_output.find(self._tool_call_tag)
        if tool_call_index != -1:
            reasoning = model_output[:tool_call_index]
            content = model_output[tool_call_index:]
            return reasoning or None, content or None
        # Thinking enabled but no </think>: output was truncated.
        # Everything generated so far is reasoning.
        return model_output, None

    def _split_reasoning_content(self, text: str) -> tuple[str, str]:
        """Split accumulated model output into ``(reasoning, content)``.

        Pure function of TEXT. Deliberately ignores token ids: the id stream
        runs ahead of the detokenizer, so using ids to place the boundary
        emits still-buffered reasoning as content.

        Reasoning ends at whichever marker comes first:
          * ``</think>``    -- consumed, never emitted
          * ``<tool_call>`` -- retained, it belongs to content

        Until a marker is complete, a trailing PREFIX of one is withheld, so a
        marker split across deltas (``'\\n</think'`` + ``'>'``) is never
        emitted piecemeal.
        """
        # Old template / edge case: the model emits <think> itself.
        if self.start_token in text:
            text = text.split(self.start_token, 1)[1]

        markers = [self.end_token]
        if self._tool_call_token_id is not None:
            markers.append(self._tool_call_tag)

        end_index = text.find(self.end_token)
        tool_index = (
            text.find(self._tool_call_tag)
            if self._tool_call_token_id is not None
            else -1
        )

        if end_index >= 0 and (tool_index < 0 or end_index < tool_index):
            # </think> is consumed: it is neither reasoning nor content.
            return text[:end_index], text[end_index + len(self.end_token) :]
        if tool_index >= 0:
            # <tool_call> stays in content for the tool parser to pick up.
            return text[:tool_index], text[tool_index:]

        # No complete marker yet. Withhold any trailing partial marker so it is
        # not emitted as reasoning and then again as part of the marker.
        hold = 0
        for marker in markers:
            for k in range(min(len(marker) - 1, len(text)), 0, -1):
                if text.endswith(marker[:k]):
                    hold = max(hold, k)
                    break
        return (text[: len(text) - hold] if hold else text), ""

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a streaming delta.

        Since <think> is placed in the prompt by the chat template, all
        generated tokens before </think> are reasoning and tokens after
        are content.

        NOTE: When thinking is disabled, no think tokens appear in the
        generated output. The serving layer detects this via
        prompt_is_reasoning_end and routes deltas as content without
        calling this method.
        """
        if self.end_token_id in delta_token_ids and not delta_text:
            # End token in IDs but not in visible text yet. Wait for the
            # detokenizer to flush the buffered text in a later chunk.
            return None

        # Place the boundary from TEXT ONLY.
        #
        # The previous implementation keyed off
        #   `self.end_token_id in previous_token_ids`
        # but the id stream runs AHEAD of the detokenized text, so that flips to
        # content mode while the reasoning tail and the marker itself are still
        # buffered -- emitting both as content and breaking
        # response_format=json_object, whose content must parse as JSON.
        #
        # _split_reasoning_content() is a pure function of the accumulated text,
        # so emitting the difference between the split of previous_text and the
        # split of previous_text + delta_text is monotonic by construction and
        # can never emit marker text, however the marker is chunked.
        prev_reasoning, prev_content = self._split_reasoning_content(previous_text)
        cur_reasoning, cur_content = self._split_reasoning_content(
            previous_text + delta_text
        )
        reasoning = cur_reasoning[len(prev_reasoning) :]
        content = cur_content[len(prev_content) :]

        if not reasoning and not content:
            return None
        return DeltaMessage(
            reasoning=reasoning if reasoning else None,
            content=content if content else None,
        )
