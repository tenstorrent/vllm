# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Structured-output integration stress test for async batch churn.

This test targets the TT async scheduling risk where a grammar bitmask from one
step is applied after the live batch slots have changed. It mixes constrained
and unconstrained requests, uses per-request mutually exclusive schemas, and
submits more requests than ``max_num_seqs`` to force slot reuse.

Example run:

```
pytest plugins/vllm-tt-plugin/tests/tt/test_structured_async_churn.py \
    --tt-model-name=meta-llama/Llama-3.2-1B-Instruct \
    --tt-server-url=http://localhost:8000 \
    --tt-max-num-seqs=64
```
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import openai
import pytest

COLORS = ("red", "blue", "green", "yellow")


@dataclass(frozen=True)
class StructuredCase:
    request_id: int
    value: str
    max_tokens: int


@dataclass(frozen=True)
class PlainCase:
    request_id: int
    max_tokens: int


def _structured_value(color: str, request_id: int, long_response: bool) -> str:
    base = f"{color}-req-{request_id}"
    if not long_response:
        return base
    return "-".join([base] * 24)


def _value_schema(value: str) -> dict[str, Any]:
    return {
        "type": "string",
        "enum": [value],
    }


def _response_format(case: StructuredCase) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"color_{case.request_id}",
            "strict": True,
            "schema": _value_schema(case.value),
        },
    }


def _parse_json_response(text: str, request_id: int) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Structured response {request_id} is not valid JSON: {text!r}"
        ) from exc


async def _send_structured(
    async_client: openai.AsyncOpenAI,
    model: str,
    case: StructuredCase,
) -> tuple[StructuredCase, str]:
    response = await async_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Return the JSON string {case.value!r}. "
                    "Return no prose and no extra fields."
                ),
            }
        ],
        max_tokens=case.max_tokens,
        temperature=0,
        response_format=_response_format(case),
    )
    choice = response.choices[0]
    content = choice.message.content
    assert content is not None, f"Structured response {case.request_id} is empty"
    assert choice.finish_reason != "length", (
        f"Structured response {case.request_id} was truncated before completing "
        f"valid JSON. Increase the matching structured churn max_tokens option. "
        f"Partial response: {content!r}"
    )
    return case, content


async def _send_plain(
    async_client: openai.AsyncOpenAI,
    model: str,
    case: PlainCase,
) -> tuple[PlainCase, str]:
    response = await async_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Answer in one short sentence. Do not return JSON. "
                    f"Plain request id: {case.request_id}."
                ),
            }
        ],
        max_tokens=case.max_tokens,
        temperature=0,
    )
    content = response.choices[0].message.content
    assert content is not None, f"Plain response {case.request_id} is empty"
    return case, content


class TestStructuredAsyncChurn:
    def test_mixed_structured_plain_churn(
        self,
        tt_server,
        tt_model_name,
        max_batch_size,
        structured_churn_config,
    ):
        structured_count = structured_churn_config["structured_count"]
        plain_count = structured_churn_config["plain_count"]
        short_max_tokens = structured_churn_config["short_max_tokens"]
        long_max_tokens = structured_churn_config["long_max_tokens"]
        plain_max_tokens = structured_churn_config["plain_max_tokens"]
        total_count = structured_count + plain_count
        if total_count <= max_batch_size:
            pytest.skip(
                f"Configured request count ({total_count}) must exceed "
                f"--tt-max-num-seqs ({max_batch_size}) to force churn."
            )

        structured_cases = [
            StructuredCase(
                request_id=i,
                value=_structured_value(
                    COLORS[i % len(COLORS)],
                    request_id=i,
                    long_response=bool(i % 2),
                ),
                max_tokens=long_max_tokens if i % 2 else short_max_tokens,
            )
            for i in range(structured_count)
        ]
        plain_cases = [
            PlainCase(request_id=i, max_tokens=plain_max_tokens)
            for i in range(structured_count, total_count)
        ]

        # Interleave plain requests into the structured stream so unconstrained
        # rows are present while constrained rows are being finalized.
        requests: list[StructuredCase | PlainCase] = []
        plain_idx = 0
        plain_interval = max(1, structured_count // max(plain_count, 1))
        for i, structured_case in enumerate(structured_cases):
            requests.append(structured_case)
            if plain_idx < plain_count and (i + 1) % plain_interval == 0:
                requests.append(plain_cases[plain_idx])
                plain_idx += 1
        while plain_idx < plain_count:
            requests.append(plain_cases[plain_idx])
            plain_idx += 1

        async def run_requests():
            async_client = tt_server.get_async_client()
            try:
                tasks = []
                for case in requests:
                    if isinstance(case, StructuredCase):
                        tasks.append(
                            _send_structured(async_client, tt_model_name, case)
                        )
                    else:
                        tasks.append(_send_plain(async_client, tt_model_name, case))
                return await asyncio.gather(*tasks)
            finally:
                await async_client.close()

        results = asyncio.run(run_requests())

        structured_seen = 0
        plain_seen = 0
        for case, text in results:
            if isinstance(case, StructuredCase):
                structured_seen += 1
                parsed = _parse_json_response(text, case.request_id)
                assert parsed == case.value, (
                    f"Structured response {case.request_id} used wrong JSON value. "
                    f"Expected {case.value!r}, got {parsed!r} from {text!r}"
                )
            else:
                plain_seen += 1
                assert text.strip(), f"Plain response {case.request_id} is empty"

        assert structured_seen == structured_count
        assert plain_seen == plain_count
