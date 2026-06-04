# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json

import regex as re

CHOICES = ["red", "green", "blue", "yellow"]
REGEX = r"LANE-[0-9]"
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok"]},
        "lane": {"type": "integer", "minimum": 0, "maximum": 31},
    },
    "required": ["status", "lane"],
    "additionalProperties": False,
}


async def _send_choice_request(async_client, model: str, request_id: int) -> str:
    response = await async_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Pick one color for request {request_id}.",
            }
        ],
        max_completion_tokens=8,
        temperature=0,
        extra_body={"structured_outputs": {"choice": CHOICES}},
    )
    content = response.choices[0].message.content
    assert content in CHOICES
    return content


async def _send_regex_request(async_client, model: str, request_id: int) -> str:
    response = await async_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Return one code matching {REGEX} for request {request_id}."
                ),
            }
        ],
        max_completion_tokens=16,
        temperature=0,
        extra_body={"structured_outputs": {"regex": REGEX}},
    )
    content = response.choices[0].message.content
    assert content is not None
    assert re.fullmatch(REGEX, content) is not None
    return content


async def _send_json_request(async_client, model: str, request_id: int) -> dict:
    response = await async_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    "Return a compact JSON object with status ok and lane "
                    f"{request_id % 32}."
                ),
            }
        ],
        max_completion_tokens=64,
        temperature=0,
        extra_body={"structured_outputs": {"json": JSON_SCHEMA}},
    )
    content = response.choices[0].message.content
    assert content is not None
    parsed = json.loads(content)
    assert set(parsed) == {"status", "lane"}
    assert parsed["status"] == "ok"
    assert isinstance(parsed["lane"], int)
    assert 0 <= parsed["lane"] <= 31
    return parsed


async def _send_plain_request(async_client, model: str, request_id: int) -> str:
    response = await async_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Reply with a short sentence for request {request_id}.",
            }
        ],
        max_completion_tokens=16,
        temperature=0,
    )
    content = response.choices[0].message.content
    assert content
    return content


def test_dp1_full_capacity_mixes_structured_and_plain_requests(
    tt_server,
    tt_model_name,
    max_batch_size,
):
    async def _run() -> None:
        async_client = tt_server.get_async_client()
        request_count = min(max_batch_size, 32)
        senders = [
            _send_choice_request,
            _send_regex_request,
            _send_json_request,
            _send_plain_request,
        ]

        tasks = [
            senders[request_id % len(senders)](
                async_client,
                tt_model_name,
                request_id,
            )
            for request_id in range(request_count)
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == request_count

    asyncio.run(_run())
