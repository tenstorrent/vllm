# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from typing import Any
from urllib import error, request

import pytest


def _post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        pytest.fail(f"POST {path} failed with HTTP {exc.code}: {body}")


def _tokenize(base_url: str, model: str, prompt: str) -> dict[str, Any]:
    return _post_json(
        base_url,
        "/tokenize",
        {
            "model": model,
            "prompt": prompt,
            "add_special_tokens": False,
        },
    )


def test_completion_prefill_close_to_max_model_len(
    tt_server_url,
    tt_model_name,
):
    seed = _tokenize(
        tt_server_url,
        tt_model_name,
        "The quick brown fox jumps over the lazy dog. ",
    )
    max_model_len = seed["max_model_len"]

    max_tokens = 1
    target_prompt_tokens = max_model_len - max_tokens
    if target_prompt_tokens <= 0:
        pytest.skip(f"max_model_len={max_model_len} is too small for this test")

    seed_tokens = seed["tokens"]
    assert seed_tokens, "tokenization probe should produce at least one token"
    repeats = (target_prompt_tokens + len(seed_tokens) - 1) // len(seed_tokens)
    prompt_token_ids = (seed_tokens * repeats)[:target_prompt_tokens]

    response = _post_json(
        tt_server_url,
        "/v1/completions",
        {
            "model": tt_model_name,
            "prompt": prompt_token_ids,
            "max_tokens": max_tokens,
            "temperature": 0,
            "add_special_tokens": False,
        },
    )

    usage = response["usage"]
    assert usage["prompt_tokens"] == target_prompt_tokens
    assert usage["total_tokens"] <= max_model_len
    assert len(response["choices"]) == 1
