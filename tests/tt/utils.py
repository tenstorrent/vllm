# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from dataclasses import dataclass
from typing import Any, Union


@dataclass
class RequestConfig:
    """Configuration for a single request in a batch."""
    prompt: str
    max_tokens: int = 10
    temperature: float = 1.0
    top_k: Union[int, None] = None
    top_p: float = 1.0
    seed: Union[int, None] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0


async def send_request(async_client, model: str, config: RequestConfig):
    """Send a single async request with its own parameters."""
    extra_body: dict[str, Any] = {}
    if config.top_k is not None:
        extra_body["top_k"] = config.top_k
    if config.repetition_penalty != 1.0:
        extra_body["repetition_penalty"] = config.repetition_penalty

    response = await async_client.completions.create(
        model=model,
        prompt=config.prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
        presence_penalty=config.presence_penalty,
        frequency_penalty=config.frequency_penalty,
        extra_body=extra_body if extra_body else None,
    )
    return response.choices[0].text


async def send_batch_concurrent(async_client, model: str,
                                configs: list[RequestConfig]):
    """
    Send multiple requests concurrently with different per-request parameters.
    The vLLM server will batch these together internally.
    """
    tasks = [send_request(async_client, model, cfg) for cfg in configs]
    return await asyncio.gather(*tasks)


def run_concurrent_batch(tt_server, tt_model_name,
                         configs: list[RequestConfig]):
    """
    Synchronous wrapper to run concurrent requests.
    Returns list of output texts in same order as configs.
    """

    async def _run():
        async_client = tt_server.get_async_client()
        try:
            return await send_batch_concurrent(async_client, tt_model_name,
                                               configs)
        finally:
            await async_client.close()

    return asyncio.run(_run())


def assert_varied(results, min_varied, explanation):
    unique_results = set(results)
    assert len(unique_results) >= min_varied, (
        f"{explanation}\n"
        f"Expected at least {min_varied} unique results.\n"
        f"Only {len(unique_results)}/{len(results)} were varied.\n"
        f"Results: {results}")


def assert_pairwise_varied(results1, results2, min_varied, explanation):
    different = [x != y for x, y in zip(results1, results2)]
    assert sum(different) >= min_varied, (f"{explanation}\n"
                                          f"Expected difference on re-run.\n"
                                          f"Results: {results1} + {results2}")


def assert_deterministic(results, explanation):
    unique_results = set(results)
    assert len(unique_results) == 1, (
        f"{explanation}\n"
        f"Expected reproducible outputs.\n"
        f"Got {len(unique_results)} unique results out of {len(results)}.\n"
        f"Results: {results}")
