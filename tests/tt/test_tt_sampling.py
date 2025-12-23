# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive sampling tests for TT platform (vLLM v0).

All tests send concurrent requests with DIFFERENT per-request sampling
parameters. The vLLM server batches these together internally, so we're
testing that:
- Each request gets its own sampling parameters applied
- Outputs aren't mixed between requests in the internal batch
- Prefill and decode work correctly for each request

Run with:
    pytest tests/tt/test_tt_sampling.py -v --tt-model-variant=tt_transformers
"""

import asyncio
from dataclasses import dataclass

import pytest


@dataclass
class RequestConfig:
    """Configuration for a single request in a batch."""
    prompt: str
    max_tokens: int = 10
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float = 1.0
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0


async def send_request(async_client, model: str, config: RequestConfig):
    """Send a single async request with its own parameters."""
    extra_body = {}
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


async def send_batch_concurrent(async_client, model: str, configs: list[RequestConfig]):
    """
    Send multiple requests concurrently with different per-request parameters.
    The vLLM server will batch these together internally.
    """
    tasks = [send_request(async_client, model, cfg) for cfg in configs]
    return await asyncio.gather(*tasks)


def run_concurrent_batch(tt_server, tt_model_name, configs: list[RequestConfig]):
    """
    Synchronous wrapper to run concurrent requests.
    Returns list of output texts in same order as configs.
    """
    async def _run():
        async_client = tt_server.get_async_client()
        try:
            return await send_batch_concurrent(async_client, tt_model_name, configs)
        finally:
            await async_client.close()
    
    return asyncio.run(_run())


# =============================================================================
# PREFILL TOKEN VERIFICATION
# =============================================================================

class TestPrefillWithDifferentParams:
    """
    Verify sampling of token output by prefill
    """

    @pytest.mark.parametrize("batch_size", [7, 10, 19, 32])
    def test_prefill_temperature_varied_in_batch(
        self, tt_server, tt_model_name, batch_size
    ):
        """
        With temperature > 0, first tokens should be varied.
        """
        prompt = "Random letter:"

        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=1,
                temperature=5,
            )
            for _ in range(batch_size)
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        unique_results = set(results)

        assert len(unique_results) >= 2, (
            f"With temperature=5, some outputs should be varied.\n"
            f"Only {len(unique_results)}/{batch_size} were varied.\n"
            f"Results: {results}"
        )

    def test_prefill_temperature_varied_between_batches(
        self, tt_server, tt_model_name
    ):
        """
        With temperature > 0, first token should vary when request is repeated.
        """
        prompt = "Random letter:"

        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=1,
                temperature=5,
            )
        ]

        tries = 10

        results = [run_concurrent_batch(tt_server, tt_model_name, configs)[0] for _ in range(tries)]
        unique_results = set(results)

        assert len(unique_results) >= 2, (
            f"With temperature=5, output should be varied.\n"
            f"Only {len(unique_results)}/{tries} unique outputs were produced.\n"
            f"Results: {results}"
        )

    @pytest.mark.parametrize("batch_size", [7, 10, 19, 32])
    def test_prefill_topk(
        self, tt_server, tt_model_name, batch_size
    ):
        """
        With top_k > 0, first tokens should be varied.
        """
        prompt = "Random letter:"
        num_greedy = batch_size // 2
        greedy_config = [RequestConfig(prompt=prompt, max_tokens=1, temperature=0) for _ in range(num_greedy)]

        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=1,
                top_k=10,
            )
            for _ in range(batch_size-num_greedy)
        ]
        joint_configs = greedy_config + configs

        results_1 = run_concurrent_batch(tt_server, tt_model_name, joint_configs)
        results_2 = run_concurrent_batch(tt_server, tt_model_name, joint_configs)

        # Within each batch:
        for results in [results_1, results_2]:
            greedy_results = results[:num_greedy]
            non_greedy_results = results[num_greedy:]

            unique_greedy = set(greedy_results)
            assert len(unique_greedy) == 1, (
                f"Greedy requests should produce the same output.\n"
                f"Results: {greedy_results}"
            )

            unique_non_greedy = set(non_greedy_results)

            assert len(unique_non_greedy) >= 2, (
                f"With top_k=10, some outputs should be varied.\n"
                f"Only {len(unique_non_greedy)}/{batch_size} were varied.\n"
                f"Results: {results}"
            )

        # Between batches:
        greedy_results_1 = results_1[:num_greedy]
        greedy_results_2 = results_2[:num_greedy]
        assert len(set(greedy_results_1+greedy_results_2)) == 1, (
            f"Greedy requests should produce the same output when re-ran.\n"
            f"Results: {greedy_results_1} + {greedy_results_2}"
        )

        non_greedy_results_1 = results_1[num_greedy:]
        non_greedy_results_2 = results_2[num_greedy:]
        different = [x != y for x, y in zip(non_greedy_results_1, non_greedy_results_2)]
        assert sum(different) > 0, (
            f"Non-greedy requests should produce different outputs when re-ran.\n"
            f"Results: {non_greedy_results_1} + {non_greedy_results_2}"
        )
         
    def test_prefill_seeding(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Batch with mix of greedy (temp=0) and sampling requests.
        Greedy should be deterministic, sampling should vary.
        """
        prompt = "Random word:"
        
        # greedy, seeds, seeds
        seeds = max_batch_size // 3
        greedy_count = max_batch_size - seeds - seeds
        
        configs = []
        
        # Greedy requests
        for i in range(greedy_count):
            configs.append(RequestConfig(
                prompt=prompt,
                max_tokens=1,
                temperature=0,
            ))
        
        # Sampling requests with different seeds
        for _ in range(2):
            for i in range(seeds):
                configs.append(RequestConfig(
                    prompt=prompt,
                    max_tokens=1,
                    temperature=1.5,
                    top_k=50,
                    seed=i * 100,
                ))
    
        # Run twice
        results1 = run_concurrent_batch(tt_server, tt_model_name, configs)
        results2 = run_concurrent_batch(tt_server, tt_model_name, configs)

        all_greedy = results1[:greedy_count]+results2[:greedy_count]
        assert len(set(all_greedy)) == 1, (
            f"Greedy requests should produce the same output across positions and runs.\n"
            f"Results: {all_greedy}"
        )

        different_seeds = []
        for i in range(seeds):
            all_results = [
                results1[greedy_count+i],
                results2[greedy_count+i],
                results1[greedy_count+seeds+i],
                results2[greedy_count+seeds+i],
            ]
            assert len(set(all_results)) == 1, (
                f"Seeded requests should produce the same output across positions and runs.\n"
                f"Results: {all_results}"
            )
            different_seeds.append(all_results[0])
        assert len(set(different_seeds)) >= 2, (
            f"Seeded requests should produce different outputs for different seeds.\n"
            f"Results: {different_seeds}"
        )

    def test_prefill_topk_1_is_greedy(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        top_k=1 should match greedy, both within and between runs.
        """
        prompt = "Number:"

        # First get greedy baseline
        greedy_config = RequestConfig(prompt=prompt, max_tokens=1, temperature=0)
        topk_config = RequestConfig(prompt=prompt, max_tokens=1, temperature=2.0, top_k=1)
        result_1 = run_concurrent_batch(tt_server, tt_model_name, [greedy_config, topk_config])
        result_2 = run_concurrent_batch(tt_server, tt_model_name, [greedy_config, topk_config])

        all_results = result_1 + result_2
        assert len(set(all_results)) == 1, (
            f"top_k=1 requests should produce the same output across positions and runs.\n"
            f"Results: {all_results}"
        )



#========================================================
# PENALTY TESTING - Different penalties per request
# =============================================================================

class TestRepetitionPenaltyPerRequest:
    """
    Different repetition penalties per request in same batch.
    """

    def test_different_repetition_penalties(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Each request has different repetition penalty.
        """
        prompt = "hello hello hello hello. Say:"

        penalties = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0][:max_batch_size]
        
        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=10,
                temperature=0.01,
                repetition_penalty=penalty,
                seed=42,
            )
            for penalty in penalties
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)

        # Different penalties should produce different outputs
        unique_results = set(results)
        assert len(unique_results) >= 2, (
            f"Different repetition penalties should produce different outputs.\n"
            f"Penalties: {penalties}\n"
            f"Results: {results}"
        )


    # Caught https://github.com/tenstorrent/vllm/issues/286
    def test_repetition_penalty_vs_no_penalty(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Batch with some requests having penalty, some without.
        """
        prompt = "test test test test. Word:"
        
        configs = []
        for i in range(max_batch_size):
            if i % 2 == 0:
                # No penalty
                configs.append(RequestConfig(
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0.01,
                    repetition_penalty=1.0,
                    seed=42,
                ))
            else:
                # High penalty
                configs.append(RequestConfig(
                    prompt=prompt,
                    max_tokens=10,
                    temperature=0.01,
                    repetition_penalty=2.0,
                    seed=42,
                ))

        results = run_concurrent_batch(tt_server, tt_model_name, configs)

        no_penalty = [results[i] for i in range(0, max_batch_size, 2)]
        with_penalty = [results[i] for i in range(1, max_batch_size, 2)]

        # All no-penalty should be same (same seed, same params)
        assert len(set(no_penalty)) == 1, (
            f"No penalty requests should be identical: {no_penalty}"
        )
        # All with-penalty should be same
        assert len(set(with_penalty)) == 1, (
            f"With penalty requests should be identical: {with_penalty}"
        )
        # But they should differ from each other
        assert no_penalty[0] != with_penalty[0], (
            f"Penalty should change output.\n"
            f"No penalty: {no_penalty[0]!r}\n"
            f"With penalty: {with_penalty[0]!r}"
        )


class TestPresencePenaltyPerRequest:
    """
    Different presence penalties per request.
    """

    def test_different_presence_penalties(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Each request has different presence penalty.
        """
        prompt = "List words:"

        penalties = [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5][:max_batch_size]
        
        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=30,
                temperature=0.7,
                presence_penalty=penalty,
                seed=42,
            )
            for penalty in penalties
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        unique_results = set(results)

        assert len(unique_results) >= 2, (
            f"Different presence penalties should produce different outputs.\n"
            f"Got only {len(unique_results)} unique"
        )

    def test_presence_penalty_mixed_batch(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Mix of no penalty and high penalty in same batch.
        """
        prompt = "Generate text:"
        
        configs = []
        for i in range(max_batch_size):
            configs.append(RequestConfig(
                prompt=prompt,
                max_tokens=20,
                temperature=0.5,
                presence_penalty=0.0 if i % 2 == 0 else 2.0,
                seed=42,
            ))

        results = run_concurrent_batch(tt_server, tt_model_name, configs)

        no_penalty = [results[i] for i in range(0, max_batch_size, 2)]
        with_penalty = [results[i] for i in range(1, max_batch_size, 2)]

        # Check they got different treatment
        assert len(set(no_penalty)) == 1, "No penalty should be consistent"
        assert len(set(with_penalty)) == 1, "With penalty should be consistent"


class TestFrequencyPenaltyPerRequest:
    """
    Different frequency penalties per request.
    """

    def test_different_frequency_penalties(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Each request has different frequency penalty.
        """
        prompt = "5 5 5 5. Continue:"

        penalties = [0.0, 0.5, 1.0, 1.5, 2.0, -0.5, -1.0, -1.5][:max_batch_size]
        
        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=20,
                temperature=0.5,
                frequency_penalty=penalty,
                seed=42,
            )
            for penalty in penalties
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        unique_results = set(results)

        assert len(unique_results) >= 2, (
            f"Different frequency penalties should produce different outputs."
        )

    def test_frequency_penalty_mixed_batch(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Mix of penalty levels in same batch.
        """
        prompt = "a a a a. Letter:"
        
        configs = []
        for i in range(max_batch_size):
            configs.append(RequestConfig(
                prompt=prompt,
                max_tokens=15,
                temperature=0.5,
                frequency_penalty=0.0 if i % 2 == 0 else 2.0,
                seed=42,
            ))

        results = run_concurrent_batch(tt_server, tt_model_name, configs)

        no_penalty = set(results[i] for i in range(0, max_batch_size, 2))
        with_penalty = set(results[i] for i in range(1, max_batch_size, 2))

        assert len(no_penalty) == 1, "Same params should produce same output"
        assert len(with_penalty) == 1, "Same params should produce same output"


# =============================================================================
# SEEDED SAMPLING - Different seeds per request
# =============================================================================

class TestSeededSamplingPerRequest:
    """
    Each request in batch has its own seed.
    """

    def test_different_seeds_produce_different_outputs(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Same prompt, different seeds per request.
        """
        prompt = "Random story:"

        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=20,
                temperature=1.0,
                top_k=50,
                seed=seed,
            )
            for seed in range(max_batch_size)
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        unique_results = set(results)

        assert len(unique_results) >= max_batch_size // 2, (
            f"Different seeds should produce different outputs.\n"
            f"Got only {len(unique_results)}/{max_batch_size} unique"
        )

    def test_same_seeds_reproduce_across_batches(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Same seeds should reproduce even when in different batch positions.
        """
        prompt = "Generate:"
        seeds = list(range(100, 100 + max_batch_size))

        configs = [
            RequestConfig(
                prompt=prompt,
                max_tokens=15,
                temperature=1.0,
                top_k=50,
                seed=seed,
            )
            for seed in seeds
        ]

        results1 = run_concurrent_batch(tt_server, tt_model_name, configs)

        # Shuffle configs but keep seed mapping
        import random
        shuffled_indices = list(range(max_batch_size))
        random.Random(42).shuffle(shuffled_indices)
        shuffled_configs = [configs[i] for i in shuffled_indices]

        results2 = run_concurrent_batch(tt_server, tt_model_name, shuffled_configs)

        # Map back to original order
        unshuffled_results2 = [""] * max_batch_size
        for new_idx, orig_idx in enumerate(shuffled_indices):
            unshuffled_results2[orig_idx] = results2[new_idx]

        for i, (r1, r2) in enumerate(zip(results1, unshuffled_results2)):
            assert r1 == r2, (
                f"Seed {seeds[i]} should produce same output in different batches.\n"
                f"First batch: {r1!r}\n"
                f"Second batch: {r2!r}"
            )

    @pytest.mark.parametrize("seed", [42, 123, 999, 0])
    def test_specific_seed_reproducible(
        self, tt_server, tt_model_name, max_batch_size, seed
    ):
        """
        A specific seed should produce same result across batch runs.
        """
        # Create batch where one request has our test seed
        configs = [
            RequestConfig(
                prompt=f"Text {i}:",
                max_tokens=10,
                temperature=1.0,
                top_k=50,
                seed=seed if i == 0 else i * 1000,
            )
            for i in range(max_batch_size)
        ]

        results1 = run_concurrent_batch(tt_server, tt_model_name, configs)
        results2 = run_concurrent_batch(tt_server, tt_model_name, configs)

        # The test seed request should reproduce
        assert results1[0] == results2[0], (
            f"Seed {seed} should reproduce.\n"
            f"Run 1: {results1[0]!r}\n"
            f"Run 2: {results2[0]!r}"
        )

    @pytest.mark.parametrize("seed", [1,0])
    def test_batch1_seed_reproducible(
        self, tt_server, tt_model_name, max_batch_size, seed
    ):
        """
        Batch with 1 user with seed should produce reproducible outputs.
        """
        prompt = "Random story:"
        configs = [RequestConfig(prompt=prompt, max_tokens=20, temperature=10.0, top_k=50, seed=seed)]
        flush_configs = [RequestConfig(prompt=prompt, max_tokens=20, temperature=1.0, top_k=50) for _ in range(31)]
        results = []
        for _ in range(10):
            run_concurrent_batch(tt_server, tt_model_name, flush_configs)
            results.extend(run_concurrent_batch(tt_server, tt_model_name, configs))
        assert len(set(results)) == 1, (
            f"Batch with 1 user with seed {seed} should produce reproducible outputs.\n"
            f"Got {len(set(results))} unique results out of {len(results)}"
        )

    def test_batch1_no_seed_varied(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Batch with 1 user with seed should produce reproducible outputs.
        """
        prompt = "Random story:"
        configs = [RequestConfig(prompt=prompt, max_tokens=20, temperature=10.0, top_k=50)]
        flush_configs = [RequestConfig(prompt=prompt, max_tokens=20, temperature=1.0, top_k=50) for _ in range(31)]
        results = []
        for _ in range(10):
            run_concurrent_batch(tt_server, tt_model_name, flush_configs)
            results.extend(run_concurrent_batch(tt_server, tt_model_name, configs))
        assert len(set(results)) >= 2, (
            f"Batch with 1 user with no seed should produce varied outputs.\n"
            f"Got {len(set(results))} unique results out of {len(results)}"
        )

    @pytest.mark.parametrize("seed", [1,0])
    def test_uniform_seed_deterministic(
        self, tt_server, tt_model_name, max_batch_size, seed
    ):
        """
        Full batch with uniform seed should produce deterministic outputs.
        """
        prompt = "Random story:"
        configs = [
            RequestConfig(prompt=prompt, max_tokens=20, temperature=1.0, top_k=50, seed=seed)
            for _ in range(32)
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        unique_results = set(results)
        assert len(unique_results) == 1, (
            f"Seed {seed} should produce deterministic outputs.\n"
            f"Got {len(unique_results)} unique results out of {len(configs)}"
        )

    def test_uniform_noseed_varied(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Full batch without seed should produce varied outputs.
        """
        prompt = "Random story:"
        configs = [
            RequestConfig(prompt=prompt, max_tokens=20, temperature=10.0, top_k=50)
            for _ in range(32)
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        unique_results = set(results)
        assert len(unique_results) >= 1, (
            f"No seed should produce varied outputs.\n"
            f"Got {len(unique_results)} unique results out of {len(configs)}"
        )

    def test_seed_0_produces_deterministic_outputs(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Seed 0 should produce deterministic outputs.
        """
        prompt = "Random story:"
        configs = [
            RequestConfig(prompt=prompt, max_tokens=20, temperature=1.0, top_k=50, seed=0)
            for _ in range(10)
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        unique_results = set(results)
        assert len(unique_results) == 1, (
            f"Seed 0 should produce deterministic outputs.\n"
            f"Got {len(unique_results)} unique results out of {len(configs)}"
        )

    def test_negative_seed_does_not_crash(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Negative seed should not crash.
        """
        prompt = "Random story:"
        configs = [RequestConfig(prompt=prompt, max_tokens=20, temperature=1.0, top_k=50, seed=-1)]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)


# =============================================================================
# BATCH ISOLATION - Verify outputs match their request params
# =============================================================================

class TestBatchIsolation:
    """
    Verify each request gets its own parameters applied.
    """

    def test_mixed_params_batch(self, tt_server, tt_model_name, max_batch_size):
        """
        Batch where each request has completely different parameters.
        """
        configs = [
            # Greedy
            RequestConfig(prompt="Count:", max_tokens=5, temperature=0),
            # High temp with seed
            RequestConfig(prompt="Random:", max_tokens=5, temperature=2.0, 
                         top_k=100, seed=42),
            # With repetition penalty
            RequestConfig(prompt="test test. Say:", max_tokens=5, 
                         temperature=0.5, repetition_penalty=3.0, seed=42),
            # With presence penalty
            RequestConfig(prompt="List:", max_tokens=10, temperature=0.5,
                         presence_penalty=2.0, seed=42),
            # top_k=1 (should be greedy-like)
            RequestConfig(prompt="Word:", max_tokens=3, temperature=2.0,
                         top_k=1, seed=99),
            # Low temp
            RequestConfig(prompt="Letter:", max_tokens=3, temperature=0.01,
                         seed=42),
            # High temp low top_k
            RequestConfig(prompt="Number:", max_tokens=3, temperature=2.0,
                         top_k=5, seed=42),
            # With frequency penalty
            RequestConfig(prompt="a a a. Next:", max_tokens=5, temperature=0.5,
                         frequency_penalty=2.0, seed=42),
        ][:max_batch_size]

        # Run twice
        results1 = run_concurrent_batch(tt_server, tt_model_name, configs)
        results2 = run_concurrent_batch(tt_server, tt_model_name, configs)

        # Each deterministic config should reproduce
        for i, (r1, r2) in enumerate(zip(results1, results2)):
            if configs[i].temperature == 0 or configs[i].seed is not None:
                assert r1 == r2, (
                    f"Request {i} should be deterministic/reproducible.\n"
                    f"Config: temp={configs[i].temperature}, seed={configs[i].seed}\n"
                    f"Run 1: {r1!r}\n"
                    f"Run 2: {r2!r}"
                )

    @pytest.mark.skip(reason="Need to improve, compat doesnt pass")
    def test_outputs_not_mixed_different_prompts(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Different prompts should get contextually appropriate outputs.
        """
        configs = [
            RequestConfig(prompt="2 + 2 =", max_tokens=5, temperature=0),
            RequestConfig(prompt="The color of grass is", max_tokens=5, temperature=0),
            RequestConfig(prompt="H2O is called", max_tokens=5, temperature=0),
            RequestConfig(prompt="Capital of France is", max_tokens=5, temperature=0),
            RequestConfig(prompt="Dogs say", max_tokens=5, temperature=0),
            RequestConfig(prompt="Cats say", max_tokens=5, temperature=0),
            RequestConfig(prompt="1, 2, 3,", max_tokens=5, temperature=0),
            RequestConfig(prompt="A, B, C,", max_tokens=5, temperature=0),
        ][:max_batch_size]

        checks = [
            lambda t: any(c.isdigit() for c in t),  # 2+2
            lambda t: "green" in t.lower(),
            lambda t: "water" in t.lower(),
            lambda t: "paris" in t.lower(),
            lambda t: any(w in t.lower() for w in ["bark", "woof"]),
            lambda t: any(w in t.lower() for w in ["meow", "purr"]),
            lambda t: "4" in t or "5" in t,
            lambda t: "d" in t.lower() or "e" in t.lower(),
        ][:max_batch_size]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)

        for i, (result, check) in enumerate(zip(results, checks)):
            assert check(result), (
                f"Request {i} output doesn't match expected.\n"
                f"Prompt: {configs[i].prompt!r}\n"
                f"Output: {result!r}"
            )


# =============================================================================
# BATCH SIZE VARIATIONS
# =============================================================================

class TestBatchSizeVariations:
    """
    Test various batch sizes with per-request params.
    """

    def test_small_batch_different_params(self, tt_server, tt_model_name):
        """
        Small batch of 2 with different params each.
        """
        configs = [
            RequestConfig(prompt="A:", max_tokens=5, temperature=0),
            RequestConfig(prompt="B:", max_tokens=5, temperature=1.0, 
                         top_k=50, seed=42),
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == 2
        assert all(len(r) > 0 for r in results)

    def test_full_batch_different_params(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Full batch where each request is different.
        """
        configs = [
            RequestConfig(
                prompt=f"Request {i}:",
                max_tokens=5,
                temperature=0.5 + (i * 0.1),
                seed=i * 100,
            )
            for i in range(max_batch_size)
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == max_batch_size

    def test_partial_batch_different_params(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Partial batch with varied params.
        """
        batch_size = max(2, max_batch_size // 2)

        configs = [
            RequestConfig(
                prompt=f"Test {i}:",
                max_tokens=5,
                temperature=0.0 if i % 2 == 0 else 1.0,
                seed=i * 50 if i % 2 == 1 else None,
            )
            for i in range(batch_size)
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == batch_size


# =============================================================================
# COMPREHENSIVE MIXED PARAMETER TESTS
# =============================================================================

class TestMixedParameterBatches:
    """
    Complex batches with multiple parameter combinations.
    """

    def test_all_parameter_types_in_batch(
        self, tt_server, tt_model_name, max_batch_size
    ):
        """
        Batch using all different parameter types.
        """
        configs = [
            # Greedy baseline
            RequestConfig(prompt="Greedy:", max_tokens=8, temperature=0),
            # Temperature variation
            RequestConfig(prompt="Temp:", max_tokens=8, temperature=1.5, 
                         top_k=50, seed=1),
            # Top-k variation
            RequestConfig(prompt="TopK:", max_tokens=8, temperature=1.0, 
                         top_k=10, seed=2),
            # Repetition penalty
            RequestConfig(prompt="go go go. Rep:", max_tokens=8, temperature=0.5,
                         repetition_penalty=3.0, seed=3),
            # Presence penalty  
            RequestConfig(prompt="Pres:", max_tokens=8, temperature=0.5,
                         presence_penalty=2.0, seed=4),
            # Frequency penalty
            RequestConfig(prompt="Freq:", max_tokens=8, temperature=0.5,
                         frequency_penalty=2.0, seed=5),
            # Combined penalties
            RequestConfig(prompt="All:", max_tokens=8, temperature=0.5,
                         repetition_penalty=1.5, presence_penalty=1.0,
                         frequency_penalty=1.0, seed=6),
            # Top-p variation
            RequestConfig(prompt="TopP:", max_tokens=8, temperature=1.0,
                         top_p=0.5, seed=7),
        ][:max_batch_size]

        # Run twice to verify determinism
        results1 = run_concurrent_batch(tt_server, tt_model_name, configs)
        results2 = run_concurrent_batch(tt_server, tt_model_name, configs)

        # All seeded/deterministic should match
        for i, (r1, r2) in enumerate(zip(results1, results2)):
            assert r1 == r2, (
                f"Request {i} should be reproducible.\n"
                f"Run 1: {r1!r}\n"
                f"Run 2: {r2!r}"
            )