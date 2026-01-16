# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from tests.tt.utils import RequestConfig, run_concurrent_batch


class TestBatchIsolation:
    """
    Verify each request gets its own parameters applied.
    """

    def test_mixed_params_batch(self, tt_server, tt_model_name,
                                max_batch_size):
        """
        Batch where each request has completely different parameters.
        """
        configs = [
            # Greedy
            RequestConfig(prompt="Count: ", max_tokens=5, temperature=0),
            # High temp with seed
            RequestConfig(prompt="Random: ",
                          max_tokens=5,
                          temperature=2.0,
                          top_k=100,
                          seed=42),
            # With repetition penalty
            RequestConfig(prompt="test test. Say: ",
                          max_tokens=5,
                          temperature=0.5,
                          repetition_penalty=3.0,
                          seed=42),
            # With presence penalty
            RequestConfig(prompt="List: ",
                          max_tokens=10,
                          temperature=0.5,
                          presence_penalty=2.0,
                          seed=42),
            # top_k=1 (should be greedy-like)
            RequestConfig(prompt="Word: ",
                          max_tokens=3,
                          temperature=2.0,
                          top_k=1,
                          seed=99),
            # Low temp
            RequestConfig(prompt="Letter: ",
                          max_tokens=3,
                          temperature=0.01,
                          seed=42),
            # High temp low top_k
            RequestConfig(prompt="Number: ",
                          max_tokens=3,
                          temperature=2.0,
                          top_k=5,
                          seed=42),
            # With frequency penalty
            RequestConfig(prompt="a a a. Next: ",
                          max_tokens=5,
                          temperature=0.5,
                          frequency_penalty=2.0,
                          seed=42),
        ][:max_batch_size]

        # Run twice
        results1 = run_concurrent_batch(tt_server, tt_model_name, configs)
        results2 = run_concurrent_batch(tt_server, tt_model_name, configs)

        # Each deterministic config should reproduce
        for i, (r1, r2) in enumerate(zip(results1, results2)):
            if configs[i].temperature == 0 or configs[i].seed is not None:
                assert r1 == r2, (
                    f"Request {i} should be deterministic/reproducible.\n"
                    f"Config:"
                    f"temp={configs[i].temperature},"
                    f"seed={configs[i].seed}\n"
                    f"Run 1: {r1!r}\n"
                    f"Run 2: {r2!r}")


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
            RequestConfig(prompt="A: ", max_tokens=5, temperature=0),
            RequestConfig(prompt="B: ",
                          max_tokens=5,
                          temperature=1.0,
                          top_k=50,
                          seed=42),
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == 2
        assert all(len(r) > 0 for r in results)

    def test_full_batch_different_params(self, tt_server, tt_model_name,
                                         max_batch_size):
        """
        Full batch where each request is different.
        """
        configs = [
            RequestConfig(
                prompt=f"Request {i}: ",
                max_tokens=5,
                temperature=0.5 + (i * 0.1),
                seed=i * 100,
            ) for i in range(max_batch_size)
        ]

        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == max_batch_size

    def test_partial_batch_different_params(self, tt_server, tt_model_name,
                                            max_batch_size):
        """
        Partial batch with varied params.
        """
        batch_size = max(2, max_batch_size // 2)

        configs = [
            RequestConfig(
                prompt=f"Test {i}: ",
                max_tokens=5,
                temperature=0.0 if i % 2 == 0 else 1.0,
                seed=i * 50 if i % 2 == 1 else None,
            ) for i in range(batch_size)
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

    def test_all_parameter_types_in_batch(self, tt_server, tt_model_name,
                                          max_batch_size):
        """
        Batch using all different parameter types.
        """
        configs = [
            # Greedy baseline
            RequestConfig(prompt="Greedy: ", max_tokens=8, temperature=0),
            # Temperature variation
            RequestConfig(prompt="Temp: ",
                          max_tokens=8,
                          temperature=1.5,
                          top_k=50,
                          seed=1),
            # Top-k variation
            RequestConfig(prompt="TopK: ",
                          max_tokens=8,
                          temperature=1.0,
                          top_k=10,
                          seed=2),
            # Repetition penalty
            RequestConfig(prompt="go go go. Rep: ",
                          max_tokens=8,
                          temperature=0.5,
                          repetition_penalty=3.0,
                          seed=3),
            # Presence penalty
            RequestConfig(prompt="Pres: ",
                          max_tokens=8,
                          temperature=0.5,
                          presence_penalty=2.0,
                          seed=4),
            # Frequency penalty
            RequestConfig(prompt="Freq: ",
                          max_tokens=8,
                          temperature=0.5,
                          frequency_penalty=2.0,
                          seed=5),
            # Combined penalties
            RequestConfig(prompt="All: ",
                          max_tokens=8,
                          temperature=0.5,
                          repetition_penalty=1.5,
                          presence_penalty=1.0,
                          frequency_penalty=1.0,
                          seed=6),
            # Top-p variation
            RequestConfig(prompt="TopP: ",
                          max_tokens=8,
                          temperature=1.0,
                          top_p=0.5,
                          seed=7),
        ][:max_batch_size]

        # Run twice to verify determinism
        results1 = run_concurrent_batch(tt_server, tt_model_name, configs)
        results2 = run_concurrent_batch(tt_server, tt_model_name, configs)

        # All seeded/deterministic should match
        for i, (r1, r2) in enumerate(zip(results1, results2)):
            assert r1 == r2, (f"Request {i} should be reproducible.\n"
                              f"Run 1: {r1!r}\n"
                              f"Run 2: {r2!r}")
