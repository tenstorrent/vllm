# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for v1 sampling parameters.

Note: Custom user-provided logits_processors are NOT supported in V1.
V1 only supports built-in logits processors (min_p, logit_bias, min_tokens)
which are configured via their respective sampling parameters.
This has been added to V1 since the version we have checked out.
"""

import pytest

from tests.tt.utils import RequestConfig, run_concurrent_batch


class TestV1Sampling:
    """
    Verify v1 sampling parameters work correctly via the OpenAI API.
    """

    @pytest.mark.parametrize("batch_fraction", [0, 0.5, 1, 1.5])
    def test_logprobs(self, tt_server, tt_model_name, max_batch_size,
                      batch_fraction):
        """Test logprobs parameter returns actual logprobs data.
        
        Parametrized by batch size to verify logprobs work correctly across
        all DP ranks. In DP mode, logprobs are computed on rank 0 and must
        be properly distributed to all ranks. Testing with full batch size
        maximizes coverage across DP ranks.
        
        batch_fraction: 1 = full batch, 0.5 = half batch, 0 = single request
        """
        num_logprobs = 5
        if batch_fraction == 0:
            num_requests = 1
        else:
            num_requests = max(1, int(max_batch_size * batch_fraction))
        
        configs = [
            RequestConfig(prompt=f"Count from {i}: ", max_tokens=10,
                          logprobs=num_logprobs)
            for i in range(num_requests)
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs,
                                       return_full_response=True)
        assert len(results) == len(configs)

        # Verify ALL responses have valid logprobs (catches DP rank issues)
        for i, response in enumerate(results):
            choice = response.choices[0]

            # Verify logprobs are returned for every request
            assert choice.logprobs is not None, \
                f"logprobs should be returned for request {i}"
            assert choice.logprobs.tokens is not None, \
                f"logprobs.tokens should exist for request {i}"
            assert len(choice.logprobs.tokens) > 0, \
                f"should have at least one token for request {i}"

            # Verify top_logprobs contains the requested number of alternatives
            assert choice.logprobs.top_logprobs is not None, \
                f"top_logprobs should be returned for request {i}"
            assert len(choice.logprobs.top_logprobs) > 0, \
                f"should have top_logprobs for tokens for request {i}"
            # Each position should have up to num_logprobs+1 alternatives
            for j, top_lp in enumerate(choice.logprobs.top_logprobs):
                assert len(top_lp) <= num_logprobs + 1, \
                    f"request {i}, token {j}: should have at most " \
                    f"{num_logprobs+1} alternatives per token"

            # Verify actual logprob values exist (not just structure)
            assert choice.logprobs.token_logprobs is not None, \
                f"request {i}: token_logprobs is None"
            for j, lp in enumerate(choice.logprobs.token_logprobs):
                assert lp is not None, \
                    f"request {i}, token {j}: logprob value is None"

    def test_min_p(self, tt_server, tt_model_name, max_batch_size):
        """Test min_p parameter (smoke test - verifies it doesn't error)."""
        configs = [
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.1),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.2),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.3),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.4),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.5),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.6),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.7),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.8),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=0.9),
            RequestConfig(prompt="Random: ", max_tokens=10, min_p=1.0),
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == len(configs)
        # min_p affects sampling distribution - just verify we get output
        assert results[0] is not None and len(results[0]) > 0, \
            "should produce non-empty output"

    def test_bad_words(self, tt_server, tt_model_name, max_batch_size):
        """Test bad_words parameter prevents specified words from appearing."""
        bad_words = ["hello", "Hello", "hi", "Hi", "hey", "Hey"]
        configs = [
            # Run multiple times with high temperature to increase coverage
            RequestConfig(prompt="Say hello to me", max_tokens=20,
                          bad_words=bad_words, temperature=1.0, seed=i)
            for i in range(5)
        ]
        # bad_words is only available in chat completions API
        results = run_concurrent_batch(tt_server, tt_model_name, configs,
                                       use_chat=True)
        assert len(results) == len(configs)

        for i, text in enumerate(results):
            for bad_word in bad_words:
                assert bad_word not in text, \
                    f"bad_word '{bad_word}' found in response {i}: {text!r}"

    def test_logit_bias(self, tt_server, tt_model_name, max_batch_size):
        """Test logit_bias parameter (smoke test - verifies it doesn't error)."""
        configs = [
            RequestConfig(prompt="Logit bias: ", max_tokens=10,
                          logit_bias={1: 0.1}),
            RequestConfig(prompt="Logit bias: ", max_tokens=10,
                          logit_bias={4: 0.2}),
            RequestConfig(prompt="Logit bias: ", max_tokens=10,
                          logit_bias={2: 0.3}),
            RequestConfig(prompt="Logit bias: ", max_tokens=10,
                          logit_bias={16: 0.4}),
            RequestConfig(prompt="Logit bias: ", max_tokens=10,
                          logit_bias={7: 0.5}),
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == len(configs)
        assert results[0] is not None, "should produce output with logit_bias"

    def test_allowed_token_ids(self, tt_server, tt_model_name, max_batch_size):
        """Test allowed_token_ids parameter (smoke test)."""
        configs = [
            RequestConfig(prompt="Allowed: ", max_tokens=10,
                          allowed_token_ids=[1, 2, 3]),
            RequestConfig(prompt="Allowed: ", max_tokens=10,
                          allowed_token_ids=[4, 5, 6]),
            RequestConfig(prompt="Allowed: ", max_tokens=10,
                          allowed_token_ids=[7, 8, 9]),
            RequestConfig(prompt="Allowed: ", max_tokens=10,
                          allowed_token_ids=[10, 11, 12]),
            RequestConfig(prompt="Allowed: ", max_tokens=10,
                          allowed_token_ids=[13, 14, 15]),
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs)
        assert len(results) == len(configs)
        # With only 3 allowed tokens, output should be limited
        for i, result in enumerate(results):
            assert result is not None, \
                f"should produce output for request {i}"
            assert len(result) > 0, \
                f"should produce non-empty output for request {i}"

    def test_min_tokens(self, tt_server, tt_model_name, max_batch_size):
        """Test min_tokens parameter ensures minimum output length."""
        min_tokens = 5
        configs = [
            # Use a prompt that might naturally produce short output
            RequestConfig(prompt="Say OK.", max_tokens=20, min_tokens=min_tokens),
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs,
                                       return_full_response=True)
        assert len(results) == len(configs)

        response = results[0]
        # Check usage stats for token count
        assert response.usage is not None, "usage stats should be returned"
        assert response.usage.completion_tokens >= min_tokens, \
            f"should produce at least {min_tokens} tokens, " \
            f"got {response.usage.completion_tokens}"
