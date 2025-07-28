# SPDX-License-Identifier: Apache-2.0
"""Tests for TT hardware logprobs support.

These tests verify:
1. Current behavior: logprobs should fail on TT hardware (until implemented)
2. Future behavior: logprobs should work with host-side sampling
3. Consistency between logprobs and prompt_logprobs
4. Different sampling configurations with logprobs
5. All combinations of greedy/non-greedy × async/non-async code paths

Run `pytest tests/tt/test_logprobs.py`.
"""
from typing import Optional

import pytest
import torch

from vllm import SamplingParams
from vllm.platforms import current_platform

from ..conftest import VllmRunner
from ..models.utils import check_logprobs_close
from ..utils import multi_gpu_test

# Test different configurations
LOGPROBS_VALUES = [None, 0, 1, 5]  # None=disabled, 0=sampled only, N=top-N
MAX_TOKENS_VALUES = [1, 5, 10]
TEMPERATURE_VALUES = [0.0, 0.8]  # Greedy and non-greedy

# Test matrix for TT-specific code paths
SAMPLING_CONFIGS = [
    # (name, sampling_params_dict, description)
    ("greedy", {"temperature": 0.0}, "Greedy sampling (deterministic)"),
    ("non_greedy", {"temperature": 1.0, "top_k": 10, "top_p": 0.9}, "Non-greedy sampling (stochastic)"),
]

ASYNC_CONFIGS = [
    # (name, config_dict, description)
    ("async", {"disable_async_output_proc": False}, "Async output processing"),
    ("non_async", {"disable_async_output_proc": True}, "Non-async output processing"),
]


class TestTTLogprobsCurrentBehavior:
    """Test current TT hardware behavior - should reject logprobs requests."""

    @pytest.mark.skipif(
        not current_platform.is_tt(),
        reason="This test is specific to TT hardware"
    )
    @pytest.mark.parametrize("logprobs", [1, 5])
    def test_logprobs_currently_rejected(
        self,
        vllm_runner: type[VllmRunner],
        small_tt_model: str,
        tt_test_config: dict,
        logprobs: int,
    ):
        """Test that logprobs are currently rejected on TT hardware."""
        
        prompts = ["Hello, my name is", "The capital of France is"]
        
        # This should fail with current implementation
        with pytest.raises(ValueError, match="Currently not supporting logprobs on tt"):
            with vllm_runner(small_tt_model, **tt_test_config) as vllm_model:
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=5,
                    logprobs=logprobs
                )
                vllm_model.model.generate(prompts, sampling_params)

    @pytest.mark.skipif(
        not current_platform.is_tt(),
        reason="This test is specific to TT hardware"
    )
    @pytest.mark.parametrize("prompt_logprobs", [1, 5])
    def test_prompt_logprobs_currently_rejected(
        self,
        vllm_runner: type[VllmRunner],
        small_tt_model: str,
        tt_test_config: dict,
        prompt_logprobs: int,
    ):
        """Test that prompt_logprobs are currently rejected on TT hardware."""
        
        prompts = ["Hello, my name is", "The capital of France is"]
        
        # This should fail with current implementation
        with pytest.raises(ValueError, match="Currently not supporting prompt_logprobs on tt"):
            with vllm_runner(small_tt_model, **tt_test_config) as vllm_model:
                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=5,
                    prompt_logprobs=prompt_logprobs
                )
                vllm_model.model.generate(prompts, sampling_params)


class TestTTLogprobsFutureBehavior:
    """Test future TT hardware behavior - should support logprobs with host-side sampling.
    
    These tests are currently expected to fail until the implementation is complete.
    Mark with @pytest.mark.xfail initially, then remove when implementation is ready.
    """

    @pytest.mark.skipif(
        not current_platform.is_tt(),
        reason="This test is specific to TT hardware"
    )
    @pytest.mark.xfail(reason="TT logprobs not yet implemented - remove when ready")
    @pytest.mark.parametrize("sampling_name,sampling_config,sampling_desc", SAMPLING_CONFIGS)
    @pytest.mark.parametrize("async_name,async_config,async_desc", ASYNC_CONFIGS)
    @pytest.mark.parametrize("logprobs", LOGPROBS_VALUES)
    @pytest.mark.parametrize("max_tokens", MAX_TOKENS_VALUES)
    def test_tt_logprobs_across_all_code_paths(
        self,
        vllm_runner: type[VllmRunner],
        hf_runner,
        small_tt_model: str,
        tt_test_config: dict,
        sampling_name, sampling_config, sampling_desc,
        async_name, async_config, async_desc,
        logprobs: Optional[int],
        max_tokens: int,
    ):
        """Test logprobs functionality across all TT code paths and configurations.
        
        This tests all combinations of:
        - Greedy vs. non-greedy sampling
        - Async vs. non-async output processing  
        - Different logprobs values (None, 0, 1, 5)
        - Different max_tokens values (1, 5, 10)
        """
        
        prompts = ["Hello, my name is", "The capital of France is", "In the beginning"]
        
        # Generate reference outputs with HuggingFace (only for greedy, deterministic comparison)
        hf_outputs = None
        if sampling_name == "greedy" and logprobs is not None:
            with hf_runner(small_tt_model, dtype="float") as hf_model:
                hf_outputs = hf_model.generate_greedy_logprobs_limit(
                    prompts, max_tokens, logprobs or 0
                )

        # Test with TT hardware - should work with host-side sampling
        config = {**tt_test_config, **async_config, "max_logprobs": logprobs or 0}
        config["override_tt_config"] = '{"sample_on_device_mode": null}'  # Force host-side sampling
        
        with vllm_runner(small_tt_model, **config) as vllm_model:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                logprobs=logprobs,
                **sampling_config
            )
            vllm_outputs = vllm_model.model.generate(prompts, sampling_params)

        # Verify logprobs structure
        for i, output in enumerate(vllm_outputs):
            if logprobs is None:
                assert output.outputs[0].logprobs is None, \
                    f"Output {i} should have no logprobs when logprobs=None ({sampling_desc}, {async_desc})"
            else:
                assert output.outputs[0].logprobs is not None, \
                    f"Output {i} should have logprobs ({sampling_desc}, {async_desc})"
                
                assert len(output.outputs[0].logprobs) == max_tokens, \
                    f"Should have {max_tokens} token logprobs ({sampling_desc}, {async_desc})"
                
                for j, token_logprobs in enumerate(output.outputs[0].logprobs):
                    if logprobs == 0:
                        # Only sampled token
                        assert len(token_logprobs) == 1, \
                            f"Token {j} should have 1 logprob (sampled only) ({sampling_desc}, {async_desc})"
                    else:
                        # Top-k tokens plus sampled
                        assert len(token_logprobs) <= logprobs + 1, \
                            f"Token {j} should have ≤{logprobs + 1} logprobs ({sampling_desc}, {async_desc})"

        # Compare with HuggingFace reference (for greedy sampling only)
        if hf_outputs is not None and sampling_name == "greedy":
            check_logprobs_close(
                outputs_0_lst=hf_outputs,
                outputs_1_lst=[(output.prompt + output.outputs[0].text,
                               output.outputs[0].logprobs) for output in vllm_outputs],
                name_0="hf",
                name_1=f"vllm_tt_{async_name}",
            )

        print(f"✓ Logprobs working with {sampling_desc} + {async_desc} (logprobs={logprobs}, max_tokens={max_tokens})")

    @pytest.mark.skipif(
        not current_platform.is_tt(),
        reason="This test is specific to TT hardware"
    )
    @pytest.mark.xfail(reason="TT logprobs not yet implemented - remove when ready")
    @pytest.mark.parametrize("sampling_name,sampling_config,sampling_desc", SAMPLING_CONFIGS)
    @pytest.mark.parametrize("async_name,async_config,async_desc", ASYNC_CONFIGS)
    @pytest.mark.parametrize("prompt_logprobs", [1, 5])
    @pytest.mark.parametrize("max_tokens", [1, 5])
    def test_tt_prompt_logprobs_across_all_code_paths(
        self,
        vllm_runner: type[VllmRunner],
        hf_runner,
        small_tt_model: str,
        tt_test_config: dict,
        sampling_name, sampling_config, sampling_desc,
        async_name, async_config, async_desc,
        prompt_logprobs: int,
        max_tokens: int,
    ):
        """Test prompt logprobs functionality across all TT code paths."""
        
        prompts = ["Hello world", "The quick brown fox"]
        
        # Generate reference with HuggingFace (for greedy only)
        hf_outputs = None
        if sampling_name == "greedy":
            with hf_runner(small_tt_model, dtype="float") as hf_model:
                hf_outputs = hf_model.generate_greedy_logprobs(prompts, max_tokens)

        # Test with TT hardware
        config = {**tt_test_config, **async_config, "max_logprobs": prompt_logprobs}
        config["override_tt_config"] = '{"sample_on_device_mode": null}'  # Host-side sampling
        
        with vllm_runner(small_tt_model, **config) as vllm_model:
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                prompt_logprobs=prompt_logprobs,
                **sampling_config
            )
            vllm_outputs = vllm_model.model.generate(prompts, sampling_params)

        # Verify prompt logprobs structure
        for i, output in enumerate(vllm_outputs):
            assert output.prompt_logprobs is not None, \
                f"Output {i} should have prompt_logprobs ({sampling_desc}, {async_desc})"
            assert len(output.prompt_logprobs) > 0, \
                f"prompt_logprobs should not be empty ({sampling_desc}, {async_desc})"
            
            # First token should be None (no previous context)
            assert output.prompt_logprobs[0] is None, \
                f"First prompt token should have no logprob ({sampling_desc}, {async_desc})"
            
            # Remaining tokens should have logprobs
            for j, token_logprobs in enumerate(output.prompt_logprobs[1:], 1):
                assert token_logprobs is not None, \
                    f"Prompt token {j} should have logprobs ({sampling_desc}, {async_desc})"
                assert len(token_logprobs) <= prompt_logprobs + 1, \
                    f"Prompt token {j} should have ≤{prompt_logprobs + 1} logprobs ({sampling_desc}, {async_desc})"

        print(f"✓ Prompt logprobs working with {sampling_desc} + {async_desc} (prompt_logprobs={prompt_logprobs})")

    @pytest.mark.skipif(
        not current_platform.is_tt(),
        reason="This test is specific to TT hardware"
    )
    @pytest.mark.xfail(reason="TT logprobs not yet implemented - remove when ready")
    @pytest.mark.parametrize("async_name,async_config,async_desc", ASYNC_CONFIGS)
    def test_tt_combined_logprobs_across_async_modes(
        self,
        vllm_runner: type[VllmRunner],
        small_tt_model: str,
        tt_test_config: dict,
        async_name, async_config, async_desc,
    ):
        """Test both logprobs and prompt_logprobs together across async modes."""
        
        prompts = ["Once upon a time", "In a galaxy far, far away"]
        
        config = {**tt_test_config, **async_config, "max_logprobs": 5}
        config["override_tt_config"] = '{"sample_on_device_mode": null}'  # Host-side sampling
        
        with vllm_runner(small_tt_model, **config) as vllm_model:
            sampling_params = SamplingParams(
                temperature=0.0,  # Use greedy for consistency
                max_tokens=3,
                logprobs=3,
                prompt_logprobs=3,
            )
            outputs = vllm_model.model.generate(prompts, sampling_params)

        for i, output in enumerate(outputs):
            # Check both logprobs and prompt_logprobs are present
            assert output.outputs[0].logprobs is not None, \
                f"Output {i} should have logprobs ({async_desc})"
            assert output.prompt_logprobs is not None, \
                f"Output {i} should have prompt_logprobs ({async_desc})"
            
            # Verify structure
            assert len(output.outputs[0].logprobs) == 3, \
                f"Should have 3 token logprobs ({async_desc})"
            assert len(output.prompt_logprobs) > 0, \
                f"Should have prompt logprobs ({async_desc})"

        print(f"✓ Combined logprobs working with {async_desc}")

    @pytest.mark.skipif(
        not current_platform.is_tt(),
        reason="This test is specific to TT hardware"
    )
    @pytest.mark.xfail(reason="TT logprobs not yet implemented - remove when ready")
    @pytest.mark.parametrize("async_name,async_config,async_desc", ASYNC_CONFIGS)
    def test_tt_logprobs_determinism_across_async_modes(
        self,
        vllm_runner: type[VllmRunner],
        small_tt_model: str,
        tt_test_config: dict,
        async_name, async_config, async_desc,
    ):
        """Test that greedy sampling with logprobs is deterministic across async modes."""
        
        prompts = ["The meaning of life is"]
        
        config = {**tt_test_config, **async_config, "max_logprobs": 5}
        config["override_tt_config"] = '{"sample_on_device_mode": null}'
        
        with vllm_runner(small_tt_model, **config) as vllm_model:
            sampling_params = SamplingParams(
                temperature=0.0,  # Greedy sampling should be deterministic
                max_tokens=5,
                logprobs=3,
                seed=42,  # Set seed for reproducibility
            )
            
            # Run multiple times
            outputs_1 = vllm_model.model.generate(prompts, sampling_params)
            outputs_2 = vllm_model.model.generate(prompts, sampling_params)
            
            # Results should be identical
            assert len(outputs_1) == len(outputs_2) == 1
            
            output_1 = outputs_1[0].outputs[0]
            output_2 = outputs_2[0].outputs[0]
            
            # Text should be identical
            assert output_1.text == output_2.text, \
                f"Greedy sampling should be deterministic ({async_desc})"
            
            # Logprobs should be identical
            assert len(output_1.logprobs) == len(output_2.logprobs)
            for logprobs_1, logprobs_2 in zip(output_1.logprobs, output_2.logprobs):
                assert set(logprobs_1.keys()) == set(logprobs_2.keys())
                for token_id in logprobs_1.keys():
                    assert abs(logprobs_1[token_id].logprob - logprobs_2[token_id].logprob) < 1e-6, \
                        f"Logprobs should be deterministic with greedy sampling ({async_desc})"

        print(f"✓ Greedy sampling determinism verified with {async_desc}")

    @pytest.mark.skipif(
        not current_platform.is_tt(),
        reason="This test is specific to TT hardware"
    )
    @pytest.mark.xfail(reason="TT logprobs not yet implemented - remove when ready")
    def test_tt_logprobs_batch_consistency_across_code_paths(
        self,
        vllm_runner: type[VllmRunner],
        small_tt_model: str,
        tt_test_config: dict,
    ):
        """Test that logprobs are consistent across batch sizes and code paths on TT hardware."""
        
        prompts = [
            "Hello world",
            "The quick brown fox",
            "In the beginning was the word",
            "To be or not to be",
        ]
        
        # Test with both async configurations
        for async_name, async_config, async_desc in ASYNC_CONFIGS:
            config = {**tt_test_config, **async_config, "max_logprobs": 3}
            config["override_tt_config"] = '{"sample_on_device_mode": null}'  # Host-side sampling
            
            with vllm_runner(small_tt_model, **config) as vllm_model:
                sampling_params = SamplingParams(
                    temperature=0.0,  # Greedy for deterministic results
                    max_tokens=3,
                    logprobs=3,
                    seed=42,  # For reproducibility
                )
                
                # Test single prompt
                single_outputs = []
                for prompt in prompts:
                    output = vllm_model.model.generate([prompt], sampling_params)
                    single_outputs.append(output[0])
                
                # Test batch of prompts
                batch_outputs = vllm_model.model.generate(prompts, sampling_params)
                
                # Compare results - they should be identical
                assert len(single_outputs) == len(batch_outputs)
                for single, batch in zip(single_outputs, batch_outputs):
                    assert single.outputs[0].text == batch.outputs[0].text, \
                        f"Batch consistency failed ({async_desc})"
                    
                    # Compare logprobs structure
                    single_logprobs = single.outputs[0].logprobs
                    batch_logprobs = batch.outputs[0].logprobs
                    assert len(single_logprobs) == len(batch_logprobs)
                    
                    for s_token_logprobs, b_token_logprobs in zip(single_logprobs, batch_logprobs):
                        assert set(s_token_logprobs.keys()) == set(b_token_logprobs.keys())
                        for token_id in s_token_logprobs.keys():
                            torch.testing.assert_close(
                                torch.tensor(s_token_logprobs[token_id].logprob),
                                torch.tensor(b_token_logprobs[token_id].logprob),
                                atol=1e-6,
                                rtol=1e-6,
                            )

            print(f"✓ Batch consistency verified with {async_desc}")


class TestTTLogprobsEdgeCases:
    """Test edge cases and error conditions for TT logprobs across all code paths."""

    @pytest.mark.skipif(
        not current_platform.is_tt(),
        reason="This test is specific to TT hardware"
    )
    @pytest.mark.xfail(reason="TT logprobs not yet implemented - remove when ready")
    @pytest.mark.parametrize("async_name,async_config,async_desc", ASYNC_CONFIGS)
    def test_tt_logprobs_with_stop_tokens_across_async_modes(
        self,
        vllm_runner: type[VllmRunner],
        small_tt_model: str,
        tt_test_config: dict,
        async_name, async_config, async_desc,
    ):
        """Test logprobs behavior with stop tokens across async modes on TT hardware."""
        
        prompts = ["Count to three: 1, 2,"]
        
        config = {**tt_test_config, **async_config, "max_logprobs": 3}
        config["override_tt_config"] = '{"sample_on_device_mode": null}'
        
        with vllm_runner(small_tt_model, **config) as vllm_model:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=10,
                logprobs=3,
                stop=[",", "."],  # Should stop generation
            )
            outputs = vllm_model.model.generate(prompts, sampling_params)
            
            # Verify logprobs are present even with early stopping
            assert len(outputs) == 1
            output = outputs[0].outputs[0]
            if output.logprobs:  # May be empty if stopped immediately
                for token_logprobs in output.logprobs:
                    assert len(token_logprobs) <= 4  # 3 top + 1 sampled

        print(f"✓ Stop tokens handled correctly with {async_desc}")

    @pytest.mark.skipif(
        not current_platform.is_tt(),
        reason="This test is specific to TT hardware"
    )
    @pytest.mark.xfail(reason="TT logprobs not yet implemented - remove when ready")
    @pytest.mark.parametrize("async_name,async_config,async_desc", ASYNC_CONFIGS)
    def test_tt_logprobs_memory_efficiency_across_async_modes(
        self,
        vllm_runner: type[VllmRunner],
        small_tt_model: str,
        tt_test_config: dict,
        async_name, async_config, async_desc,
    ):
        """Test that TT logprobs don't cause memory issues across async modes."""
        
        # Create longer prompts to test memory handling
        long_prompts = [
            "This is a longer prompt that should test memory handling " * 10,
            "Another long prompt to verify memory efficiency " * 10,
        ]
        
        config = {**tt_test_config, **async_config, "max_logprobs": 10}  # Request more logprobs
        config["override_tt_config"] = '{"sample_on_device_mode": null}'
        
        with vllm_runner(small_tt_model, **config) as vllm_model:
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=20,  # Longer generation
                logprobs=10,
                prompt_logprobs=5,
            )
            
            # This should complete without memory errors
            outputs = vllm_model.model.generate(long_prompts, sampling_params)
            
            assert len(outputs) == 2
            for output in outputs:
                assert output.outputs[0].logprobs is not None
                assert output.prompt_logprobs is not None

        print(f"✓ Memory efficiency verified with {async_desc}")


# Test helper functions
def verify_logprobs_structure(outputs, expected_logprobs: Optional[int]):
    """Helper function to verify logprobs structure."""
    for output in outputs:
        if expected_logprobs is None:
            assert output.outputs[0].logprobs is None
        else:
            assert output.outputs[0].logprobs is not None
            for token_logprobs in output.outputs[0].logprobs:
                if expected_logprobs == 0:
                    assert len(token_logprobs) == 1  # Only sampled token
                else:
                    assert len(token_logprobs) <= expected_logprobs + 1


def verify_prompt_logprobs_structure(outputs, expected_prompt_logprobs: Optional[int]):
    """Helper function to verify prompt logprobs structure."""
    for output in outputs:
        if expected_prompt_logprobs is None:
            assert output.prompt_logprobs is None
        else:
            assert output.prompt_logprobs is not None
            assert output.prompt_logprobs[0] is None  # First token has no logprob
            for token_logprobs in output.prompt_logprobs[1:]:
                if token_logprobs is not None:
                    assert len(token_logprobs) <= expected_prompt_logprobs + 1 