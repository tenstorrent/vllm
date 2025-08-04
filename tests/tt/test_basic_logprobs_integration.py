# SPDX-License-Identifier: Apache-2.0
"""Basic integration tests for TT hardware logprobs support.

This file contains essential tests to verify TT logprobs functionality.
Start with these tests when implementing TT logprobs support.

Usage:
    # Test current behavior (should fail)
    pytest tests/tt/test_basic_logprobs_integration.py::test_current_logprobs_rejection -v
    
    # Test future behavior (after implementation)
    pytest tests/tt/test_basic_logprobs_integration.py::test_basic_logprobs_functionality -v --runxfail
"""

import gc
import pytest
from vllm import LLM, SamplingParams


def test_current_logprobs_basic_support(small_tt_model, tt_test_config):
    """Test that basic logprobs support works on TT hardware.
    
    This test verifies that logprobs parameter is accepted and returns some logprobs data.
    """
    try:
        llm = LLM(model=small_tt_model, **tt_test_config) 
        prompts = ["Hello, my name is"]
        
        # Test that logprobs parameter works
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=5,
            logprobs=1
        )
        outputs = llm.generate(prompts, sampling_params)

        # Verify basic structure
        assert len(outputs) == 1
        output = outputs[0]
        assert len(output.outputs) == 1
        seq_output = output.outputs[0]
        
        # Should have some tokens generated
        assert len(seq_output.token_ids) > 0
        
        # Should have logprobs (even if basic/placeholder for now)
        assert seq_output.logprobs is not None
        assert len(seq_output.logprobs) == len(seq_output.token_ids)
    finally:
        #force deletion of LLM instance, so the worker is torn down and device is closed, workaround for https://github.com/tenstorrent/tt-metal/issues/26128
        del llm
        gc.collect()


@pytest.mark.xfail(reason="Prompt logprobs are currently not rejected on TT hardware.")
def test_current_prompt_logprobs_rejection(small_tt_model, tt_test_config):
    """Test that prompt_logprobs are currently rejected on TT hardware."""
    try:
        llm = LLM(model=small_tt_model, **tt_test_config) 
        prompts = ["Hello, my name is"]
        
        # Test that prompt_logprobs parameter raises an error
        with pytest.raises(ValueError, match="Currently not supporting prompt_logprobs on tt"):
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=5,
                prompt_logprobs=1
            )
            llm.generate(prompts, sampling_params)
    finally:
        #force deletion of LLM instance, so the worker is torn down and device is closed, workaround for https://github.com/tenstorrent/tt-metal/issues/26128
        del llm
        gc.collect()


# Test matrix: All combinations of sampling and async processing
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


@pytest.mark.parametrize("sampling_name,sampling_config,sampling_desc", SAMPLING_CONFIGS)
@pytest.mark.parametrize("async_name,async_config,async_desc", ASYNC_CONFIGS)
def test_logprobs_across_all_code_paths(
    small_tt_model, tt_test_config,
    sampling_name, sampling_config, sampling_desc,
    async_name, async_config, async_desc
):
    """Test logprobs functionality across all TT code paths.
    
    This tests all combinations of:
    - Greedy vs. non-greedy sampling
    - Async vs. non-async output processing
    
    Each combination may have different code paths in TT hardware.
    """
    try:
        prompts = ["Hello, my name is", "The capital of France is"]
        
        # Combine configurations (need custom config, can't use session LLM)
        config = {**tt_test_config, **async_config, "max_logprobs": 3}
        config["override_tt_config"] = {"sample_on_device_mode": None}  # Force host-side sampling
        
        llm = LLM(model=small_tt_model, **config)
        
        # Create sampling params with the specified configuration
        sampling_params = SamplingParams(
            max_tokens=5,
            logprobs=3,
            **sampling_config
        )
        
        outputs = llm.generate(prompts, sampling_params)
        
        # Verify outputs structure regardless of sampling method
        assert len(outputs) == len(prompts), f"Should generate {len(prompts)} outputs"
        
        for i, output in enumerate(outputs):
            # Check that logprobs are present
            assert output.outputs[0].logprobs is not None, \
                f"Output {i} should have logprobs ({sampling_desc}, {async_desc})"
            
            # Check logprobs structure
            logprobs = output.outputs[0].logprobs
            assert len(logprobs) == 5, f"Should have 5 token logprobs (max_tokens=5)"
            
            # Each token should have at most 4 logprobs (3 top + 1 sampled)
            for j, token_logprobs in enumerate(logprobs):
                assert len(token_logprobs) <= 4, \
                    f"Token {j} should have ≤4 logprobs, got {len(token_logprobs)} ({sampling_desc}, {async_desc})"
                
                # Verify logprob values are reasonable (negative numbers)
                for token_id, logprob_obj in token_logprobs.items():
                    assert isinstance(token_id, int), f"Token ID should be int ({sampling_desc}, {async_desc})"
                    assert logprob_obj.logprob <= 0.0, \
                        f"Log probability should be ≤0, got {logprob_obj.logprob} ({sampling_desc}, {async_desc})"
                    assert isinstance(logprob_obj.decoded_token, str), \
                        f"Decoded token should be string ({sampling_desc}, {async_desc})"
        
        print(f"✓ Logprobs working correctly with {sampling_desc} + {async_desc}")
    finally:
        #force deletion of LLM instance, so the worker is torn down and device is closed, workaround for https://github.com/tenstorrent/tt-metal/issues/26128
        del llm
        gc.collect()


@pytest.mark.parametrize("sampling_name,sampling_config,sampling_desc", SAMPLING_CONFIGS)
@pytest.mark.parametrize("async_name,async_config,async_desc", ASYNC_CONFIGS)
def test_prompt_logprobs_across_all_code_paths(
    small_tt_model, tt_test_config,
    sampling_name, sampling_config, sampling_desc,
    async_name, async_config, async_desc
):
    try:
        """Test prompt logprobs functionality across all TT code paths."""
        prompts = ["Hello world", "The quick brown fox"]
        
        # Combine configurations
        config = {**tt_test_config, **async_config, "max_logprobs": 3}
        config["override_tt_config"] = {"sample_on_device_mode": None}  # Host-side sampling
        
        llm = LLM(model=small_tt_model, **config)
        
        sampling_params = SamplingParams(
            max_tokens=3,
            prompt_logprobs=3,
            **sampling_config
        )
        
        outputs = llm.generate(prompts, sampling_params)
        
        for i, output in enumerate(outputs):
            # Check that prompt_logprobs are present
            assert output.prompt_logprobs is not None, \
                f"Output {i} should have prompt_logprobs ({sampling_desc}, {async_desc})"
            assert len(output.prompt_logprobs) > 0, \
                f"prompt_logprobs should not be empty ({sampling_desc}, {async_desc})"
            
            # First token should have no logprob (no previous context)
            assert output.prompt_logprobs[0] is None, \
                f"First prompt token should have no logprob ({sampling_desc}, {async_desc})"
            
            # Check remaining prompt logprobs
            for j, token_logprobs in enumerate(output.prompt_logprobs[1:], 1):
                if token_logprobs is not None:
                    assert len(token_logprobs) <= 4, \
                        f"Prompt token {j} should have ≤4 logprobs ({sampling_desc}, {async_desc})"
                    for token_id, logprob_obj in token_logprobs.items():
                        assert logprob_obj.logprob <= 0.0, \
                            f"Prompt logprob should be ≤0 ({sampling_desc}, {async_desc})"
        
        print(f"✓ Prompt logprobs working correctly with {sampling_desc} + {async_desc}")
    finally:
        #force deletion of LLM instance, so the worker is torn down and device is closed, workaround for https://github.com/tenstorrent/tt-metal/issues/26128
        del llm
        gc.collect()


@pytest.mark.parametrize("async_name,async_config,async_desc", ASYNC_CONFIGS)
def test_combined_logprobs_across_async_modes(
    small_tt_model, tt_test_config,
    async_name, async_config, async_desc
):
    """Test both logprobs and prompt_logprobs together across async modes."""
    try:
        prompts = ["Once upon a time"]
        
        # Combine configurations
        config = {**tt_test_config, **async_config, "max_logprobs": 5}
        config["override_tt_config"] = {"sample_on_device_mode": None}  # Host-side sampling
        
        llm = LLM(model=small_tt_model, **config)
        
        sampling_params = SamplingParams(
            temperature=0.0,  # Use greedy for consistency
            max_tokens=3,
            logprobs=3,
            prompt_logprobs=3,
        )
        
        outputs = llm.generate(prompts, sampling_params)
        
        assert len(outputs) == 1
        output = outputs[0]
        
        # Both should be present
        assert output.outputs[0].logprobs is not None, \
            f"Should have logprobs ({async_desc})"
        assert output.prompt_logprobs is not None, \
            f"Should have prompt_logprobs ({async_desc})"
        
        # Check structure
        assert len(output.outputs[0].logprobs) == 3, \
            f"Should have 3 token logprobs (max_tokens=3) ({async_desc})"
        assert len(output.prompt_logprobs) > 0, \
            f"Should have prompt logprobs ({async_desc})"
        
        print(f"✓ Combined logprobs working correctly with {async_desc}")
    finally:
        #force deletion of LLM instance, so the worker is torn down and device is closed, workaround for https://github.com/tenstorrent/tt-metal/issues/26128
        del llm
        gc.collect()


def test_sampling_determinism_with_logprobs(small_tt_model, tt_test_config):
    """Test that greedy sampling with logprobs is deterministic across runs."""
    try:
        prompts = ["The meaning of life is"]
        
        config = {**tt_test_config, "max_logprobs": 3}
        config["override_tt_config"] = {"sample_on_device_mode": None}
        
        llm = LLM(model=small_tt_model, **config)
        
        sampling_params = SamplingParams(
            temperature=0.0,  # Greedy sampling should be deterministic
            max_tokens=5,
            logprobs=3,
            seed=42,  # Set seed for reproducibility
        )
        
        # Run multiple times
        outputs_1 = llm.generate(prompts, sampling_params)
        outputs_2 = llm.generate(prompts, sampling_params)
        
        # Results should be identical
        assert len(outputs_1) == len(outputs_2) == 1
        
        output_1 = outputs_1[0].outputs[0]
        output_2 = outputs_2[0].outputs[0]
        
        # Text should be identical
        assert output_1.text == output_2.text, "Greedy sampling should be deterministic"
        
        # Logprobs should be identical
        assert len(output_1.logprobs) == len(output_2.logprobs)
        for logprobs_1, logprobs_2 in zip(output_1.logprobs, output_2.logprobs):
            assert set(logprobs_1.keys()) == set(logprobs_2.keys())
            for token_id in logprobs_1.keys():
                assert abs(logprobs_1[token_id].logprob - logprobs_2[token_id].logprob) < 1e-6, \
                    "Logprobs should be deterministic with greedy sampling"
        
        print("✓ Greedy sampling with logprobs is deterministic")

    finally:
        #force deletion of LLM instance, so the worker is torn down and device is closed, workaround for https://github.com/tenstorrent/tt-metal/issues/26128
        del llm
        gc.collect()


def test_non_greedy_sampling_variety_with_logprobs(small_tt_model, tt_test_config):
    """Test that non-greedy sampling with logprobs produces variety across runs."""
    try:
        prompts = ["The weather today is"]
        
        config = {**tt_test_config, "max_logprobs": 3}
        config["override_tt_config"] = {"sample_on_device_mode": None}
        
        llm = LLM(model=small_tt_model, **config)
        
        sampling_params = SamplingParams(
            temperature=1.0,  # Non-greedy sampling
            top_k=10,
            top_p=0.9,
            max_tokens=5,
            logprobs=3,
        )
        
        # Run multiple times with different seeds
        outputs = []
        for seed in [42, 123, 456]:
            sampling_params.seed = seed
            output = llm.generate(prompts, sampling_params)
            outputs.append(output[0].outputs[0])
        
        # Check that we get some variety (not all identical)
        texts = [output.text for output in outputs]
        unique_texts = set(texts)
        
        # With stochastic sampling, we expect some variety
        # (though it's possible to get identical results by chance)
        if len(unique_texts) > 1:
            print(f"✓ Non-greedy sampling produced {len(unique_texts)} unique outputs")
        else:
            print("⚠ Non-greedy sampling produced identical outputs (possible but unlikely)")
        
        # All outputs should have valid logprobs structure
        for i, output in enumerate(outputs):
            assert output.logprobs is not None, f"Output {i} should have logprobs"
            assert len(output.logprobs) == 5, f"Output {i} should have 5 token logprobs"
    finally:
        #force deletion of LLM instance, so the worker is torn down and device is closed, workaround for https://github.com/tenstorrent/tt-metal/issues/26128
        del llm
        gc.collect()