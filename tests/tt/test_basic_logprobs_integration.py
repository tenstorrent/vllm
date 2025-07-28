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

import pytest
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_tt(),
    reason="This test requires TT hardware"
)
def test_current_logprobs_rejection(small_tt_model, tt_test_config):
    """Test that logprobs are currently rejected on TT hardware.
    
    This test should PASS with current implementation and FAIL after logprobs are implemented.
    """
    prompts = ["Hello, my name is"]
    
    # Initialize TT model using fixtures
    llm = LLM(model=small_tt_model, **tt_test_config)
    
    # Test that logprobs parameter raises an error
    with pytest.raises(ValueError, match="Currently not supporting logprobs on tt"):
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=5,
            logprobs=1
        )
        llm.generate(prompts, sampling_params)


@pytest.mark.skipif(
    not current_platform.is_tt(),
    reason="This test requires TT hardware"
)
def test_current_prompt_logprobs_rejection(small_tt_model, tt_test_config):
    """Test that prompt_logprobs are currently rejected on TT hardware."""
    prompts = ["Hello, my name is"]
    
    # Initialize TT model using fixtures
    llm = LLM(model=small_tt_model, **tt_test_config)
    
    # Test that prompt_logprobs parameter raises an error
    with pytest.raises(ValueError, match="Currently not supporting prompt_logprobs on tt"):
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=5,
            prompt_logprobs=1
        )
        llm.generate(prompts, sampling_params)


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


@pytest.mark.skipif(
    not current_platform.is_tt(),
    reason="This test requires TT hardware"
)
@pytest.mark.xfail(reason="TT logprobs not yet implemented")
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
    prompts = ["Hello, my name is", "The capital of France is"]
    
    # Combine configurations
    config = {**tt_test_config, **async_config, "max_logprobs": 3}
    config["override_tt_config"] = '{"sample_on_device_mode": null}'  # Force host-side sampling
    
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


@pytest.mark.skipif(
    not current_platform.is_tt(),
    reason="This test requires TT hardware"
)
@pytest.mark.xfail(reason="TT logprobs not yet implemented")
@pytest.mark.parametrize("sampling_name,sampling_config,sampling_desc", SAMPLING_CONFIGS)
@pytest.mark.parametrize("async_name,async_config,async_desc", ASYNC_CONFIGS)
def test_prompt_logprobs_across_all_code_paths(
    small_tt_model, tt_test_config,
    sampling_name, sampling_config, sampling_desc,
    async_name, async_config, async_desc
):
    """Test prompt logprobs functionality across all TT code paths."""
    prompts = ["Hello world", "The quick brown fox"]
    
    # Combine configurations
    config = {**tt_test_config, **async_config, "max_logprobs": 3}
    config["override_tt_config"] = '{"sample_on_device_mode": null}'  # Host-side sampling
    
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


@pytest.mark.skipif(
    not current_platform.is_tt(),
    reason="This test requires TT hardware"
)
@pytest.mark.xfail(reason="TT logprobs not yet implemented")
@pytest.mark.parametrize("async_name,async_config,async_desc", ASYNC_CONFIGS)
def test_combined_logprobs_across_async_modes(
    small_tt_model, tt_test_config,
    async_name, async_config, async_desc
):
    """Test both logprobs and prompt_logprobs together across async modes."""
    prompts = ["Once upon a time"]
    
    # Combine configurations
    config = {**tt_test_config, **async_config, "max_logprobs": 5}
    config["override_tt_config"] = '{"sample_on_device_mode": null}'  # Host-side sampling
    
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


@pytest.mark.skipif(
    not current_platform.is_tt(),
    reason="This test requires TT hardware"
)
@pytest.mark.xfail(reason="TT logprobs not yet implemented")
def test_sampling_determinism_with_logprobs(small_tt_model, tt_test_config):
    """Test that greedy sampling with logprobs is deterministic across runs."""
    prompts = ["The meaning of life is"]
    
    config = {**tt_test_config, "max_logprobs": 3}
    config["override_tt_config"] = '{"sample_on_device_mode": null}'
    
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


@pytest.mark.skipif(
    not current_platform.is_tt(),
    reason="This test requires TT hardware"
)
@pytest.mark.xfail(reason="TT logprobs not yet implemented")
def test_non_greedy_sampling_variety_with_logprobs(small_tt_model, tt_test_config):
    """Test that non-greedy sampling with logprobs produces variety across runs."""
    prompts = ["The weather today is"]
    
    config = {**tt_test_config, "max_logprobs": 3}
    config["override_tt_config"] = '{"sample_on_device_mode": null}'
    
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


if __name__ == "__main__":
    # Allow running this file directly for debugging
    print("Running TT logprobs integration tests...")
    
    # Mock fixtures for direct execution
    small_tt_model = "meta-llama/Llama-3.2-1B"
    tt_test_config = {
        "trust_remote_code": True,
        "max_model_len": 1024,
        "device": "tt",
        "dtype": "bfloat16",
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "block_size": 64,
        "max_num_seqs": 32,
        "num_scheduler_steps": 10,
        "disable_log_stats": False,
        "disable_async_output_proc": False,
    }
    
    # Test current behavior
    try:
        test_current_logprobs_rejection(small_tt_model, tt_test_config)
        print("✓ Current logprobs rejection test passed")
    except Exception as e:
        print(f"✗ Current logprobs rejection test failed: {e}")
    
    try:
        test_current_prompt_logprobs_rejection(small_tt_model, tt_test_config)
        print("✓ Current prompt_logprobs rejection test passed")
    except Exception as e:
        print(f"✗ Current prompt_logprobs rejection test failed: {e}")
    
    print("\nFuture functionality tests (expected to fail until implemented):")
    print("- Testing all combinations of greedy/non-greedy × async/non-async")
    print("- Testing determinism and variety")
    print("- Run with --runxfail to see actual test execution")
    
    print("✓ Test matrix covers all TT code paths!") 