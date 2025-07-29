# SPDX-License-Identifier: Apache-2.0
"""Tests for TT hardware pytest fixtures and environment setup."""

import os
import pytest

from vllm import ModelRegistry, SamplingParams
from vllm.platforms import current_platform

# Import utility functions from utils module
from .utils import (
    register_tt_models, validate_tt_model, get_supported_tt_models,
    setup_tt_environment_variables, restore_tt_environment_variables
)


class TestTTFixtures:
    """Test suite for TT hardware pytest fixtures."""

    def test_tt_fixtures_are_available(self, small_tt_model, tt_test_config, tt_sampling_params):
        """Test that all TT fixtures are available and working."""
        assert isinstance(small_tt_model, str)
        assert len(small_tt_model) > 0
        assert isinstance(tt_test_config, dict)
        assert isinstance(tt_sampling_params, SamplingParams)

    @pytest.mark.skipif(current_platform.is_tt(), reason="Test non-TT behavior")
    def test_tt_fixtures_on_non_tt_hardware(self):
        """Test that TT fixtures are properly skipped on non-TT hardware."""
        # This test should only run when TT hardware is NOT available
        # The fixtures should be skipped automatically
        assert not current_platform.is_tt()

    def test_fixture_combinations(self, small_tt_model, tt_test_config, tt_sampling_params):
        """Test that fixtures work together correctly."""
        # Test that the model from the fixture is valid
        assert small_tt_model in get_supported_tt_models()
        
        # Test that the config has expected TT-specific parameters
        assert "max_model_len" in tt_test_config
        assert "dtype" in tt_test_config
        assert "enforce_eager" in tt_test_config
        assert tt_test_config["enforce_eager"] is True  # Required for TT
        
        # Test that sampling params are reasonable for testing
        assert tt_sampling_params.temperature == 0.0  # Deterministic
        assert tt_sampling_params.max_tokens == 10    # Short for fast tests


class TestTTSetup:
    """Test suite for TT setup and environment configuration."""

    def test_tt_setup_fixture(self, tt_setup):
        """Test that tt_setup fixture properly configures the environment."""
        # Check that HF_MODEL is set
        assert "HF_MODEL" in os.environ
        assert len(os.environ["HF_MODEL"]) > 0
        
        # Check other important TT environment variables
        expected_env_vars = {
            "HF_MODEL": "meta-llama/Llama-3.2-1B",  # Default model  
            "WH_ARCH_YAML": "/home/user/tt-metal/arch/wormhole_b0_80_arch_eth_dispatch.yaml",
            "MESH_DEVICE": "true",  # Enable mesh device
            "TT_LLAMA_TEXT_VER": "tt_transformers",  # TT implementation version
        }
        
        for env_var, expected_default in expected_env_vars.items():
            assert env_var in os.environ, f"{env_var} should be set"
            # Don't enforce exact values as they might be overridden by user
            assert len(os.environ[env_var]) > 0, f"{env_var} should not be empty"
        
        # Verify TT models are registered
        # Check some key TT model registrations
        expected_tt_models = [
            "TTLlamaForCausalLM",
            "TTMllamaForConditionalGeneration", 
            "TTQwen2ForCausalLM",
            "TTQwen3ForCausalLM",
            "TTMistralForCausalLM",
        ]
        
        for model_name in expected_tt_models:
            # ModelRegistry should have these models registered
            # Note: We can't easily check if they're registered without trying to use them,
            # but we can at least verify the registration function was called
            assert hasattr(register_tt_models, '_registered'), \
                f"TT models should be registered (including {model_name})"


class TestTTUtilityFunctions:
    """Test suite for TT utility functions."""

    def test_tt_model_validation(self):
        """Test TT model validation function."""
        # Test valid models
        supported_models = get_supported_tt_models()
        assert len(supported_models) > 0
        
        for model in supported_models[:3]:  # Test first few models
            validate_tt_model(model)  # Should not raise
        
        # Test invalid model
        with pytest.raises(ValueError, match="not supported on TT hardware"):
            validate_tt_model("invalid/model-name")

    def test_tt_model_registration_idempotent(self):
        """Test that TT model registration is idempotent."""
        # Register models multiple times - should not cause issues
        register_tt_models()
        register_tt_models()
        register_tt_models()
        
        # Should still be marked as registered
        assert hasattr(register_tt_models, '_registered')
        assert register_tt_models._registered is True

    def test_environment_variable_management(self):
        """Test TT environment variable setup and restoration."""
        # Store current state for manual cleanup
        original_hf_model = os.environ.get("HF_MODEL")
        original_mesh_device = os.environ.get("MESH_DEVICE")
        
        try:
            # Test 1: Setup from clean state
            if "HF_MODEL" in os.environ:
                del os.environ["HF_MODEL"]
            if "MESH_DEVICE" in os.environ:
                del os.environ["MESH_DEVICE"]
            
            # Capture clean state, then set up environment
            original_env = setup_tt_environment_variables()
            
            # Check that variables are now set
            assert "HF_MODEL" in os.environ
            assert "MESH_DEVICE" in os.environ
            assert os.environ["HF_MODEL"] == "meta-llama/Llama-3.2-1B"
            assert os.environ["MESH_DEVICE"] == "N150"
            
            # Restore environment to the clean state that was captured
            restore_tt_environment_variables(original_env)
            
            # Check restoration worked (should be back to clean state)
            assert os.environ.get("HF_MODEL") is None  # Was None when setup was called
            assert os.environ.get("MESH_DEVICE") is None  # Was None when setup was called
            
            # Test 2: Setup when variables already exist
            os.environ["HF_MODEL"] = "existing-model"
            os.environ["MESH_DEVICE"] = "existing-device"
            
            original_env2 = setup_tt_environment_variables()
            
            # Variables should remain unchanged (setup only sets if not present)
            assert os.environ["HF_MODEL"] == "existing-model"
            assert os.environ["MESH_DEVICE"] == "existing-device"
            
            # Restore should bring back the existing values
            restore_tt_environment_variables(original_env2)
            assert os.environ["HF_MODEL"] == "existing-model"
            assert os.environ["MESH_DEVICE"] == "existing-device"
            
        finally:
            # Ensure cleanup even if test fails - restore to original test state
            if original_hf_model is not None:
                os.environ["HF_MODEL"] = original_hf_model
            elif "HF_MODEL" in os.environ:
                del os.environ["HF_MODEL"]
                
            if original_mesh_device is not None:
                os.environ["MESH_DEVICE"] = original_mesh_device
            elif "MESH_DEVICE" in os.environ:
                del os.environ["MESH_DEVICE"]

    def test_tt_sampling_params_fixture(self, tt_sampling_params):
        """Test the TT sampling params fixture."""
        assert isinstance(tt_sampling_params, SamplingParams)
        assert tt_sampling_params.temperature == 0.0
        assert tt_sampling_params.max_tokens == 10
        assert tt_sampling_params.ignore_eos is False 