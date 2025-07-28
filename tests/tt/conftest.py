"""
Pytest configuration for TT hardware tests.

This module contains pytest fixtures specific to TT hardware testing.
Utility functions are in utils.py.
"""

import os
import pytest

from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

from .utils import (
    register_tt_models, 
    validate_tt_model, 
    setup_tt_environment_variables,
    restore_tt_environment_variables
)


def pytest_configure(config):
    """Configure pytest for TT tests."""
    config.addinivalue_line(
        "markers", "tt: mark test as requiring TT hardware"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark TT tests and skip if hardware not available."""
    for item in items:
        if "tt" in item.nodeid or item.fspath.dirname.endswith("tt"):
            item.add_marker(pytest.mark.tt)
            if not current_platform.is_tt():
                item.add_marker(pytest.mark.skip(reason="TT hardware not available"))


@pytest.fixture(scope="session") 
def tt_available():
    """Check if TT hardware is available."""
    if not current_platform.is_tt():
        pytest.skip("TT hardware not available")
    return True


@pytest.fixture(scope="session")
def tt_setup():
    """Session-wide fixture to set up TT environment.
    
    This fixture:
    1. Registers TT models with vLLM ModelRegistry
    2. Sets up required environment variables (HF_MODEL, WH_ARCH_YAML, etc.)
    3. Validates that test models are supported
    4. Cleans up after all tests complete
    
    This should be used by other TT fixtures to ensure proper setup.
    """
    if not current_platform.is_tt():
        pytest.skip("TT hardware not available")
    
    # Initialize original_env to None so cleanup works even if setup fails
    original_env = None
    
    try:
        # Register TT models
        register_tt_models()
        
        # Set up environment variables
        original_env = setup_tt_environment_variables()
        
        # Validate that the model we're using is supported
        current_model = os.environ["HF_MODEL"]
        validate_tt_model(current_model)
        
        yield  # Run all tests
        
    finally:
        # Clean up environment variables only if they were set up
        if original_env is not None:
            restore_tt_environment_variables(original_env)


@pytest.fixture(scope="session")
def small_tt_model(tt_setup):
    """Fixture providing a small TT model name for testing.
    
    Uses tt_setup to ensure TT environment is properly configured.
    Returns the HF_MODEL environment variable which should be set to a supported model.
    """
    return os.environ["HF_MODEL"]


@pytest.fixture(scope="session") 
def tt_test_config(tt_setup):
    """Fixture providing TT-specific LLM configuration for testing.
    
    Based on configurations from TT examples with conservative settings for testing.
    Uses tt_setup to ensure TT environment is properly configured.
    """
    return {
        "max_model_len": 1024,
        "dtype": "bfloat16",
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "block_size": 64,
        "max_num_seqs": 32, 
        "num_scheduler_steps": 10,
        "disable_log_stats": False,
        "disable_async_output_proc": False,
    }


@pytest.fixture(scope="session")
def tt_sampling_params():
    """Fixture providing conservative sampling parameters for TT tests."""
    return SamplingParams(
        temperature=0.0,  # Deterministic sampling
        max_tokens=10,    # Short generation for fast tests
        ignore_eos=False  # Respect EOS tokens
    ) 