"""
Utility functions for TT hardware tests.

This module contains helper functions that are used by TT tests but are not fixtures.
"""

import os
from typing import List, Dict, Any

def register_tt_models():
    """Register TT models with vLLM ModelRegistry.
    
    This function is adapted from examples/offline_inference_tt.py.
    It registers TT-specific model classes using string-based paths that are
    dynamically resolved at runtime, avoiding direct imports of TT modules.
    """
    # Avoid multiple registrations by checking if already registered
    if hasattr(register_tt_models, '_registered'):
        return
    
    try:
        import os
        from vllm import ModelRegistry
        
        # Get TT Llama version (same logic as offline_inference_tt.py)
        llama_text_version = os.getenv("TT_LLAMA_TEXT_VER", "tt_transformers")
        if llama_text_version == "tt_transformers":
            path_llama_text = "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
        elif llama_text_version == "llama3_subdevices":
            path_llama_text = "models.demos.llama3_subdevices.tt.generator_vllm:LlamaForCausalLM"
        elif llama_text_version == "llama2_70b":
            path_llama_text = "models.demos.t3000.llama2_70b.tt.generator_vllm:TtLlamaForCausalLM"
        else:
            raise ValueError(
                f"Unsupported TT Llama version: {llama_text_version}, "
                "pick one of [tt_transformers, llama3_subdevices, llama2_70b]")

        # Register TT model classes using string-based paths (same as offline_inference_tt.py)
        ModelRegistry.register_model("TTLlamaForCausalLM", path_llama_text)
        
        # Llama3.2 - Vision
        ModelRegistry.register_model(
            "TTMllamaForConditionalGeneration",
            "models.tt_transformers.tt.generator_vllm:MllamaForConditionalGeneration"
        )

        # Qwen2.5 - Text
        path_qwen_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
        ModelRegistry.register_model("TTQwen2ForCausalLM", path_qwen_text)
        ModelRegistry.register_model("TTQwen3ForCausalLM", path_qwen_text)

        # Mistral
        ModelRegistry.register_model(
            "TTMistralForCausalLM",
            "models.tt_transformers.tt.generator_vllm:MistralForCausalLM")
        
        # Mark as registered to avoid duplicate registrations
        register_tt_models._registered = True
        
    except Exception as e:
        raise ImportError(f"Failed to register TT models: {e}")


def get_supported_tt_models() -> List[str]:
    """Get list of models supported on TT hardware.
    
    Based on the models listed in tt-metal/models/tt_transformers/README.md.
    
    Returns:
        List of supported model names
    """
    return [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Llama-3.2-3B", 
        "meta-llama/Llama-3.1-8B",
        "meta-llama/Llama-3.1-70B",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/CodeLlama-7b-Python-hf",
        "microsoft/DialoGPT-medium",
        "EleutherAI/gpt-j-6b",
        "tiiuae/falcon-7b-instruct",
        "mosaicml/mpt-7b",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct", 
        "Qwen/Qwen2.5-32B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.1",
    ]


def validate_tt_model(model_name: str) -> None:
    """Validate that a model is supported on TT hardware.
    
    Args:
        model_name: Name of the model to validate
        
    Raises:
        ValueError: If model is not supported on TT hardware
    """
    supported_models = get_supported_tt_models()
    if model_name not in supported_models:
        raise ValueError(
            f"Model {model_name} is not supported on TT hardware. "
            f"Supported models: {supported_models}"
        )


def setup_tt_environment_variables() -> Dict[str, str]:
    """Set up required TT environment variables.
    
    Returns:
        Dictionary of original environment variable values for cleanup
    """
    original_env = {}
    
    # Required environment variables for TT hardware
    tt_env_vars = {
        "HF_MODEL": "meta-llama/Llama-3.2-1B",  # Default model
        "WH_ARCH_YAML": "/home/user/tt-metal/arch/wormhole_b0_80_arch_eth_dispatch.yaml",
        "MESH_DEVICE": "N150",  # Single device mesh configuration
        "TT_LLAMA_TEXT_VER": "tt_transformers",  # TT implementation version
        "TT_CACHE_PATH": "/tmp/tt_cache",  # Cache directory
    }
    
    for env_var, default_value in tt_env_vars.items():
        # Store original value for cleanup
        original_env[env_var] = os.environ.get(env_var)
        
        # Set environment variable if not already set
        if env_var not in os.environ:
            os.environ[env_var] = default_value
    
    return original_env


def restore_tt_environment_variables(original_env: Dict[str, str]) -> None:
    """Restore original environment variable values.
    
    Args:
        original_env: Dictionary of original environment values from setup_tt_environment_variables
    """
    for env_var, original_value in original_env.items():
        if original_value is None:
            # Variable wasn't set originally, remove it
            os.environ.pop(env_var, None)
        else:
            # Restore original value
            os.environ[env_var] = original_value 