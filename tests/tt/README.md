# TT Hardware Logprobs Tests

This directory contains tests for implementing logprobs support on TT (Tenstorrent) hardware with host-side sampling.

## Comprehensive Test Coverage

**🎯 The tests now cover ALL TT hardware code paths!**

Our test matrix ensures logprobs work correctly across all combinations of:

### **Sampling Methods**
- **Greedy Sampling** (`temperature=0.0`) - Deterministic, argmax selection
- **Non-Greedy Sampling** (`temperature=1.0, top_k=10, top_p=0.9`) - Stochastic selection

### **Output Processing Modes**
- **Async Mode** (`disable_async_output_proc=False`) - Default async processing
- **Non-Async Mode** (`disable_async_output_proc=True`) - Synchronous processing

### **Logprobs Configurations**
- **Regular Logprobs**: `None`, `0` (sampled only), `1`, `5` (top-N)
- **Prompt Logprobs**: `1`, `3`, `5` (top-N for prompt tokens)
- **Combined**: Both logprobs and prompt_logprobs together

### **Test Matrix Coverage**
```
2 Sampling × 2 Async × 4 Logprobs × 3 Max_tokens = 48 test combinations
+ Additional edge cases and consistency tests
= Comprehensive coverage of all TT code paths
```

This ensures logprobs work correctly regardless of how TT hardware processes the request!

## Automatic TT Setup

**✨ The tests now automatically handle TT-specific setup!**

The `tt_setup` fixture automatically:

1. **🔗 Registers TT Models**: Registers out-of-tree TT model implementations with vLLM
   - `TTLlamaForCausalLM`, `TTMllamaForConditionalGeneration`
   - `TTQwen2ForCausalLM`, `TTQwen3ForCausalLM`, `TTMistralForCausalLM`

2. **🔧 Sets Environment Variables**: Configures TT-specific environment variables
   - `HF_MODEL=meta-llama/Llama-3.2-1B` - Model for tt_transformers to load
   - `WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml` - TT hardware architecture config
   - `MESH_DEVICE=N150` - Default TT device (N150 for single-chip testing)
   - `TT_LLAMA_TEXT_VER=tt_transformers` - TT implementation version

3. **⚙️ Uses TT Example Defaults**: Applies configuration parameters from official examples
   - `block_size=64` - Consistent with TT examples
   - `max_num_seqs=32` - Default batch size from examples
   - `num_scheduler_steps=10` - Scheduler configuration from examples

4. **🧹 Cleans Up**: Restores original environment after tests complete

This means **no manual setup is required** - just run the tests!

## Test Structure

### Current Tests

1. **`test_basic_logprobs_integration.py`** - Essential integration tests
   - Tests current rejection behavior (should pass initially)
   - **Matrix Tests**: All combinations of greedy/non-greedy × async/non-async
   - Determinism and variety validation across code paths

2. **`test_logprobs.py`** - Comprehensive test suite
   - **Full Matrix Coverage**: 48+ test combinations across all code paths
   - Edge cases and error conditions for each code path
   - Batch consistency and memory efficiency across async modes
   - HuggingFace comparison for greedy modes

3. **`test_fixtures.py`** - Fixture validation tests
   - Verifies that pytest fixtures are working correctly
   - Tests fixture combinations and configurations
   - Validates TT model registration and environment setup
   - Tests new TT-specific configuration parameters

4. **`conftest.py`** - Test configuration and fixtures
   - `tt_setup` - Session-wide TT environment setup (models + env vars + config)
   - `small_tt_model` - Provides the smallest TT-supported model
   - `tt_test_config` - Complete TT hardware configuration with example defaults
   - `tt_sampling_params` - TT-appropriate sampling parameters
   - `tt_available` - Checks if TT hardware is available

## Fixtures Usage

The tests use pytest fixtures for consistent TT-specific configuration:

```python
def test_example(small_tt_model, tt_test_config, tt_sampling_params):
    """Example of using TT fixtures."""
    # Model: meta-llama/Llama-3.2-1B (pre-validated)
    # Config: Complete TT setup with example defaults
    # Sampling: Conservative params for testing
    # Environment: All TT env vars set automatically
    
    llm = LLM(model=small_tt_model, **tt_test_config)
    sampling_params = SamplingParams(**tt_sampling_params, logprobs=5)
    
    # Extend config for specific needs
    config = {**tt_test_config, "max_logprobs": 5}
    config["override_tt_config"] = '{"sample_on_device_mode": null}'
```

**Available Fixtures:**
- `tt_setup` → Session-wide TT environment setup (auto-used by other fixtures)
- `small_tt_model` → `"meta-llama/Llama-3.2-1B"` (validated TT-supported model)
- `tt_test_config` → Complete TT configuration dict with example defaults
- `tt_sampling_params` → Conservative sampling parameters for testing
- `tt_available` → Boolean indicating TT hardware availability

### TT Test Configuration Details

The `tt_test_config` fixture includes all parameters from the TT examples:

```python
{
    # Model configuration
    "trust_remote_code": True,
    "max_model_len": 1024,
    "device": "tt",
    "dtype": "bfloat16",
    "enforce_eager": True,
    "tensor_parallel_size": 1,
    
    # TT-specific engine configuration (from examples)
    "block_size": 64,                    # Consistent with TT examples  
    "max_num_seqs": 32,                  # Default batch size from examples
    "num_scheduler_steps": 10,           # Scheduler configuration from examples
    "disable_log_stats": False,          # Enable logging for debugging
    "disable_async_output_proc": False,  # Keep async processing enabled
}
```

### TT Sampling Parameters

The `tt_sampling_params` fixture provides conservative defaults for testing:

```python
{
    "temperature": 0.0,    # Greedy sampling for reproducibility
    "max_tokens": 10,      # Short generation for fast tests
    "ignore_eos": False,   # Respect EOS tokens
}
```

## Supported Models

TT hardware only supports specific models. The tests use the smallest supported models for faster execution:

**Primary Test Models:**
- `meta-llama/Llama-3.2-1B` (smallest, fastest)
- `meta-llama/Llama-3.2-3B` (slightly larger backup)

**All TT-Supported Models:**
- Llama-3.1-8B: `meta-llama/Llama-3.1-8B`
- Llama-3.2-1B: `meta-llama/Llama-3.2-1B`  
- Llama-3.2-3B: `meta-llama/Llama-3.2-3B`
- Qwen-2.5-7B: `Qwen/Qwen2.5-7B` (N300 only)
- Qwen-2.5-72B: `Qwen/Qwen2.5-72B` (T3K only)  
- Llama-3.1-70B: `meta-llama/Llama-3.1-70B`
- DeepSeek-R1-Distill-Llama-70B: `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`

See the [TT-Metal LLMs table](https://github.com/tenstorrent/tt-metal?tab=readme-ov-file#llms) for the complete list and hardware requirements.

**⚠️ Model Access Requirements:**
Meta-Llama models require HuggingFace access approval. Follow these steps:
1. Request access at [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
2. Get your HF token and login: `huggingface-cli login`

## Environment Setup

### Automatic Setup (Recommended)

**✨ No manual setup needed!** The fixtures handle everything automatically.

Just ensure you have:
1. TT hardware available and properly configured
2. vLLM installed with TT support  
3. Access to supported TT models (HuggingFace login if needed)

**The fixtures will automatically set:**
```bash
HF_MODEL=meta-llama/Llama-3.2-1B
WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
MESH_DEVICE=N150
TT_LLAMA_TEXT_VER=tt_transformers
```

### Manual Setup (Advanced)

If you need to override the automatic setup, you can set environment variables before running tests:

```bash
# Optional: Override the default model
export HF_MODEL="meta-llama/Llama-3.2-3B"

# Optional: Set TT-specific environment variables
export MESH_DEVICE=N300  # or N150, T3K, TG based on your hardware
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml  # For N150/N300/T3K

# Optional: Set TT text version (default: tt_transformers)
export TT_LLAMA_TEXT_VER=tt_transformers  # or llama3_subdevices, llama2_70b

# Optional: Set cache path for TT weights
export TT_CACHE_PATH=/path/to/cache
```

See the [TT setup guide](../../tt_metal/README.md) for complete installation instructions.

## Running the Tests

### Prerequisites

- TT hardware available and properly configured
- vLLM installed with TT support  
- Access to supported TT models (see [TT-Metal LLMs table](https://github.com/tenstorrent/tt-metal?tab=readme-ov-file#llms))
- HuggingFace login if using Meta-Llama models

**⚠️ No manual environment setup required** - fixtures handle everything!

### Running Tests from the Correct Directory

**Important**: Always run pytest from the vLLM repository root directory (`vllm/`), not from the test subdirectory. This ensures proper module discovery and fixture loading.

```bash
# From vllm/ directory (correct):
cd /path/to/vllm
pytest tests/tt/ -v

# NOT from tests/tt/ directory (incorrect):
# cd tests/tt/
# pytest . -v  # This won't work properly
```

The test directory structure follows vLLM conventions:
- `conftest.py` - Contains pytest fixtures (automatic discovery)
- `utils.py` - Contains utility functions (explicit imports)  
- `test_*.py` - Test files that import utilities and use fixtures

### Test Fixtures First

Verify that fixtures are working correctly:

```bash
# Test fixture configuration and TT setup
pytest tests/tt/test_fixtures.py -v

# Test specific new features
pytest tests/tt/test_fixtures.py::test_tt_sampling_params_fixture -v
pytest tests/tt/test_fixtures.py::test_environment_variable_management -v

# Run fixture test directly for debugging
python tests/tt/test_fixtures.py
```

### Test Current Behavior (Before Implementation)

These tests should **PASS** with the current codebase:

```bash
# Test that logprobs are currently rejected
pytest tests/tt/test_basic_logprobs_integration.py::test_current_logprobs_rejection -v
pytest tests/tt/test_basic_logprobs_integration.py::test_current_prompt_logprobs_rejection -v

# Run all current behavior tests
pytest tests/tt/test_logprobs.py::TestTTLogprobsCurrentBehavior -v
```

### Test Future Behavior (After Implementation)

These tests are marked with `@pytest.mark.xfail` and should **FAIL** initially:

```bash
# Test basic functionality across all code paths (expected to fail until implemented)
pytest tests/tt/test_basic_logprobs_integration.py::test_logprobs_across_all_code_paths -v

# Run with --runxfail to see actual failure details
pytest tests/tt/test_basic_logprobs_integration.py::test_logprobs_across_all_code_paths -v --runxfail

# Test specific combinations
pytest tests/tt/test_basic_logprobs_integration.py -k "greedy and async" -v --runxfail
pytest tests/tt/test_basic_logprobs_integration.py -k "non_greedy and non_async" -v --runxfail

# Run comprehensive matrix tests (48+ combinations)
pytest tests/tt/test_logprobs.py::TestTTLogprobsFutureBehavior::test_tt_logprobs_across_all_code_paths -v --runxfail
```

### Running All Tests

```bash
# Run all TT logprobs tests (comprehensive matrix)
pytest tests/tt/ -v

# Run only on TT hardware (automatically skipped on other platforms)
pytest tests/tt/ -v -m "not tt_hardware or tt_hardware"

# Run specific test categories
pytest tests/tt/test_logprobs.py::TestTTLogprobsFutureBehavior -v --runxfail
pytest tests/tt/test_logprobs.py::TestTTLogprobsEdgeCases -v --runxfail
```

## Implementation Checklist

When implementing TT logprobs support, follow this checklist:

### Phase 1: Remove Current Restrictions

1. **Update `vllm/vllm/platforms/tt.py`**:
   - Remove or modify the `validate_request` method
   - Allow `logprobs` and `prompt_logprobs` parameters with host-side sampling

2. **Test**: Run current behavior tests - they should now fail (which is expected)

### Phase 2: Implement Basic Functionality

1. **Update TT model runner** to support logprobs computation:
   - Modify `TTModelRunner._make_sampler_output` to include logprobs
   - Ensure host-side sampling is used when logprobs are requested

2. **Test**: Run basic functionality tests across all code paths:
   ```bash
   pytest tests/tt/test_basic_logprobs_integration.py::test_logprobs_across_all_code_paths -v --runxfail
   ```

### Phase 3: Full Implementation

1. **Add prompt logprobs support**
2. **Handle edge cases** (stop tokens, memory efficiency)
3. **Add batch consistency across async modes**

4. **Test**: Run comprehensive matrix test suite:
   ```bash
   pytest tests/tt/test_logprobs.py::TestTTLogprobsFutureBehavior -v --runxfail
   ```

### Phase 4: Validate All Code Paths

1. **Test all combinations**: Ensure each code path works correctly:
   ```bash
   # Test greedy + async
   pytest tests/tt/ -k "greedy and async" -v
   
   # Test non-greedy + non-async  
   pytest tests/tt/ -k "non_greedy and non_async" -v
   
   # Test all combinations
   pytest tests/tt/test_logprobs.py::TestTTLogprobsFutureBehavior::test_tt_logprobs_across_all_code_paths -v
   ```

### Phase 5: Remove Test Markers

Once implementation is complete:

1. Remove `@pytest.mark.xfail` from all future behavior tests
2. Update current behavior tests to expect the new behavior
3. All tests should pass across all code paths:
   ```bash
   pytest tests/tt/ -v
   ```

## Key Implementation Notes

### Host-Side Sampling Required

For initial implementation, logprobs support requires host-side sampling:

```python
# In TT configuration (handled by fixtures)
config = {**tt_test_config, "max_logprobs": 5}
config["override_tt_config"] = '{"sample_on_device_mode": null}'
```

### Code Path Considerations

Different TT code paths may require different handling:

1. **Greedy vs Non-Greedy**: 
   - Greedy: Deterministic, single token selection
   - Non-Greedy: Stochastic, requires top-k/top-p logic

2. **Async vs Non-Async**:
   - Async: May process logprobs asynchronously
   - Non-Async: Synchronous logprobs computation

3. **Memory Management**: Different modes may have different memory constraints

### Expected Logprobs Structure

- `logprobs=N`: Return top-N most likely tokens per position
- `logprobs=0`: Return only the sampled token
- `prompt_logprobs=N`: Return top-N logprobs for each prompt token

### TT-Specific Configuration

The tests use configuration parameters consistent with TT examples:
- **Block size**: 64 (consistent across all TT examples)
- **Batch size**: 32 sequences max (default from examples)
- **Scheduler steps**: 10 (scheduler configuration from examples)
- **Environment**: Automatically configured for optimal TT performance

### Error Handling

Tests verify proper error handling for:
- Unsupported sampling configurations
- Memory constraints
- Invalid parameter combinations
- **Code path specific errors** for each async/sampling combination

## Debug Mode

Run tests in debug mode by executing the test files directly:

```bash
python tests/tt/test_basic_logprobs_integration.py
python tests/tt/test_fixtures.py
```

This provides detailed output about which tests pass/fail and why.

## Contributing

When adding new tests:

1. **Use fixtures** for consistent configuration:
   ```python
   def test_new_feature(small_tt_model, tt_test_config, tt_sampling_params):
       config = {**tt_test_config, "max_logprobs": 3}
       sampling = SamplingParams(**tt_sampling_params, logprobs=3)
       # Test implementation
   ```

2. **Consider all code paths**: Use parametrize for sampling and async modes:
   ```python
   @pytest.mark.parametrize("sampling_name,sampling_config,sampling_desc", SAMPLING_CONFIGS)
   @pytest.mark.parametrize("async_name,async_config,async_desc", ASYNC_CONFIGS)
   def test_new_feature(...):
   ```

3. Follow the existing pattern of current vs. future behavior
4. Use `@pytest.mark.xfail` for tests that should pass after implementation
5. Include comprehensive assertions about logprobs structure
6. Test both positive and negative cases for each code path
7. Add docstrings explaining the test purpose

## Troubleshooting

### Test Skipped: "TT hardware not available"
- Ensure you're running on a system with TT hardware
- Verify `current_platform.is_tt()` returns `True`

### Test Fails: "Currently not supporting logprobs on tt"
- This is expected before implementation
- Use `--runxfail` flag to see the actual test execution

### Test Fails: Fixture not found
- Make sure you're running tests from the correct directory
- Check that `conftest.py` is in the same directory
- Verify fixture names match the function parameters

### Test Fails: TT models not registered
- The `tt_setup` fixture should handle this automatically
- Check if `register_tt_models()` was called successfully
- Verify you have access to the tt-metal models directory

### Test Fails: Environment variables not set
- The `tt_setup` fixture should set these automatically
- Manual override: `export HF_MODEL=meta-llama/Llama-3.2-1B`
- Ensure you have access to the specified model on HuggingFace
- Check WH_ARCH_YAML and MESH_DEVICE are appropriate for your hardware

### Test Fails: Invalid TT configuration
- Verify `block_size=64` is supported by your TT hardware
- Check that `max_num_seqs=32` doesn't exceed hardware limits
- Ensure `num_scheduler_steps=10` is appropriate for your setup

### Test Fails: Code path specific errors
- **Async mode failures**: Check async processing configuration
- **Greedy mode failures**: Verify deterministic behavior is maintained
- **Non-greedy mode failures**: Check stochastic sampling configuration
- **Mixed mode failures**: Ensure all combinations work correctly

### Test Fails: Unexpected logprobs structure
- Check that host-side sampling is enabled
- Verify `max_logprobs` parameter is set correctly
- Ensure model is properly loaded with TT configuration
- Validate that TT-specific parameters don't conflict with logprobs
- **Verify code path**: Different sampling/async modes may have different logprobs structures 