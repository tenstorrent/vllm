# TT Hardware Logprobs Tests

Tests for implementing logprobs support on TT (Tenstorrent) hardware using test-driven development.

## Overview

**Current Status**: TT hardware rejects logprobs requests  
**Goal**: Implement logprobs with host-side sampling  
**Approach**: Test-first development with comprehensive code path coverage  

**Test Matrix**: All combinations of:
- **Sampling**: Greedy vs. Non-greedy
- **Processing**: Async vs. Non-async  
- **Logprobs**: Various configurations (None, 0, 1, 5)

## Quick Start

```bash
# From vLLM root directory
cd /path/to/vllm

# Test current behavior (should pass)
pytest tests/tt/test_basic_logprobs_integration.py::test_current_logprobs_rejection -v

# Test future behavior (should fail until implemented)
pytest tests/tt/test_basic_logprobs_integration.py::test_logprobs_across_all_code_paths -v --runxfail

# Run all tests
pytest tests/tt/ -v
```

## Test Files

- **`test_basic_logprobs_integration.py`** - Essential tests, start here
- **`test_logprobs.py`** - Comprehensive test matrix (48+ combinations)
- **`test_fixtures.py`** - Fixture validation tests

## Implementation Checklist

### Phase 1: Remove Restrictions
1. **Update `vllm/platforms/tt.py`**:
   - Remove/modify the `validate_request` method
   - Allow `logprobs` and `prompt_logprobs` with host-side sampling
2. **Test**: Current behavior tests should now fail (expected)

### Phase 2: Basic Implementation  
1. **Update TT model runner** (`vllm/worker/tt_model_runner.py`):
   - Modify `TTModelRunner._make_sampler_output` to include logprobs
   - Ensure host-side sampling when logprobs requested
2. **Test**: `pytest tests/tt/test_basic_logprobs_integration.py::test_logprobs_across_all_code_paths -v --runxfail`

### Phase 3: Full Implementation
1. **Add prompt logprobs support**
2. **Handle edge cases** (stop tokens, memory efficiency)  
3. **Add batch consistency across async modes**
4. **Test**: `pytest tests/tt/test_logprobs.py::TestTTLogprobsFutureBehavior -v --runxfail`

### Phase 4: Validate All Code Paths
1. **Test all combinations**:
   ```bash
   pytest tests/tt/ -k "greedy and async" -v
   pytest tests/tt/ -k "non_greedy and non_async" -v
   ```
2. **Test comprehensive matrix**: All 48+ combinations should pass

### Phase 5: Remove Test Markers
1. Remove `@pytest.mark.xfail` from all future behavior tests
2. Update current behavior tests to expect new behavior
3. **Final test**: `pytest tests/tt/ -v` (all should pass)

## Implementation Notes

### Host-Side Sampling Required
Logprobs support requires host-side sampling initially:
```python
config = {**tt_test_config, "max_logprobs": 5}
config["override_tt_config"] = '{"sample_on_device_mode": null}'
```

### Code Path Considerations
Different TT code paths require different handling:

- **Greedy vs Non-Greedy**: 
  - Greedy: Deterministic, single token selection
  - Non-Greedy: Stochastic, requires top-k/top-p logic

- **Async vs Non-Async**:
  - Async: May process logprobs asynchronously  
  - Non-Async: Synchronous logprobs computation

### Expected Logprobs Structure
- `logprobs=N`: Return top-N most likely tokens per position
- `logprobs=0`: Return only the sampled token  
- `prompt_logprobs=N`: Return top-N logprobs for each prompt token

### TT Configuration
Tests use parameters consistent with TT examples:
- `block_size=64`, `max_num_seqs=32`, `num_scheduler_steps=10`
- Environment automatically configured by fixtures

## Prerequisites

- TT hardware available
- vLLM with TT support
- HuggingFace access to Meta-Llama models: `huggingface-cli login`

**No manual setup required** - fixtures handle TT environment automatically.

## Troubleshooting

- **"TT hardware not available"** → Running on wrong system
- **"Currently not supporting logprobs on tt"** → Expected before implementation
- **Fixture not found** → Run from vLLM root, not `tests/tt/` 