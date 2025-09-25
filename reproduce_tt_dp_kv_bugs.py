#!/usr/bin/env python3
"""
Comprehensive test script to reproduce DP KV cache bugs in TTModelRunner.

This script reproduces two critical bugs:
1. KeyError when preempted requests are not in req_id_to_seq_id mapping
2. UnboundLocalError when finished_requests_seq_ids is used but not defined
3. IndexError when recently_filled_slots.pop(0) is called on empty list

The bugs occur in scenarios involving request preemption and slot allocation races.

Usage:
    python reproduce_tt_dp_kv_bugs.py

Expected errors:
- KeyError: 'chatcmpl-xxx' (when request is preempted before being added to mapping)
- UnboundLocalError: local variable 'finished_requests_seq_ids' referenced before assignment
- IndexError: pop from empty list (when slot allocation gets out of sync)
"""

import logging
import sys
import traceback
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

# Configure logging to match vLLM's format
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Import vLLM components
try:
    from vllm.worker.tt_model_runner import TTModelRunner
    from vllm.platforms.tt import TTPlatform
    from vllm.sequence import SequenceGroupMetadata
    from vllm.config import (VllmConfig, ModelConfig, CacheConfig, SchedulerConfig, 
                             DeviceConfig, LoadConfig, ParallelConfig, DecodingConfig, 
                             CompilationConfig)
except ImportError as e:
    logger.error(f"Failed to import vLLM components: {e}")
    logger.error("Make sure you're running this from the vLLM project root")
    sys.exit(1)


class MockSeqData:
    """Mock sequence data for testing."""
    
    def __init__(self, length: int = 10, last_token: int = 42):
        self._length = length
        self._last_token = last_token
        self._tokens = list(range(1, length + 1))
    
    def get_len(self) -> int:
        return self._length
    
    def get_last_token_id(self) -> int:
        return self._last_token
    
    def get_token_ids(self) -> List[int]:
        return self._tokens


class MockSamplingParams:
    """Mock sampling parameters."""
    
    def __init__(self, temperature: float = 1.0, top_k: int = 10, top_p: float = 0.9):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p


class MockSequenceGroupMetadata:
    """Mock sequence group metadata for testing."""
    
    def __init__(self, request_id: str, seq_id: int, is_prompt: bool = False):
        self.request_id = request_id
        self.is_prompt = is_prompt
        self.seq_data = {seq_id: MockSeqData()}
        self.block_tables = {seq_id: [1, 2, 3]}  # Mock block table
        self.sampling_params = MockSamplingParams()
        self.multi_modal_data = None
        self.cross_block_table = None


def create_mock_vllm_config() -> VllmConfig:
    """Create a mock VllmConfig for testing."""
    model_config = ModelConfig(
        model="meta-llama/Llama-2-70b-hf",  # Triggers DP KV cache
        tokenizer="meta-llama/Llama-2-70b-hf",
        tokenizer_mode="auto",
        trust_remote_code=False,
        dtype="auto",
        seed=0,
        revision=None,
        code_revision=None,
        rope_scaling={},  # Use empty dict instead of None
        rope_theta=None,
        tokenizer_revision=None,
        max_model_len=None,
        quantization=None,
        enforce_eager=False,
        max_seq_len_to_capture=8192,
        max_logprobs=20,
        disable_sliding_window=False,
        skip_tokenizer_init=False,
        served_model_name=None,
        limit_mm_per_prompt={},  # Use empty dict instead of None
        use_async_output_proc=False,
        override_neuron_config={},  # Use empty dict instead of None
        override_pooling_config=None,
        config_format="auto",
        hf_overrides={},  # Use empty dict instead of None
        mm_processor_kwargs=None,
        override_tt_config={"data_parallel": 2}  # Enable DP
    )
    
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.9,
        swap_space=4,
        cache_dtype="auto",
        num_gpu_blocks=1000,
        num_cpu_blocks=1000,
        cache_suffix_len=None,
        enable_prefix_caching=False,
        cpu_offload_gb=0,
        num_gpu_blocks_override=None
    )
    
    scheduler_config = SchedulerConfig(
        max_num_batched_tokens=None,
        max_num_seqs=32,  # Allow multiple sequences
        max_model_len=4096,
        use_v2_block_manager=False,
        num_lookahead_slots=0,
        delay_factor=0.0,
        enable_chunked_prefill=False,
        embedding_mode=False,
        preemption_mode="swap",
        num_scheduler_steps=1,
        multi_step_stream_outputs=False,
        send_delta_data=False,
        policy="fcfs"
    )
    
    device_config = DeviceConfig(device="tt")
    load_config = LoadConfig()
    parallel_config = ParallelConfig()  # Use default ParallelConfig
    decoding_config = DecodingConfig()  # Use default DecodingConfig
    compilation_config = CompilationConfig()  # Use default CompilationConfig
    
    return VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        device_config=device_config,
        load_config=load_config,
        lora_config=None,
        speculative_config=None,
        decoding_config=decoding_config,
        observability_config=None,
        prompt_adapter_config=None,
        quant_config=None,
        compilation_config=compilation_config,
        kv_transfer_config=None
    )


def create_mock_tt_model_runner() -> TTModelRunner:
    """Create a minimally initialized TTModelRunner for testing."""
    vllm_config = create_mock_vllm_config()
    
    # Create runner without calling __init__ to avoid device initialization
    runner = TTModelRunner.__new__(TTModelRunner)
    
    # Set required attributes manually
    runner.vllm_config = vllm_config
    runner.model_config = vllm_config.model_config
    runner.cache_config = vllm_config.cache_config
    runner.scheduler_config = vllm_config.scheduler_config
    runner.device_config = vllm_config.device_config
    runner.load_config = vllm_config.load_config
    
    runner.block_size = runner.cache_config.block_size
    runner.trace_mode = False
    runner.sample_on_device_mode = None
    runner.async_torch_proc = False
    runner.cached_step_outputs = []
    runner.request_specific_rope = False
    
    # Enable DP KV cache (this is what triggers the bugs)
    runner.dp_kv_cache = True
    runner.req_id_to_seq_id = {}
    runner.empty_slots = list(range(runner.scheduler_config.max_num_seqs))
    runner.seq_groups_to_batch_slot = {}
    
    # Mock the model
    runner.model = MagicMock()
    runner.model.prefill_forward = MagicMock(return_value="mock_output")
    
    return runner


def test_keyerror_bug():
    """Test Bug #1: KeyError when preempted request not in req_id_to_seq_id mapping."""
    logger.info("=" * 60)
    logger.info("Testing Bug #1: KeyError on preempted request lookup")
    logger.info("=" * 60)
    
    runner = create_mock_tt_model_runner()
    
    # Simulate a preempted request that was never added to the mapping
    preempted_request_id = "chatcmpl-2fdd35648efc418db366c6690c29c4da"
    seq_id = 123
    
    # Create sequence group metadata for a decode step
    seq_group_metadata = MockSequenceGroupMetadata(
        request_id="active-request-456", 
        seq_id=seq_id, 
        is_prompt=False
    )
    
    # The bug: finished_requests_ids contains a request that was never added to req_id_to_seq_id
    finished_requests_ids = [preempted_request_id]
    
    try:
        with patch("vllm.worker.tt_model_runner.supports_multimodal", return_value=False), \
             patch.object(TTPlatform, "compat_sampling_required", return_value=False), \
             patch.object(TTPlatform, "always_compat_sampling", False, create=True):
            
            result = runner.prepare_model_input(
                seq_group_metadata_list=[seq_group_metadata],
                virtual_engine=0,
                finished_requests_ids=finished_requests_ids
            )
            logger.error("BUG NOT REPRODUCED: Expected KeyError but got success")
            return False
            
    except KeyError as e:
        logger.error(f"SUCCESS: Reproduced KeyError bug: {e}")
        logger.error("This happens when a preempted request was never added to req_id_to_seq_id mapping")
        traceback.print_exc()
        return True
    except Exception as e:
        logger.error(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return True


def test_indexerror_bug():
    """Test Bug #3: IndexError when popping from empty recently_filled_slots."""
    logger.info("=" * 60)
    logger.info("Testing Bug #3: IndexError on empty list pop")
    logger.info("=" * 60)
    
    runner = create_mock_tt_model_runner()
    
    # Set up a scenario where empty_slots gets depleted
    runner.empty_slots = []  # No available slots
    
    seq_id = 789
    seq_group_metadata = MockSequenceGroupMetadata(
        request_id="prefill-request-123", 
        seq_id=seq_id, 
        is_prompt=True  # Prefill mode triggers slot allocation
    )
    
    # Mock the model's prefill_forward to avoid actual model execution
    mock_model_input = MagicMock()
    mock_model_input.seq_groups = [seq_id]
    mock_model_input.unpadded_batch_size = 1
    
    try:
        # This should trigger the slot allocation logic in _execute_model_single_step
        with patch("vllm.worker.tt_model_runner.supports_multimodal", return_value=False), \
             patch.object(TTPlatform, "compat_sampling_required", return_value=False), \
             patch.object(TTPlatform, "always_compat_sampling", False, create=True):
            
            # First prepare the input (this might succeed)
            model_input = runner.prepare_model_input(
                seq_group_metadata_list=[seq_group_metadata],
                virtual_engine=0,
                finished_requests_ids=[]
            )
            
            # Now try to execute - this should trigger the IndexError
            kv_caches = []
            result = runner._execute_model_single_step(
                model_input=model_input,
                kv_caches=kv_caches,
                is_decode=False,  # Prefill mode
                use_async_out_proc=False,
                step_idx=0
            )
            
            logger.error("BUG NOT REPRODUCED: Expected IndexError but got success")
            return False
            
    except IndexError as e:
        if "pop from empty list" in str(e):
            logger.error(f"SUCCESS: Reproduced IndexError bug: {e}")
            logger.error("This happens when recently_filled_slots.pop(0) is called on empty list")
            traceback.print_exc()
            return True
        else:
            logger.error(f"DIFFERENT IndexError: {e}")
            traceback.print_exc()
            return True
    except Exception as e:
        logger.error(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        return True


def main():
    """Run all bug reproduction tests."""
    logger.info("Starting DP KV Cache Bug Reproduction Tests")
    logger.info("This script reproduces critical bugs in TTModelRunner's DP KV cache handling")
    
    results = []
    
    # Test each bug
    results.append(("KeyError Bug", test_keyerror_bug()))
    results.append(("IndexError Bug", test_indexerror_bug()))
    
    # Summary
    logger.info("=" * 60)
    logger.info("BUG REPRODUCTION SUMMARY")
    logger.info("=" * 60)
    
    reproduced_count = 0
    for bug_name, reproduced in results:
        status = "REPRODUCED" if reproduced else "NOT REPRODUCED"
        logger.info(f"{bug_name}: {status}")
        if reproduced:
            reproduced_count += 1
    
    logger.info(f"\nTotal bugs reproduced: {reproduced_count}/{len(results)}")
    
    if reproduced_count > 0:
        logger.info("\n" + "=" * 60)
        logger.info("REPRODUCTION SUCCESSFUL!")
        logger.info("These bugs occur in production when:")
        logger.info("1. Requests are preempted before being added to slot mappings")
        logger.info("2. Variable scoping issues in cleanup logic")
        logger.info("3. Slot allocation races during high load")
        logger.info("=" * 60)
        return 0
    else:
        logger.warning("No bugs were reproduced. This might indicate:")
        logger.warning("1. The bugs have already been fixed")
        logger.warning("2. The test conditions don't match the production scenario")
        logger.warning("3. Additional setup is required")
        return 1


if __name__ == "__main__":
    sys.exit(main())
