import dataclasses
import os
from typing import List, Optional, Tuple
import time
import math

import torch

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size, is_pin_memory_available
from vllm.worker.worker import raise_if_cache_size_invalid
from vllm.worker.tt_model_runner import TTModelRunner, TTModelInput
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoraNotSupportedWorkerBase, WorkerInput)

import ttnn
from ttnn import ReplicateTensorToMesh

logger = init_logger(__name__)


class TTCacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the TT and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)

        self.num_kv_heads = TTCacheEngine.get_num_kv_heads(
            model_config, parallel_config
        )

        self.block_size = cache_config.block_size
        self.num_tt_blocks = cache_config.num_gpu_blocks
        self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Initialize the cache.
        # List of KV caches. Entry is a list containing K and V tensors.
        logger.info("Allocating kv caches")
        self.tt_cache = self._allocate_kv_cache(
            self.num_tt_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device.
        The assumption is that KV cache for a layer is packed into one tensor. 
        We will have a separate tensor for K and V.
        """
        # K and V each have the following shape: (num_blocks, num_kv_heads, block_size, head_size)
        kv_cache_shape = (num_blocks, self.num_kv_heads, self.block_size, self.head_size)
        kv_cache: List[torch.Tensor] = []
        num_layers = self.num_attention_layers
        if device == "cpu":
            for _ in range(num_layers):
                # null block in CpuGpuBlockAllocator requires at least that
                # block to be zeroed-out.
                # Zero-initialize CPU cache
                cache_k = torch.zeros(kv_cache_shape,
                                      dtype=self.dtype,
                                      device=device)
                cache_v = torch.zeros(kv_cache_shape,
                                      dtype=self.dtype,
                                      device=device)
                kv_cache.append([cache_k, cache_v])
        else:
            cache_kv = torch.zeros(kv_cache_shape, dtype=self.dtype)
            for _ in range(num_layers):
                kv_tt = [ttnn.as_tensor(
                    lp,
                    device=self.device_config.device,
                    # TODO: this could be ShardTensorToMesh, removing need for init to know about TP=8. Could affect other calculations which use self.num_kv_heads, though.
                    mesh_mapper=ReplicateTensorToMesh(self.device_config.device),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b,
                    cache_file_name = self.cache_config.tt_cache_path / f"empty_cache_paged_attention{kv_cache_shape}"
                ) for lp in (cache_kv, cache_kv)]
                
                kv_cache.append(kv_tt)
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        raise NotImplementedError

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        raise NotImplementedError

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        raise NotImplementedError
    
    @staticmethod
    def get_num_kv_heads(
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        '''
        Returns the number of KV heads per attention layer. Makes the assumption
        that we are tensor parallel by a factor of 8.
        '''
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        # TODO: get num devices from device_config.device (device_mesh)
        # TODO: add get_num_devices to worker
        num_kv_heads //= 8 # TP=8, tries to use distributed worker if you give LLM 8 TP
        return num_kv_heads

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = TTCacheEngine.get_num_kv_heads(model_config, parallel_config)
        num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total


class TTWorker(LoraNotSupportedWorkerBase, LocalOrDistributedWorkerBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        is_driver_worker: bool,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.is_driver_worker = is_driver_worker

        assert self.device_config.device_type == "tt"
        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        self.trace_mode = True  # whether to use ttnn tracing for model execution, TODO: make this configurable

        self.model_runner: TTModelRunner = TTModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config,
            trace_mode=self.trace_mode,
        )
        
        self.cache_engine: List[TTCacheEngine]
        self.tt_cache: List[List]
        
        self.mesh_device = None  # initialized by init_device
        
        self.cached_model_input: Optional[TTModelInput] = None  # Only used for multi-step execution

        
    @property
    def do_metadata_broadcast(self) -> bool:
        return False  # TTWorker only supports single-worker execution

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.tt_cache

    def init_device(self) -> None:
        # TODO: Add support for devices other than T3K
        self.mesh_device = self._open_t3k_mesh_device()
        self.device_config.device = self.mesh_device
        
        # TODO: Add flag for enabling program cache
        self._enable_program_cache()
        
        # TODO: Add flag for enabling async mode
        self._enable_async_mode()

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available blocks for the TT KV cache and
        swappable CPU KV cache.

        The implementation may run profiling or other heuristics to determine
        the size of caches.

        Returns a Tuple[num_tt_blocks, num_cpu_blocks], where num_tt_blocks
        are blocks that are "active" on the device and can be appended to.
        num_cpu_blocks refers to "swapped" blocks in CPU memory and cannot be
        appended to.
        """
        # TODO: Add proper implementation which runs profiling on TT devices
        max_tokens_all_users = 131072
        num_tt_blocks = math.ceil(max_tokens_all_users / self.cache_config.block_size)
        num_tt_blocks = int(num_tt_blocks * 1.01)  # Add 1% to account for vLLM's watermark_blocks
        num_cpu_blocks = 0
        return num_tt_blocks, num_cpu_blocks

    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        """Initialize the KV cache with the given size in blocks.
        
        - Checks cache size is valid
        - Updates cache_config with num_gpu_blocks and num_cpu_blocks
        - init cache engine
        
        Note that CPU, TPU, and openvino workers don't use standard CacheEngine
        """
        # SKip check, since we're setting num_gpu_blocks much lower than would fit max_model_len
        # raise_if_cache_size_invalid(num_gpu_blocks, self.cache_config.block_size, self.model_config.max_model_len)
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks
        
        self._init_cache_engine()
        
    def _init_cache_engine(self):
        assert self.cache_config.num_gpu_blocks is not None
        
        # Get cache path from TT model for caching kv blocks
        self.cache_config.tt_cache_path = self.model_runner.model.cache_path
        
        self.cache_engine = TTCacheEngine(
            self.cache_config, 
            self.model_config, 
            self.parallel_config, 
            self.device_config)
        self.tt_cache = self.cache_engine.tt_cache
    
    def get_cache_block_size_bytes(self) -> int:
        """Return the size of a single cache block, in bytes. Used in
        speculative decoding.
        """
        raise NotImplementedError

    def prepare_worker_input(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> WorkerInput:
        """
        Prepare the inputs to WorkerBase.execute_worker from an execution
        request. This method may move data to the worker's local device. It is
        not allowed to communicate with other workers or devices.
        """
        virtual_engine = execute_model_req.virtual_engine
        num_steps = execute_model_req.num_steps
        num_seq_groups = len(execute_model_req.seq_group_metadata_list)
        
        # TODO: Add proper implementation, ignoring block allocation for now
        blocks_to_swap_in = 0
        blocks_to_swap_out = 0
        blocks_to_copy = 0

        return WorkerInput(
            num_seq_groups=num_seq_groups,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            virtual_engine=virtual_engine,
            num_steps=num_steps,
        )

    def execute_worker(self, worker_input: WorkerInput) -> None:
        """
        Process an execution request.
        
        Appears to do swap_in, swap_out, copy for KV blocks, right before executing the model.
        """
        # TODO: Add proper implementation, ignoring block allocation for now
        pass
    
    # Based on LocalOrDistributedWorkerBase::execute_model, excluding the distributed execution
    def execute_model(
        self,
        execute_model_req: Optional[ExecuteModelRequest] = None,
    ) -> Optional[List[SamplerOutput]]:
        """Executes at least one model step on the given sequences, unless no
        sequences are provided."""
        start_time = time.perf_counter()

        is_first_multi_step = execute_model_req.is_first_multi_step

        if not self.scheduler_config.is_multi_step or is_first_multi_step:
            inputs = self.prepare_input(execute_model_req)
            if inputs is None:
                return None
            model_input, worker_input, _ = inputs
            
        if self.scheduler_config.is_multi_step:
            if is_first_multi_step:
                self.cached_model_input = model_input
                worker_input = dataclasses.replace(
                    worker_input,
                    num_steps=execute_model_req.num_lookahead_slots + 1
                )
            else:
                assert self.cached_model_input is not None
                model_input = self.cached_model_input
                worker_input = WorkerInput()  # no worker input needed for subsequent steps
            model_input = dataclasses.replace(
                model_input,
                is_first_multi_step=is_first_multi_step,
                is_last_step=execute_model_req.is_last_step
            )

        num_steps = worker_input.num_steps

        self.execute_worker(worker_input)

        # If there is no input, we don't need to execute the model.
        if len(execute_model_req.seq_group_metadata_list) == 0:
            return []

        intermediate_tensors = None
        orig_model_execute_time = 0.0
        
        output = self.model_runner.execute_model(
            model_input=model_input,
            kv_caches=self.kv_cache
            if self.kv_cache is not None else None,
            intermediate_tensors=intermediate_tensors,
            num_steps=num_steps,
        )

        model_execute_time = time.perf_counter() - start_time
        
        if (self.observability_config is not None
                and self.observability_config.collect_model_execute_time
                and output is not None):
            for o in output:
                o.model_execute_time = (orig_model_execute_time +
                                        model_execute_time)

        # output is List[SamplerOutput]
        return output
    
    # TT-NN utilities
    
    def _get_devices(self):
        if self.mesh_device:
            devices = self.mesh_device.get_devices()
        else:
            devices = []
            logger.warning("No devices exist")
        return devices
    
    def _get_dispatch_core_type(self):
        dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
        if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
            dispatch_core_type = ttnn.device.DispatchCoreType.ETH
        return dispatch_core_type
    
    def _open_t3k_mesh_device(self):
        if self.trace_mode:
            device_params = {"trace_region_size": 14227456}  # TODO: make this configurable
        else:
            device_params = {}
        mesh_device = ttnn.open_mesh_device(
            ttnn.MeshShape(2, 4),
            dispatch_core_type=self._get_dispatch_core_type(),
            **device_params,
            mesh_type=ttnn.MeshType.Ring,
        )
        logger.debug(f"multidevice with {mesh_device.get_num_devices()} devices is created")
        return mesh_device
    
    def _enable_program_cache(self):
        devices = self._get_devices()
        if not devices or len(devices) == 0:
            logger.warning("No devices found to apply program cache to: PROGRAM CACHE DISABLED")
        for dev in devices:
            dev.enable_program_cache()
            
    def _enable_async_mode(self):
        devices = self._get_devices()
        if not devices or len(devices) == 0:
            logger.warning("No devices found to apply async mode to: ASYNC MODE DISABLED")
        for dev in devices:
            dev.enable_async(True)
        
    ## Destructor (used to close devices)
    
    def __del__(self):
        del self.model_runner  # Delete model runner first in case there are model arifacts (e.g ttnn trace)
        
        if self.mesh_device:
            devices = self.mesh_device.get_devices()
            
            # Disable program cache
            for dev in devices:
                dev.disable_and_clear_program_cache()
            
            # Disable async mode
            for dev in devices:
                dev.enable_async(False)
            
            # Dump device profiler
            for device in devices:
                ttnn.DumpDeviceProfiler(device)

            # Close devices
            ttnn.close_mesh_device(self.mesh_device)
            del self.mesh_device
        
        if hasattr(super(TTWorker, self), '__del__'):
            super().__del__()