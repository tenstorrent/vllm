import os
from typing import List, Optional, Tuple

import torch

from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, ModelConfig,
                         ParallelConfig, SchedulerConfig)
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.worker.tt_model_runner import TTModelRunner
from vllm.worker.worker_base import (LocalOrDistributedWorkerBase,
                                     LoraNotSupportedWorkerBase, WorkerInput)

import ttnn

logger = init_logger(__name__)


class TTWorker(LoraNotSupportedWorkerBase, LocalOrDistributedWorkerBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config

        assert self.device_config.device_type == "tt"
        if self.cache_config.cache_dtype == "auto":
            self.cache_dtype = self.model_config.dtype
        else:
            self.cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                self.cache_config.cache_dtype]

        self.model_runner: TTModelRunner = TTModelRunner(
            model_config,
            parallel_config,
            scheduler_config,
            device_config,
            cache_config,
            load_config
        )
        
        self.mesh_device = None  # initialized by init_device
        
    @property
    def do_metadata_broadcast(self) -> bool:
        return False  # TTWorker only supports single-worker execution

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return self.tt_cache

    def init_device(self) -> None:
        # TODO: Add support for devices other than T3K
        self.mesh_device = self._open_t3k_mesh_device()
        
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
        raise NotImplementedError

    def initialize_cache(
        self,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
    ) -> None:
        """Initialize the KV cache with the given size in blocks.
        """
        raise NotImplementedError
    
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
        raise NotImplementedError

    def execute_worker(self, worker_input: WorkerInput) -> None:
        """
        Process an execution request.
        """
        raise NotImplementedError
    
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
        device_ids = [0, 4, 5, 1, 2, 6, 7, 3]
        num_devices_requested = len(device_ids)
        device_params = {}
        
        self.pci_ids = [ttnn.GetPCIeDeviceID(i) for i in device_ids[:num_devices_requested]]

        mesh_device = ttnn.open_mesh_device(
            ttnn.MeshShape(1, num_devices_requested),
            device_ids[:num_devices_requested],
            dispatch_core_type=self._get_dispatch_core_type(),
            **device_params,
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