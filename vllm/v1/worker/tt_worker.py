
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.worker_base import WorkerBase
from vllm.worker.tt_worker import open_mesh_device, close_mesh_device


logger = init_logger(__name__)

class TTWorker(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = True,
    ):
        super().__init__(vllm_config, local_rank, rank, distributed_init_method, is_driver_worker)
        
        # initialized by init_device
        self.mesh_device = None
        
        # whether to use ttnn tracing for model execution,
        # TODO: make this configurable
        self.trace_mode = True
    
    def init_device(self) -> None:
        self.mesh_device = open_mesh_device(
            self.model_config.override_tt_config,
            self.trace_mode
        )
        self.device_config.device = self.mesh_device
        
    def load_model(self):
        raise NotImplementedError
        
    ## Destructor (used to close devices)

    def __del__(self):
        # Delete model runner first in case there are model arifacts
        del self.model_runner

        if self.mesh_device:
            close_mesh_device(self.mesh_device, self.model_config.override_tt_config)
            del self.mesh_device

        if hasattr(super(), '__del__'):
            super().__del__()  # type: ignore
    