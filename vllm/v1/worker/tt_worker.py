# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.worker.tt_model_runner import TTModelRunner
from vllm.v1.worker.worker_base import WorkerBase
from vllm.worker.tt_worker import close_mesh_device, open_mesh_device

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
        super().__init__(vllm_config, local_rank, rank,
                         distributed_init_method, is_driver_worker)

        # Initialized by init_device
        self.mesh_device = None

        # Whether to use ttnn tracing for model execution
        override_tt_config = self.model_config.override_tt_config
        trace_key = "trace_mode"
        self.trace_mode = True
        if override_tt_config and trace_key in override_tt_config:
            assert override_tt_config[trace_key] in [True, False], \
                f"Invalid {trace_key}: {override_tt_config[trace_key]}"
            self.trace_mode = override_tt_config[trace_key]

    def init_device(self) -> None:
        self.mesh_device = open_mesh_device(
            self.model_config.override_tt_config, self.trace_mode)
        self.device_config.device = self.mesh_device

        # Init ModelRunner here, so that we have access to self.mesh_device.
        self.model_runner: TTModelRunner = TTModelRunner(
            vllm_config=self.vllm_config,
            mesh_device=self.mesh_device,
            trace_mode=self.trace_mode,
        )

    def load_model(self):
        self.model_runner.load_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    ## Destructor (used to close devices)

    def __del__(self):
        # Delete model runner first in case there are model arifacts
        del self.model_runner

        if self.mesh_device:
            close_mesh_device(self.mesh_device,
                              self.model_config.override_tt_config)
            del self.mesh_device

        if hasattr(super(), '__del__'):
            super().__del__()  # type: ignore
