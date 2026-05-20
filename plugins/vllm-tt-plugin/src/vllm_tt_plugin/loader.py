# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import get_model_architecture
from vllm_tt_plugin.config import get_tt_config, get_tt_data_parallel_size

logger = init_logger(__name__)


class TTModelLoader(BaseModelLoader):
    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig
    ) -> nn.Module:
        """Load a model with the given configurations."""

        device_config = vllm_config.device_config
        scheduler_config = vllm_config.scheduler_config

        model_class, _ = get_model_architecture(model_config)
        optimizations = get_tt_config(vllm_config).get("optimizations", None)
        if optimizations is not None:
            assert optimizations in [
                "performance",
                "accuracy",
            ], f"""Invalid optimizations configuration `{optimizations}`,
            allowed values are 'performance' or 'accuracy'"""

        # TT model init expects the global DP-sized batch contract:
        # batch_size_per_dp * total_dp.
        tt_data_parallel = get_tt_data_parallel_size(vllm_config)
        max_batch_size = scheduler_config.max_num_seqs * tt_data_parallel

        model = model_class.initialize_vllm_model(
            model_config.hf_config,
            device_config.device,
            max_batch_size,
            max_seq_len=model_config.max_model_len,
            tt_data_parallel=tt_data_parallel,
            optimizations=optimizations,
        )
        return model

    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError
