# SPDX-License-Identifier: Apache-2.0
from torch import nn

from vllm.config import ModelConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import get_model_architecture

logger = init_logger(__name__)


class TTModelLoader(BaseModelLoader):

    def load_model(self, *, vllm_config: VllmConfig) -> nn.Module:
        """Load a model with the given configurations."""

        # For TT models, prepend "TT" to the architecture name,
        # e.g. "TTLlamaForCausalLM"
        model_config = vllm_config.model_config
        device_config = vllm_config.device_config
        scheduler_config = vllm_config.scheduler_config

        arch_names = model_config.hf_config.architectures
        for i in range(len(arch_names)):
            arch_names[i] = "TT" + arch_names[i]

        model_class, _ = get_model_architecture(model_config)

        data_parallel = 1
        if (model_config.override_tt_config
                and 'data_parallel' in model_config.override_tt_config):
            data_parallel = model_config.override_tt_config['data_parallel']
            logger.info("Overriding data_parallel to %d", data_parallel)

        model = model_class.initialize_vllm_model(
            model_config.hf_config,
            device_config.device,
            scheduler_config.max_num_seqs,
            max_seq_len=model_config.max_model_len,
            tt_data_parallel=data_parallel,
        )
        return model

    def download_model(self, model_config: ModelConfig) -> None:
        """Download a model so that it can be immediately loaded."""
        raise NotImplementedError

    def load_weights(self, model: nn.Module,
                     model_config: ModelConfig) -> None:
        """Load weights into a model. This standalone API allows 
        inplace weights loading for an already-initialized model"""
        raise NotImplementedError
