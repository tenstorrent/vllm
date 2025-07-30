# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ttnn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.tt_loader import TTModelLoader
from vllm.utils import LayerBlockType
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig

logger = init_logger(__name__)


class TTModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
        mesh_device: ttnn.MeshDevice,
        trace_mode: bool,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config
        self.device_config = vllm_config.device_config

        # Currently, TT model runner doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False

        self.mesh_device = mesh_device
        self.trace_mode = trace_mode

        # Whether to sample on device
        override_tt_config = self.model_config.override_tt_config
        sample_key = "sample_on_device_mode"
        self.sample_on_device_mode = None
        if (override_tt_config and sample_key in override_tt_config):
            assert override_tt_config[sample_key] in [
                "all", "decode_only"
            ], f"Invalid {sample_key}: {self.sample_on_device_mode}"
            self.sample_on_device_mode = override_tt_config[sample_key]

        logger.info(
            "TTModelRunner: trace_mode=%s, %s=%s",
            self.trace_mode,
            sample_key,
            self.sample_on_device_mode,
        )

    def load_model(self) -> None:
        loader = TTModelLoader(self.load_config)
        self.model = loader.load_model(vllm_config=self.vllm_config,
                                       model_config=self.model_config)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """

        kv_cache_groups = kv_cache_config.kv_cache_groups
        if len(kv_cache_groups) > 1:
            raise NotImplementedError(
                "Hybrid models with more than one KV cache type are not "
                "supported yet.")
        if isinstance(kv_cache_groups[0].kv_cache_spec, AttentionSpec):
            kv_cache_spec = kv_cache_groups[0].kv_cache_spec
        else:
            raise TypeError("Expected AttentionSpec")

        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            assert len(kv_cache_tensor.shared_by) == 1, (
                "KV cache shared by multiple layers is not supported for TT")

        # Make the assumption that we are tensor parallel by
        # min(number of devices, number of KV heads).
        # TODO: move this into model.allocate_kv_cache.
        model_config = self.model_config
        data_parallel = 1
        if (self.model_config.override_tt_config
                and "data_parallel" in model_config.override_tt_config):
            data_parallel = model_config.override_tt_config["data_parallel"]
        num_devices = self.mesh_device.get_num_devices() // data_parallel
        total_kv_heads = kv_cache_spec.num_kv_heads
        num_kv_heads = total_kv_heads // min(num_devices, total_kv_heads)

        kv_cache_shape = (kv_cache_config.num_blocks, num_kv_heads,
                          kv_cache_spec.block_size, kv_cache_spec.head_size)
        dtype = kv_cache_spec.dtype
        num_layers = model_config.get_num_layers_by_block_type(
            self.parallel_config, LayerBlockType.attention)

        # Allocate KV cache tensors.
        self.kv_caches = self.model.allocate_kv_cache(kv_cache_shape, dtype,
                                                      num_layers)
