from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import torch

from vllm.attention import AttentionMetadata
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig,
                         ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader.tt_loader import TTModelLoader
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)


@dataclass(frozen=True)
class TTModelInput(ModelRunnerInputBase):
    """
    Used by the TTModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    prompt_lens: Optional[torch.Tensor] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "prompt_lens": self.prompt_lens,
        }
        
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
            cls: Type["TTModelInput"],
            tensor_dict: Dict[str, Any],
    ) -> "TTModelInput":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        return cls(**tensor_dict)


class TTModelRunner(ModelRunnerBase[TTModelInput]):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        # Currently, TT worker doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config

        self.device = self.device_config.device

        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size

    def load_model(self) -> None:
        # Note: using custom TT loader instead of selecting from default vllm loaders
        loader = TTModelLoader(self.load_config)
        self.model = loader.load_model(model_config=self.model_config,
            device_config=self.device_config,
            lora_config=None,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config
        )

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> TTModelInput:
        return TTModelInput.from_broadcasted_tensor_dict(
            tensor_dict,
        )

    def prepare_model_input(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            virtual_engine: int = 0,
            finished_requests_ids: Optional[List[str]] = None
    ) -> TTModelInput:
        
        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt  # prefill if True, otherwise decode
        
        batch_size = len(seq_group_metadata_list)
        assert batch_size > 0
        
        input_tokens: List[int] = []
        input_positions: List[int] = []
        prompt_lens: List[int] = []

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1   # Only support one sequence per request group
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            
            if is_prompt:
                # tokens
                prompt_tokens = seq_data.get_token_ids()
                input_tokens.extend(prompt_tokens)
                
                # positions
                prompt_len = len(prompt_tokens)
                prompt_lens.append(prompt_len)
                input_positions.extend(list(range(prompt_len)))
            else:
                # tokens
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)
                
                # positions
                position = seq_data.get_len() - 1
                input_positions.append(position)
                
            # TODO: Get block table using seq_group_metadata.block_tables[seq_id]
                
        input_tokens = torch.tensor(input_tokens, dtype=torch.int32, device="cpu")
        input_positions = torch.tensor(input_positions, dtype=torch.int32, device="cpu")
        if is_prompt:
            prompt_lens = torch.tensor(prompt_lens,
                                    dtype=torch.int32,
                                    device="cpu")
        else:
            prompt_lens = None
        
        return TTModelInput(input_tokens, input_positions, prompt_lens)

    @torch.no_grad()
    def execute_model(
        self,
        model_input: TTModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "TT worker does not support multi-step execution.")

        model_executable = self.model
        execute_model_kwargs = {
            "input_ids":
            model_input.input_tokens,
            "positions":
            model_input.input_positions,
            "kv_caches":
            kv_caches,
            "attn_metadata":
            model_input.attn_metadata,
        }

        hidden_states = model_executable(**execute_model_kwargs)

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states,
                                           model_input.sampling_metadata)

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        return [output]
