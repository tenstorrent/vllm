# SPDX-License-Identifier: Apache-2.0
import dataclasses
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import ttnn
from transformers import TopPLogitsWarper

from vllm.attention.backends.abstract import AttentionBackend
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import get_sampler, SamplerOutput
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.logits_processor import LogitsProcessor, _apply_logits_processors
from vllm.model_executor.model_loader.tt_loader import TTModelLoader
from vllm.model_executor.models import supports_multimodal
from vllm.sequence import (CompletionSequenceGroupOutput, IntermediateTensors,
                           Logprob, SequenceGroupMetadata, SequenceOutput)
from vllm.utils import make_tensor_with_pad
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase

logger = init_logger(__name__)

"""
TLDR we drop support for TG llama (it was using device-side sampling anyway) and get a bunch of benefits
We don't support:
- data parallel, since it's only used by tg llama
- for async_out_proc we fully construct sample output synchronousl, only send it out async #TODO check if it makes sense
- aysnc_torch_proc, because that is a tg llama specific feature



TG llama always uses sampling on device, 
"""

@dataclass(frozen=True)
class TTLogprobData:
    chosen_ranks: torch.Tensor
    chosen_logprobs: torch.Tensor
    top_n_tokens: torch.Tensor
    top_n_logprobs: torch.Tensor

@dataclass(frozen=True)
class TTModelInput(ModelRunnerInputBase):
    """
    Used by the TTModelRunner.
    """
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    prompt_lens: Optional[List[int]]
    seq_groups: List[int]
    block_tables: torch.Tensor
    unpadded_batch_size: int
    sampling_params_list: List[Any] #TODO add proper type
    sampling_metadata: Optional["SamplingMetadata"]
    seq_lens: Optional[List[int]]
    query_lens: Optional[List[int]]
    multi_modal_kwargs: Dict[str, Any]
    cross_block_tables: torch.Tensor
    is_first_multi_step: bool = True
    is_last_step: bool = True
    async_callback: Optional[Callable] = None

    def as_broadcastable_tensor_dict(
            self) -> Dict[str, Union[int, torch.Tensor]]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "prompt_lens": self.prompt_lens,
            "seq_groups": self.seq_groups,
            "block_tables": self.block_tables,
            "unpadded_batch_size": self.unpadded_batch_size,
            "sampling_params_list": self.sampling_params_list,
            "sampling_metadata": self.sampling_metadata,
            "seq_lens": self.seq_lens,
            "query_lens": self.query_lens,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "cross_block_tables": self.cross_block_tables,
            "is_first_multi_step": self.is_first_multi_step,
            "is_last_step": self.is_last_step,
        }

        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type["TTModelInput"],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "TTModelInput":
        return cls(**tensor_dict)


class TTModelRunner(ModelRunnerBase[TTModelInput]):

    def __init__(
        self,
        vllm_config: VllmConfig,
        trace_mode: bool = True,
    ):
        ModelRunnerBase.__init__(self, vllm_config=vllm_config)

        # Currently, TT worker doesn't support chunked prefill.
        assert self.scheduler_config.chunked_prefill_enabled is False

        self.block_size = self.cache_config.block_size

        # whether to use ttnn tracing for model execution
        self.trace_mode = trace_mode
        override_tt_config = self.model_config.override_tt_config
        if (override_tt_config is not None
                and "sample_on_device_mode" in override_tt_config
                and override_tt_config["sample_on_device_mode"] is not None):
            raise ValueError("sample_on_device_mode is not supported")
        else:
            self.sample_on_device_mode = None  # whether to sample on device
        logger.info(
            "TTModelRunner: trace_mode=%s, sample_on_device_mode=%s",
            self.trace_mode,
            self.sample_on_device_mode,
        )


        if self.model_config.is_encoder_decoder:
            self.cached_enc_dec_data: Optional[Dict[int, Dict[
                str, Any]]] = None  # seq_id -> enc_dec_data

        # Detect if the model has "mrope" rope_scaling type.
        # mrope requires keep "rope_deltas" between prompt and decoding phases.
        if self.model_config.uses_mrope:
            assert ("TTModelRunner does not currently support models with "
                    "mrope rope_scaling")
                

    def load_model(self) -> None:
        # Note: using custom TT loader
        # instead of selecting from default vllm loaders
        loader = TTModelLoader(self.load_config)

        self.model = loader.load_model(vllm_config=self.vllm_config)
        if self.model_config.is_encoder_decoder:
            self.max_cross_blocks = (self.model.max_cross_attn_tokens //
                                     self.cache_config.block_size)

        # Initialize vLLM sampling components
        vocab_size = self.model_config.get_vocab_size()
        self.logits_processor = LogitsProcessor(vocab_size, logits_as_input=True)
        #TODO we are banking on having our logits shaped correctly, as if they came froma regular vllm model
        # and then got trimmed by the logitsprocessor. If we add prompt_logprobs or something,
        # we need to subclass logitsprocessor and do the prune_hidden_states but on logits.
        self.sampler = get_sampler()

        is_dp = (self.model_config.override_tt_config
                 and self.model_config.override_tt_config.get(
                     "data_parallel", 1) > 1)

        # Detect if the model is a TG Llama to use DP KV cache
        # vLLM doesn't know which blocks correspond to which DP device pool so
        # may allocate non-local blocks to a user. To avoid bad output because
        # of this, we maintain a seq_id_to_batch_slot mapping so that we can
        # place the users on the correct devices. This requires passing seq_id
        # and finished requests to the generator.
        # TODO: Extend this to support other DP models

        if ("Llama" in self.model_config.model
                and "70B" in self.model_config.model
                and self.device_config.device.get_num_devices() == 32
                and not is_dp):
            self.llama_tg = True
        else:
            self.llama_tg = False

        if  is_dp:
            self.dp_kv_cache = True
        else:
            self.dp_kv_cache = False


        if self.dp_kv_cache:
            # Map request id strs to seq group ids
            self.req_id_to_seq_id: Dict[str, int] = {}
            self.empty_slots = list(range(self.scheduler_config.max_num_seqs))
            self.seq_groups_to_batch_slot: Dict[int, int] = {}


    def get_model(self) -> nn.Module:
        return self.model
        
    def _compute_seq_lens_and_query_lens(self, seq_group_metadata_list, is_prompt):
        """Compute seq_lens and query_lens needed for SamplingMetadata"""
        seq_lens = []
        query_lens = []
        """
        This is needed for sampling, because regular vllm models process flattened batches.
        seq_len means how many tokens are in teh sequence in total,
        query lens means how many tokens are newly being processed,
        and are contained in the output logits.
        """
        for seq_group_metadata in seq_group_metadata_list:
            for seq_id, seq_data in seq_group_metadata.seq_data.items():
                if is_prompt:
                    seq_len = seq_data.get_len()
                    query_len = seq_len
                else:
                    seq_len = seq_data.get_len()
                    query_len = 1
                
                seq_lens.append(seq_len)
                query_lens.append(query_len)
        
        return seq_lens, query_lens

    def make_model_input_from_broadcasted_tensor_dict(
        self,
        tensor_dict: Dict[str, Any],
    ) -> TTModelInput:
        return TTModelInput.from_broadcasted_tensor_dict(tensor_dict, )

    def prepare_model_input(
            self,
            seq_group_metadata_list: List[SequenceGroupMetadata],
            virtual_engine: int = 0,
            finished_requests_ids: Optional[List[str]] = None) -> TTModelInput:

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[
            0].is_prompt  # prefill if True, otherwise decode
        assert all(
            x.is_prompt == is_prompt for x in seq_group_metadata_list
        ), "Currently only supporting all prefills or all decodes in seq group"

        unpadded_batch_size = len(seq_group_metadata_list)
        assert unpadded_batch_size > 0

        input_tokens_list: List[int] = []
        input_positions_list: List[int] = []
        prompt_lens_list: List[int] = []
        block_tables_list: List[List[int]] = []
        seq_groups_list: List[int] = []
        sampling_params_list = []
        multi_modal_kwargs: Dict[str, Any] = {}
        if supports_multimodal(self.model) and is_prompt:
            multi_modal_kwargs = {"images": []}
        cross_block_tables_list: List[List[int]] = []
        if self.dp_kv_cache and finished_requests_ids is not None:
            # Delete finished requests from req_id_to_seq_id
            finished_requests_seq_ids = []
            for req_id in finished_requests_ids:
                finished_requests_seq_ids.append(self.req_id_to_seq_id[req_id])
                del self.req_id_to_seq_id[req_id]

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(
                seq_ids
            ) == 1, "Currently only supporting one sequence per request group"
            seq_id = seq_ids[0]
            seq_groups_list.append(seq_id)
            if self.dp_kv_cache:
                # Add new request id to req_id_to_seq_id
                self.req_id_to_seq_id[seq_group_metadata.request_id] = seq_id

            multi_modal_data = seq_group_metadata.multi_modal_data
            seq_data = seq_group_metadata.seq_data[seq_id]

            if is_prompt:
                # tokens
                prompt_tokens = seq_data.get_token_ids()
                input_tokens_list.append(prompt_tokens)

                # prompt lengths
                prompt_lens_list.append(len(prompt_tokens))
            else:
                # tokens
                generation_token = seq_data.get_last_token_id()
                input_tokens_list.append(generation_token)

                # positions
                position = seq_data.get_len() - 1
                input_positions_list.append(position)

            block_table = seq_group_metadata.block_tables[seq_id]
            block_tables_list.append(block_table)

            # Multi-modal data
            # TODO: Replace with multi_modal_input_mapper
            # (used by CPU/GPU model runners) once TT models
            # no longer require raw PIL images
            if supports_multimodal(self.model) and is_prompt:
                if (multi_modal_data := seq_group_metadata.multi_modal_data):
                    assert "image" in multi_modal_data, (
                        "Currently only supporting image multi-modal inputs")
                    image = multi_modal_data[
                        "image"]  # this is of type PIL.Image.Image
                    multi_modal_kwargs["images"].append(image)
                else:
                    multi_modal_kwargs["images"].append(None)

            # Encoder-decoder data
            # (currently only supporting cross attention metadata
            # and not additional encoder data)
            if self.model_config.is_encoder_decoder:
                cross_block_table = seq_group_metadata.cross_block_table
                cross_block_tables_list.append(cross_block_table)

            sampling_params_list.append(seq_group_metadata.sampling_params)


        # Remove cached encoder-decoder data
        # for any seq ids that are not in the current batch
        # (assume they were either finished or preempted)
        if (self.model_config.is_encoder_decoder and not is_prompt
                and self.cached_enc_dec_data):
            seq_ids_to_del = []
            for seq_id in self.cached_enc_dec_data:
                if seq_id not in seq_groups_list:
                    seq_ids_to_del.append(seq_id)
            for seq_id in seq_ids_to_del:
                del self.cached_enc_dec_data[seq_id]

        # Convert lists to tensors and add padding

        block_tables = make_tensor_with_pad(block_tables_list,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pad=0)
        if self.model_config.is_encoder_decoder:
            cross_block_tables = make_tensor_with_pad(cross_block_tables_list,
                                                      dtype=torch.int32,
                                                      device="cpu",
                                                      pad=0)
        else:
            cross_block_tables = None
        if is_prompt:
            input_tokens = make_tensor_with_pad(input_tokens_list,
                                                dtype=torch.int32,
                                                device="cpu",
                                                pad=0)
            input_positions = 0
            prompt_lens = prompt_lens_list
        else:
            input_tokens = torch.tensor(input_tokens_list,
                                        dtype=torch.int32,
                                        device="cpu").view(-1, 1)
            input_positions = torch.tensor(input_positions_list,
                                           dtype=torch.int32,
                                           device="cpu")
            prompt_lens = None

            # TODO: Remove once TT models can support arbitrary batch sizes
            # Pad batch to max_num_seqs
            if input_tokens.shape[0] < self.scheduler_config.max_num_seqs:
                batch_pad_len = self.scheduler_config.max_num_seqs - \
                    input_tokens.shape[0]
                input_tokens = torch.cat([
                    input_tokens,
                    torch.zeros(batch_pad_len,
                                1,
                                dtype=torch.int32,
                                device="cpu")
                ])
                input_positions = torch.cat([
                    input_positions,
                    torch.ones(batch_pad_len, dtype=torch.int32, device="cpu")
                    * -1  # Pad with -1 to indicate no position
                ])
                block_tables = torch.cat([
                    block_tables,
                    torch.zeros(batch_pad_len,
                                block_tables.shape[1],
                                dtype=torch.int32,
                                device="cpu")
                ])
                if self.model_config.is_encoder_decoder:
                    cross_block_tables = torch.cat([
                        cross_block_tables,
                        torch.zeros(batch_pad_len,
                                    cross_block_tables.shape[1],
                                    dtype=torch.int32,
                                    device="cpu")
                    ])

            # Pad block_tables to max num blocks
            # so ttnn tracing can work (requires constant shape)
            if self.trace_mode:
                block_tables = torch.cat([
                    block_tables,
                    torch.zeros(block_tables.shape[0],
                                self.cache_config.num_gpu_blocks -
                                block_tables.shape[1],
                                dtype=torch.int32,
                                device="cpu")
                ],
                                         dim=1)
                if self.model_config.is_encoder_decoder:
                    # Note for vision models: the number of cross blocks
                    # may change if the number of image tiles changes
                    # or if prompts are text-only
                    cross_block_tables = torch.cat([
                        cross_block_tables,
                        torch.zeros(cross_block_tables.shape[0],
                                    self.max_cross_blocks -
                                    cross_block_tables.shape[1],
                                    dtype=torch.int32,
                                    device="cpu")
                    ],
                                                   dim=1)

        if self.dp_kv_cache:
            prev_seq_groups_list = list(self.seq_groups_to_batch_slot.keys())

            # check for preempted requests
            if seq_groups_list != prev_seq_groups_list and not is_prompt:
                finished_requests_seq_ids_current = [
                    seq_id for seq_id in prev_seq_groups_list
                    if seq_id not in seq_groups_list
                ]
            else:
                finished_requests_seq_ids_current = []

            # check for any remaining finished requests
            for seq_id in finished_requests_seq_ids:
                if seq_id not in finished_requests_seq_ids_current:
                    finished_requests_seq_ids_current.append(seq_id)

            # update the empty slots
            for req in finished_requests_seq_ids_current:
                empty_batch_slot = self.seq_groups_to_batch_slot[req]
                self.empty_slots.append(empty_batch_slot)
                del self.seq_groups_to_batch_slot[req]


        # Compute seq_lens and query_lens
        seq_lens, query_lens = self._compute_seq_lens_and_query_lens(
            seq_group_metadata_list, is_prompt)

        # Build sampling metadata
        generators = self.get_generators(finished_requests_ids)
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            query_lens,
            "cpu",
            pin_memory=False,
            generators=generators
        )

        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            seq_groups=seq_groups_list,
            block_tables=block_tables, unpadded_batch_size=unpadded_batch_size,
            sampling_params_list=sampling_params_list, multi_modal_kwargs=multi_modal_kwargs,
            cross_block_tables=cross_block_tables,
            sampling_metadata=sampling_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens
        )

    @torch.no_grad()
    def execute_model(
        self,
        model_input: TTModelInput,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        is_decode = model_input.prompt_lens is None

        # Note on async_out_proc + multi-step: for gpu/tpu, the N steps are
        # enqueued on device and the last step will trigger the output
        # processor for all outputs but the last. Currently for TT,
        # the inputs/outputs of each step are transferred between host/device,
        # and async_out_proc will trigger the output processor for step (i)
        # on host while device is executing step (i+1).
        use_async_out_proc = model_input.async_callback is not None
        # model_input.async_callback is set in the worker.prepare_input function
        # it's a partial function, and ctx is the context object that contains the output queue


        if not is_decode:
            assert num_steps == 1, "Num steps must be 1 for prefill"
        # always true if not using multi-step
        if model_input.is_first_multi_step:
            # This is a queue of torch tensor with sampled tokens
            # If we do async output processing, the queue is consumed by _send_prev_step_async_out, except the last step
            # If not, we consume the whole queue after executing the last step.
            self.cached_sampler_outputs = []
            for i in range(num_steps):
                sampler_outputs = self._execute_model_single_step(
                    model_input,
                    kv_caches,
                    is_decode,
                    use_async_out_proc,
                    step_idx=i)
                self.cached_sampler_outputs.append(sampler_outputs)
                next_token_ids = self._get_next_token_ids(sampler_outputs)
                if i < num_steps - 1:
                    # Prepare the inputs for the next step
                    new_input_tokens = next_token_ids.unsqueeze(dim=1).int()
                    if new_input_tokens.shape[
                            0] < self.scheduler_config.max_num_seqs:
                        # Pad batch to max_num_seqs
                        batch_pad_len = model_input.input_tokens.shape[
                            0] - new_input_tokens.shape[0]
                        new_input_tokens = torch.cat([
                            new_input_tokens,
                            torch.zeros(batch_pad_len,
                                        1,
                                        dtype=torch.int32,
                                        device="cpu")
                        ])

                    # Update input positions for all
                    # except those that are -1 (padding)
                    new_input_positions = torch.where(
                        model_input.input_positions == -1,
                        model_input.input_positions,
                        model_input.input_positions + 1)

                    model_input = dataclasses.replace(
                        model_input,
                        input_tokens=new_input_tokens,
                        input_positions=new_input_positions)

            if use_async_out_proc:
                assert model_input.async_callback is not None
                model_input.async_callback()  # trigger output processor, i'm not sure why we trigger here without appending any outputs to the context?

        sampler_outputs = []  # no outputs unless last step
        if model_input.is_last_step:  # always true if not using multi-step
            num_outputs = len(self.cached_sampler_outputs)
            if use_async_out_proc:
                # The queue should be getting consumed by _send_prev_step_async_out
                # the last step should have 1 output unless we have
                # scheduled less than self.scheduler_config.num_lookahead_slots
                # + 1 steps in which case there will be 0 outputs
                assert num_outputs <= 1, (
                    "Last step should have at most one output")
            for i in range(num_outputs):
                sampler_output = self.cached_sampler_outputs.pop(0)
                sampler_outputs.append(sampler_output)

        return sampler_outputs

    def _send_prev_step_async_out(self, model_input: TTModelInput, step_idx):
        # Get previous step's sampled tokens and send them to the output queue
        if step_idx > 0:
            sampler_output = self.cached_sampler_outputs.pop(0)
            async_callback = model_input.async_callback
            is_first_step_output = (step_idx == 1)
            ctx = async_callback.keywords["ctx"]
            ctx.append_output(outputs=[sampler_output],
                            seq_group_metadata_list=ctx.seq_group_metadata_list,
                            scheduler_outputs=ctx.scheduler_outputs,
                            is_async=False,
                            is_last_step=False,
                            is_first_step_output=is_first_step_output)
            async_callback()  # trigger output processor
        else:
            # trigger output processor in case last step was prefill
            # i don't quite get this, 
            assert model_input.async_callback is not None
            model_input.async_callback()

    def _execute_model_single_step(self,
                                   model_input: TTModelInput,
                                   kv_caches: List[torch.Tensor],
                                   is_decode,
                                   use_async_out_proc=False,
                                   step_idx=0):
        execute_model_kwargs = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": kv_caches,
            **(model_input.multi_modal_kwargs or {}),
        }
        if not is_decode:
            execute_model_kwargs["prompt_lens"] = model_input.prompt_lens
        else:
            execute_model_kwargs["start_pos"] = model_input.input_positions
        if model_input.cross_block_tables is not None:
            execute_model_kwargs[
                "cross_page_table"] = model_input.cross_block_tables

        if not is_decode:
            if self.dp_kv_cache:
                execute_model_kwargs[
                    "empty_slots"] = self.empty_slots[:model_input.
                                                      unpadded_batch_size]

            outputs = self.model.prefill_forward(**execute_model_kwargs)

            if self.dp_kv_cache:
                # update the batch slot table
                recently_filled_slots = self.empty_slots[:model_input.
                                                         unpadded_batch_size]
                self.empty_slots = self.empty_slots[model_input.
                                                    unpadded_batch_size:]

                for s in model_input.seq_groups:
                    self.seq_groups_to_batch_slot[
                        s] = recently_filled_slots.pop(0)

            if self.model_config.is_encoder_decoder:
                # Save encoder-decoder data for use in subsequent decode steps
                # (may need to be updated for future models)
                tt_out, prefill_cross_attention_masks, \
                prefill_full_text_row_masked_out_mask, \
                decode_cross_attention_masks, \
                 decode_full_text_row_masked_out_mask = outputs
                if self.cached_enc_dec_data is None:
                    self.cached_enc_dec_data = {}
                for i, seq_id in enumerate(model_input.seq_groups):
                    enc_dec_data = {
                        "prefill_cross_attention_masks":
                        prefill_cross_attention_masks[i],
                        "prefill_full_text_row_masked_out_mask":
                        prefill_full_text_row_masked_out_mask[i],
                        "decode_cross_attention_masks":
                        decode_cross_attention_masks[i],
                        "decode_full_text_row_masked_out_mask":
                        decode_full_text_row_masked_out_mask[i]
                    }
                    self.cached_enc_dec_data[seq_id] = enc_dec_data
            else:
                tt_out = outputs  # [batch_size, 1, vocab_size]
        else: #decode
            if self.model_config.is_encoder_decoder:
                assert self.cached_enc_dec_data is not None

                # Use encoder-decoder data from prefill step
                prefill_cross_attention_masks = [
                    self.cached_enc_dec_data[seq_id]
                    ["prefill_cross_attention_masks"]
                    for seq_id in model_input.seq_groups
                ]
                prefill_full_text_row_masked_out_mask = [
                    self.cached_enc_dec_data[seq_id]
                    ["prefill_full_text_row_masked_out_mask"]
                    for seq_id in model_input.seq_groups
                ]
                decode_cross_attention_masks = [
                    self.cached_enc_dec_data[seq_id]
                    ["decode_cross_attention_masks"]
                    for seq_id in model_input.seq_groups
                ]
                decode_full_text_row_masked_out_mask = [
                    self.cached_enc_dec_data[seq_id]
                    ["decode_full_text_row_masked_out_mask"]
                    for seq_id in model_input.seq_groups
                ]
                enc_dec_kwargs = {
                    "prefill_cross_attention_masks":
                    prefill_cross_attention_masks,
                    "prefill_full_text_row_masked_out_mask":
                    prefill_full_text_row_masked_out_mask,
                    "decode_cross_attention_masks":
                    decode_cross_attention_masks,
                    "decode_full_text_row_masked_out_mask":
                    decode_full_text_row_masked_out_mask
                }
            else:
                enc_dec_kwargs = {}

            if self.dp_kv_cache:
                # Calculate perm_table_tensor:
                # perm_table_tensor[new_idx] = current_slot_idx
                perm_table_tensor = torch.as_tensor(
                    [
                        self.seq_groups_to_batch_slot[s]
                        for s in model_input.seq_groups
                    ] + self.empty_slots,
                    dtype=torch.long,
                )

                assert perm_table_tensor.shape[
                    0] == self.scheduler_config.max_num_seqs
                # Calculate inverse_perm_indices:
                # inverse_perm_indices[current_slot_idx] = new_idx
                inverse_perm_indices = torch.empty_like(perm_table_tensor)
                inverse_perm_indices[perm_table_tensor] = torch.arange(
                    perm_table_tensor.size(0),
                    dtype=torch.long,
                )

                # permute the start_pos, tokens, and page_table
                execute_model_kwargs["start_pos"] = execute_model_kwargs[
                    "start_pos"][inverse_perm_indices]
                execute_model_kwargs["tokens"] = execute_model_kwargs[
                    "tokens"][inverse_perm_indices, :]
                execute_model_kwargs["page_table"] = execute_model_kwargs[
                    "page_table"][inverse_perm_indices, :]

            tt_out = self.model.decode_forward(**execute_model_kwargs,
                                               **enc_dec_kwargs,
                                               enable_trace=self.trace_mode,
                                               read_from_device=False)
            if use_async_out_proc:
                # trigger output processor on host while device is executing
                # next step
                self._send_prev_step_async_out(model_input, step_idx)
            tt_out = self.model.read_decode_output(
                tt_out,
                model_input.unpadded_batch_size,
                is_tokens=False)
            if self.dp_kv_cache:
                tt_out = tt_out[perm_table_tensor]

        # TT model already produced logits
        tt_logits = tt_out[:model_input.unpadded_batch_size, -1, :]  # [unpadded batch, vocab]
        #This is coincidentally the same shape as the logits we would get from a regular vllm model,
        # assuming we have no prompt logprobs, and one sequence per group.

        # Apply logits processing (including structured output filtering!)
        filtered_logits = self.logits_processor(
            lm_head=None,  # Ignored in our subclass
            hidden_states=tt_logits,  # Pass pre-computed logits
            sampling_metadata=model_input.sampling_metadata
        )

        # Sample tokens using standard vLLM sampler
        sampler_output = self.sampler(
            logits=filtered_logits,
            sampling_metadata=model_input.sampling_metadata
        )

        return sampler_output
        
    def _get_next_token_ids(self, sampler_output: SamplerOutput) -> torch.Tensor:
        """Extract next token IDs from sampler output."""
        next_token_ids = []
        for seq_group_output in sampler_output.outputs:
            for seq_output in seq_group_output.samples:
                next_token_ids.append(seq_output.output_token)
        return torch.tensor(next_token_ids, dtype=torch.int32, device="cpu")
