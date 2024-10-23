from typing import List
import os
import sys
import json
import argparse
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm import ModelRegistry

import torch
import time
import copy
import requests
import os
import sys
import json
from dataclasses import dataclass


from unittest.mock import patch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tt_metal.models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration
from tt_metal.models.demos.t3000.llama2_70b.tt.llama_common import (
    BASE_URL,
    load_llama_state_dict,
    setup_llama_env,
    check_mesh_device,
)
from tt_metal.models.demos.t3000.llama2_70b.tt.model_config import (
    get_model_config,
)
from vllm.worker.tt_worker import TTWorker
class MockModel(TtLlamaModelForGeneration):
    # mock implementation in TtLlamaModelForGeneration
    # see: tt-metal/models/demos/t3000/llama2_70b/tt/llama_generation.py
    # inherits from llama at the moment since only this model is currently used with vllm 
    def __init__(self, configuration, state_dict, model_args, tt_args, paged_attention_config=None, vllm=False):

        # Cache Weights setup
        n_layers = model_args.num_layers or 80

        self.params = copy.deepcopy(configuration)

        # required to setup model config 
        self.llama_version = model_args.llama_version
        self.max_batch_size = model_args.max_batch_size
        self.max_kv_context_len = model_args.max_kv_context_len

        self.mesh_device = tt_args.mesh_device

        # Initial model_config is set in decode mode
        # model conifg is required for vllm 
        model_config = get_model_config(
            llama_version=self.llama_version,
            max_batch_size=self.max_batch_size,
            max_context_len=self.max_kv_context_len,
            vllm=vllm,
        )
        self.model_config = model_config

    def prefill_forward_single_user(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        user_id: int,
        last_token_idx=None,
        page_table=None,
        kv_cache=None,
    ):
        return self.decode_forward(tokens=tokens, start_pos=start_pos)

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        page_table=None,
        kv_cache=None,
    ):
        assert len(tokens.shape) == 2
        batch, seqlen = tokens.shape
        forward_start = time.time()
        simulated_tps = 10000.0
        simulated_duration = 1.0 / simulated_tps
        # update the new tokens generated to the input id
        # vocab_size = tokenizer.nwords
        # logits: [batch, seqlen, vocab_size]
        logits = torch.randn((batch, seqlen, 128256))
        # send a token every period loops
        EOT_ID = 128009
        # EOS_ID = 128001
        send_index = 200
        send_token = EOT_ID
        if start_pos is not None:
            if isinstance(start_pos, int):
                # if single input per batch 
                cache_idxs = torch.tensor([start_pos for _ in range(batch)], dtype=torch.int64)
            else: # if start_pos is a tensor 
                # if start position is greater than index to send EOT
                send_token_mask = cache_idxs > send_index 
                cache_idxs = start_pos.to(dtype=torch.int64)
                # find positions where start pos passes send_index (ie we are done decording) + make 1D
                batch_indices = torch.nonzero(send_token_mask).squeeze() 
                # assign a high logit at at the send _token index so model will select it and generate the EOT so that generation stops 
                logits[batch_indices, 0, send_token] = 100.0 


        actual_duration = time.time() - forward_start
        # simulate forward latency
        time.sleep(max(simulated_duration - actual_duration, 0))
        return logits
    
    def forward(self, tokens: torch.Tensor, start_pos: int, page_table=None, kv_cache=None, prompt_lens=None):
        _, seq_len = tokens.shape
        if seq_len == 1:
            return self.decode_forward(tokens, start_pos, page_table=page_table, kv_cache=kv_cache)
        else:
            return self.prefill_forward(
                tokens, start_pos, page_table=page_table, kv_cache=kv_cache, prompt_lens=prompt_lens

            )

    @classmethod
    def initialize_vllm_model(cls, hf_config, t3k_mesh_device, max_batch_size):
        # TODO: pass in model args and tt args as parameters from vllm
        @dataclass
        class ModelArgs:
            llama_version: str = None
            ckpt_dir: str = None
            max_batch_size: int = 32  # overwritten by max_num_seqs from vllm
            num_layers: int = 80
            max_kv_context_len: int = 131072

        @dataclass
        class TTArgs:
            mesh_device: object = None
            cache_path: str = None

        # setup configs
        llama_version = "llama3"
        model_config, ckpt_dir, _, cache_path = setup_llama_env(
            llama_version=llama_version,
        )
        # check_mesh_device(t3k_mesh_device, model_config)

        # initialize arg classes
        model_args = ModelArgs(llama_version=llama_version, ckpt_dir=ckpt_dir, max_batch_size=max_batch_size)
        tt_args = TTArgs(mesh_device=t3k_mesh_device, cache_path=cache_path)

        # load state dict
        # state_dict = load_llama_state_dict(model_args.ckpt_dir, n_layers=model_args.num_layers)

        # TODO: delete this configuration setup once llama can directly accept hf_config
        from models.demos.t3000.llama2_70b.reference.llama.llama.model import ModelArgs as ReferenceModelArgs
        from pathlib import Path
        import json

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())
        configuration = ReferenceModelArgs(
            max_seq_len=model_args.max_kv_context_len,
            max_batch_size=model_args.max_batch_size,
            **params,
        )

        return cls(
            configuration=configuration, state_dict=state_dict, model_args=model_args, tt_args=tt_args, vllm=True
        )

@patch.object(TTWorker, "init_device", new=lambda x: None)
def run_inference(
    prompts_json,
    default_max_tokens=128,
    max_seqs_in_batch=32,
    num_repeat_prompts=2,
    measure_perf=False,
    perf_prompt_len=None,
    greedy_sampling=False,  # Option to use greedy decoding instead of top-k/p
):
    # Generation args
    ignore_eos = True if measure_perf else False
    
    if greedy_sampling:
        sampling_params = SamplingParams(max_tokens=default_max_tokens, ignore_eos=ignore_eos, temperature=0.0)
    else:
        sampling_params = SamplingParams(max_tokens=default_max_tokens, ignore_eos=ignore_eos, top_k=10, top_p=0.9, temperature=1.0)

    # Create an LLM.
    ModelRegistry.register_model("TTLlamaForCausalLM", MockModel)
    llm = LLM(model="meta-llama/Meta-Llama-3.1-70B", block_size=64, max_num_seqs=max_seqs_in_batch, max_model_len=131072, disable_log_stats=False, max_num_batched_tokens=131072)

    if not measure_perf:
        # Load prompts from a JSON file
        with open(prompts_json, 'r') as file:
            prompts = json.load(file)
        assert isinstance(prompts, list), "Prompts must be a list of strings"
        if num_repeat_prompts is not None:
            prompts = prompts * num_repeat_prompts
        print("Number of prompts:", len(prompts))
        
        generate_tokens(llm, prompts, sampling_params, print_output=True)
    else:
        print("Note: Ignoring prompts for performance measurement")
        run_inference_perf(llm, sampling_params, max_seqs_in_batch, input_prompt_len=perf_prompt_len)


def run_inference_perf(
    llm : LLM,
    sampling_params,
    max_seqs_in_batch,
    prompts=None,
    input_prompt_len=None  # Used to generate dummy prompts if prompts is None
):
    assert llm.llm_engine.log_stats, "disable_log_stats=False is required for llm to use stat loggers"
    if prompts is not None:
        print("Measuring performance with given prompts")
        prompts = prompts[:max_seqs_in_batch]  # Only run a single batch for performance measurement
    else:
        assert input_prompt_len is not None, "input_prompt_len is required to generate dummy prompts"
        print("Measuring performance with dummy prompts of length", input_prompt_len)
        prompt_token_ids = [[0]*input_prompt_len]*max_seqs_in_batch  # dummy prompts
    sampling_params = sampling_params[:max_seqs_in_batch] if isinstance(sampling_params, list) else sampling_params
    
    # Set an arbitrary max_tokens to simulate generating multiple tokens consecutively
    sampling_params.max_tokens = 33  # 1 prefill output token + 32 decode output tokens
    
    assert_str = f"prompt length ({input_prompt_len}) + num generated tokens ({sampling_params.max_tokens}) will exceed max_model_len ({llm.llm_engine.model_config.max_model_len})"
    assert input_prompt_len + sampling_params.max_tokens <= llm.llm_engine.model_config.max_model_len, assert_str

    # Compile run
    print("Starting compile run")
    generate_tokens(llm, prompts, sampling_params, prompt_token_ids, print_output=False)
    print("Finished compile run")
    llm.llm_engine.stat_loggers['global'].reset()  # Reset stats before inference run

    # Inference runs
    print("Starting inference runs")
    N_warmup = 1
    N_inference = 5
    for i in tqdm(range(N_inference), desc="Inference runs"):
        if i == N_warmup:  # Reset stats after warmup
            llm.llm_engine.stat_loggers['global'].reset()
        generate_tokens(llm, prompts, sampling_params, prompt_token_ids, print_output=False)
    print("Finished inference runs")

    # Collect stats
    ttft = llm.llm_engine.stat_loggers['global'].time_to_first_token.avg
    tpot = llm.llm_engine.stat_loggers['global'].time_per_output_token.avg
    print(f"Average time to first token (batch): {ttft} s")
    print(f"Average decode throughput: {1/tpot} t/s/u")
    

def generate_tokens(llm : LLM, prompts, sampling_params, prompt_token_ids=None, print_output=True):
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params, prompt_token_ids)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if print_output:
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_json", type=str, default="tt_metal/prompts.json", help="Path to JSON file containing prompts")
    parser.add_argument("--measure_perf", action="store_true", help="Measure performance")
    parser.add_argument("--perf_prompt_len", type=int, default=128, help="Length of dummy prompts for performance measurement")
    parser.add_argument("--greedy_sampling", action="store_true", help="Use greedy decoding instead of top-k/p")
    parser.add_argument("--max_seqs_in_batch", type=int, default=32, help="Maximum batch size for inference")
    args = parser.parse_args()

    run_inference(
        args.prompts_json,
        measure_perf=args.measure_perf,
        perf_prompt_len=args.perf_prompt_len,
        greedy_sampling=args.greedy_sampling,
        max_seqs_in_batch=args.max_seqs_in_batch
    )
