from typing import List
import os
import sys
import json
import argparse
from tqdm import tqdm
import uvloop
import asyncio

from vllm import LLM, SamplingParams
from vllm import ModelRegistry
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import merge_async_iterators
from vllm.inputs.data import TokensPrompt
from vllm.engine.multiprocessing.client import MQLLMEngineClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tt_metal.models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration
ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaModelForGeneration)


def run_inference(
    prompts_json,
    max_tokens=128,
    max_seqs_in_batch=32,
    num_repeat_prompts=2,
    measure_perf=False,
    perf_prompt_len=None,
    greedy_sampling=False,  # Option to use greedy decoding instead of top-k/p
    async_engine=False,
):
    async def _run_inference(llm):
        # Generation args
        ignore_eos = True if measure_perf else False

        if greedy_sampling:
            sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos, temperature=0.0)
        else:
            sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos, top_k=10, top_p=0.9, temperature=1.0)

        if not measure_perf:
            # Load prompts from a JSON file
            with open(prompts_json, 'r') as file:
                prompts = json.load(file)
            assert isinstance(prompts, list), "Prompts must be a list of strings"
            if num_repeat_prompts is not None:
                prompts = prompts * num_repeat_prompts
            print("Number of prompts:", len(prompts))

            if not async_engine:
                generate_tokens(llm, prompts, sampling_params, print_output=True)
            else:
                await generate_tokens_async(llm, prompts, sampling_params, print_output=True)
        else:
            print("Note: Ignoring prompts for performance measurement")
            await run_inference_perf(llm, sampling_params, max_seqs_in_batch, max_tokens, input_prompt_len=perf_prompt_len, async_engine=async_engine)

    # Create an LLM.
    engine_kw_args = {
        "model": "meta-llama/Meta-Llama-3.1-70B",
        "block_size": 64,
        "max_num_seqs": max_seqs_in_batch,
        "max_model_len": 131072,
        "disable_log_stats": False,
        "max_num_batched_tokens": 131072,
        "log_global_stats": True if measure_perf else False,
        "num_scheduler_steps": 10,
    }
    if not async_engine:
        llm = LLM(**engine_kw_args)
        _run_inference(llm)
    else:
        print("Using async engine")
        engine_args = AsyncEngineArgs(**engine_kw_args)
        async def _run_inference_async():
            async with build_async_engine_client_from_engine_args(engine_args) as llm:
                await _run_inference(llm)
        uvloop.run(_run_inference_async())


async def run_inference_perf(
    llm : LLM,
    sampling_params,
    max_seqs_in_batch,
    max_tokens,
    prompts=None,
    input_prompt_len=None,  # Used to generate dummy prompts if prompts is None
    async_engine=False,
):
    if not async_engine:
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
    print("Generating prompts with output length", max_tokens)
    sampling_params.max_tokens = max_tokens

    model_config = llm.llm_engine.model_config if not async_engine else llm.model_config
    assert_str = f"prompt length ({input_prompt_len}) + num generated tokens ({sampling_params.max_tokens}) will exceed max_model_len ({model_config.max_model_len})"
    assert input_prompt_len + sampling_params.max_tokens <= model_config.max_model_len, assert_str

    # Compile run
    # print("Starting compile run")
    # generate_tokens(llm, prompts, sampling_params, prompt_token_ids, print_output=False)
    # print("Finished compile run")
    # llm.llm_engine.stat_loggers['global'].reset()  # Reset stats before inference run

    # Inference runs
    print("Starting inference runs")
    N_warmup = 1
    N_inference = 2
    for i in tqdm(range(N_inference), desc="Inference runs"):
        if i == N_warmup:  # Reset stats after warmup
            if not async_engine:
                llm.llm_engine.stat_loggers['global'].reset()
        if not async_engine:
            generate_tokens(llm, prompts, sampling_params, prompt_token_ids, print_output=False)
        else:
            await generate_tokens_async(llm, prompts, sampling_params, prompt_token_ids, print_output=False)
    print("Finished inference runs")

    # Collect stats
    if not async_engine:
        ttft = llm.llm_engine.stat_loggers['global'].time_to_first_token.avg / max_seqs_in_batch
        tpot = llm.llm_engine.stat_loggers['global'].time_per_output_token.avg
        print(f"Average time to first token per user: {ttft} s")
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


async def generate_tokens_async(llm : MQLLMEngineClient, prompts, sampling_params, prompt_token_ids=None, print_output=True):
    # async def _generate_tokens_async(llm, prompts, sampling_params, prompt_token_ids, print_output):
    # Use tokenized prompts if provided
    if prompt_token_ids is not None:
        prompts = []
        for single_prompt_token_ids in prompt_token_ids:
            prompts.append(TokensPrompt(prompt_token_ids=single_prompt_token_ids))
    
    if not isinstance(sampling_params, list):
        sampling_params = [sampling_params] * len(prompts)
    
    generators = []
    for i, (prompt, sp) in enumerate(zip(prompts, sampling_params)):
        generator = llm.generate(prompt, sp, request_id=f"test{i}")
        generators.append(generator)
    all_gens = merge_async_iterators(*generators)
    async for i, res in all_gens:
        prompt = res.prompt
        generated_text = res.outputs[0].text
        if print_output:
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        
    # try:
    #     loop = asyncio.get_event_loop()
    #     task = loop.create_task(_generate_tokens_async(llm, prompts, sampling_params, prompt_token_ids, print_output))
    #     def handle_task_result(task):
    #         try:
    #             task.result()  # Retrieve the task's result and catch the exception
    #         except Exception as e:
    #             print(f"Handled exception from task: {e}")
    #     task.add_done_callback(handle_task_result)
    # except RuntimeError:
    #     raise RuntimeError("generate_tokens_async must be called in an activate running event loop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_json", type=str, default="tt_metal/prompts.json", help="Path to JSON file containing prompts")
    parser.add_argument("--measure_perf", action="store_true", help="Measure performance")
    parser.add_argument("--perf_prompt_len", type=int, default=128, help="Length of dummy prompts for performance measurement")
    parser.add_argument("--max_tokens", type=int, default=128, help="Length of outputs")
    parser.add_argument("--greedy_sampling", action="store_true", help="Use greedy decoding instead of top-k/p")
    parser.add_argument("--max_seqs_in_batch", type=int, default=32, help="Maximum batch size for inference")
    parser.add_argument("--async_engine", action="store_true", help="Use async engine")
    args = parser.parse_args()

    run_inference(
        args.prompts_json,
        measure_perf=args.measure_perf,
        perf_prompt_len=args.perf_prompt_len,
        max_tokens=args.max_tokens,
        greedy_sampling=args.greedy_sampling,
        max_seqs_in_batch=args.max_seqs_in_batch,
        async_engine=args.async_engine,
    )
