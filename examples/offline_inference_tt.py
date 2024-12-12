import os
import json
import argparse
from tqdm import tqdm
import uvloop
import time
from pathlib import Path
from pkg_resources import resource_filename
from PIL import Image as PIL_Image
import numpy as np

from vllm import LLM, SamplingParams
from vllm import ModelRegistry
from vllm.entrypoints.openai.api_server import build_async_engine_client_from_engine_args
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import merge_async_iterators
from vllm.inputs.data import TokensPrompt
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.model_executor.models.mllama import MLLAMA_IMAGE_TOKEN, MLLAMA_IMAGE_TOKEN_ID

# Import and register models from tt-metal
from models.demos.t3000.llama2_70b.tt.generator_vllm import TtLlamaForCausalLM
from models.demos.llama3.tt.generator_vllm import TtMllamaForConditionalGeneration
ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaForCausalLM)
ModelRegistry.register_model("TTMllamaForConditionalGeneration", TtMllamaForConditionalGeneration)


def get_sample_multi_modal_llama_inputs():
    IMG_PATH = Path(resource_filename("llama_models", "scripts/resources/"))
    relative_img_paths = ["dog.jpg", "pasta.jpeg", "ocr_image.jpeg", "clutter.jpeg"]
    questions = [
        "Write a haiku for this image.",
        "What is for dinner?",
        "What is the full text of this image? Do OCR",
        "What objects are in this image?"
    ]
    inputs = []
    for relative_img_path, question in zip(relative_img_paths, questions):
        with open(IMG_PATH / relative_img_path, "rb") as f:
            img = PIL_Image.open(f).convert("RGB")
        prompt = f"{MLLAMA_IMAGE_TOKEN}{question}"
        inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})
    return inputs


def run_inference(
    prompts_json,
    max_tokens=128,
    max_seqs_in_batch=32,
    num_repeat_prompts=2,
    measure_perf=False,
    perf_prompt_len=None,
    greedy_sampling=False,  # Option to use greedy decoding instead of top-k/p
    async_engine=False,
    num_scheduler_steps=10,
    disable_async_output_proc=False,
    multi_modal=False,
):
    if multi_modal:
        model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        os.environ["MESH_DEVICE"] = "N300"
    else:
        model = "meta-llama/Meta-Llama-3.1-70B"
        os.environ["MESH_DEVICE"] = "T3K_RING"
    
    # LLM args
    engine_kw_args = {
        "model": model,
        "block_size": 64,
        "max_num_seqs": max_seqs_in_batch,
        "max_model_len": 131072,
        "disable_log_stats": False,
        "max_num_batched_tokens": 131072,
        "log_global_stats": True if measure_perf else False,
        "num_scheduler_steps": num_scheduler_steps,
        "disable_async_output_proc": disable_async_output_proc,
    }
    
    # Generation args
    ignore_eos = True if measure_perf else False

    if greedy_sampling:
        sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos, temperature=0.0)
    else:
        sampling_params = SamplingParams(max_tokens=max_tokens, ignore_eos=ignore_eos, top_k=10, top_p=0.9, temperature=1.0)

    # Prepare inputs
    if not measure_perf:
        if not multi_modal:
            # Load prompts from a JSON file
            with open(prompts_json, 'r') as file:
                prompts = json.load(file)
            assert isinstance(prompts, list), "Prompts must be a list of strings"
            if num_repeat_prompts is not None:
                prompts = prompts * num_repeat_prompts
        else:
            print("Ignoring prompts json for multi-modal inference")
            prompts = get_sample_multi_modal_llama_inputs() 
        print("Number of prompts:", len(prompts))
    else:
        assert perf_prompt_len is not None, "perf_prompt_len is required to generate dummy prompts"
        print("Measuring performance with dummy prompts of length", perf_prompt_len)
        print("Generating prompts with output length", max_tokens)
        
        # Prompt token ids (dummy prompts)
        prompt_token_ids_user = [0]*perf_prompt_len
        if not multi_modal:
            prompts = [{"prompt_token_ids": prompt_token_ids_user} for _ in range(max_seqs_in_batch)]
        else:
            prompt_token_ids_user.insert(0, MLLAMA_IMAGE_TOKEN_ID)
            random_pixels = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            rand_img = PIL_Image.fromarray(random_pixels, 'RGB')  # Create a PIL Image from the random pixel data
            prompts = [{"prompt_token_ids": prompt_token_ids_user, "multi_modal_data": {"image": rand_img}} for _ in range(max_seqs_in_batch)]
        
        # Sampling params
        sampling_params = sampling_params[:max_seqs_in_batch] if isinstance(sampling_params, list) else sampling_params
        sampling_params.max_tokens = max_tokens

        max_model_len = engine_kw_args["max_model_len"]
        assert_str = f"prompt length ({perf_prompt_len}) + num generated tokens ({sampling_params.max_tokens}) will exceed max_model_len ({max_model_len})"
        assert perf_prompt_len + sampling_params.max_tokens <= max_model_len, assert_str

    # Create and run LLM
    if not async_engine:
        llm = LLM(**engine_kw_args)
        if not measure_perf:
            generate_tokens(llm, prompts, sampling_params, print_output=True)
        else:
            run_inference_perf(llm, prompts, sampling_params)
    else:
        print("Using async engine")
        engine_args = AsyncEngineArgs(**engine_kw_args)
        async def _run_inference_async():
            async with build_async_engine_client_from_engine_args(engine_args) as llm:
                if not measure_perf:
                    await generate_tokens_async(llm, prompts, sampling_params, print_output=True)
                else:
                    await run_inference_perf_async(llm, prompts, sampling_params)
        uvloop.run(_run_inference_async())


def run_inference_perf(
    llm : LLM,
    prompts,
    sampling_params,
    N_warmup=1,
    N_inference=4,
):
    for i in tqdm(range(N_inference), desc="Inference runs"):
        if i == N_warmup:
            start_time = time.perf_counter()
        generate_tokens(llm, prompts, sampling_params, print_output=False)
    avg_time = (time.perf_counter()-start_time) / (N_inference-N_warmup)
    print(f"Average time taken per inference run: {avg_time:.2f} s")


async def run_inference_perf_async(
    llm : LLM,
    prompts,
    sampling_params,
    N_warmup=1,
    N_inference=4,
):
    for i in tqdm(range(N_inference), desc="Inference runs"):
        if i == N_warmup:
            start_time = time.perf_counter()
        await generate_tokens_async(llm, prompts, sampling_params, print_output=False)
    avg_time = (time.perf_counter()-start_time) / (N_inference-N_warmup)
    print(f"Average time taken per inference run: {avg_time:.2f} s")


def generate_tokens(llm : LLM, prompts, sampling_params, prompt_token_ids=None, print_output=True):
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params, prompt_token_ids)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        num_tokens_prompt = len(output.prompt_token_ids)
        num_tokens_output = len(output.outputs[0].token_ids)
        if print_output:
            print(f"Prompt ({num_tokens_prompt} tokens): {prompt!r}, Generated text ({num_tokens_output} tokens): {generated_text!r}\n")


async def generate_tokens_async(llm : MQLLMEngineClient, prompts, sampling_params, prompt_token_ids=None, print_output=True):
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
        num_tokens_prompt = len(res.prompt_token_ids)
        num_tokens_output = len(res.outputs[0].token_ids)
        if print_output and res.finished:
            print(f"Prompt ({num_tokens_prompt} tokens): {prompt!r}, Generated text ({num_tokens_output} tokens): {generated_text!r}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_json", type=str, default="tt_metal/prompts.json", help="Path to JSON file containing prompts")
    parser.add_argument("--measure_perf", action="store_true", help="Measure performance")
    parser.add_argument("--perf_prompt_len", type=int, default=128, help="Length of dummy prompts for performance measurement")
    parser.add_argument("--max_tokens", type=int, default=128, help="Length of outputs")
    parser.add_argument("--greedy_sampling", action="store_true", help="Use greedy decoding instead of top-k/p")
    parser.add_argument("--max_seqs_in_batch", type=int, default=32, help="Maximum batch size for inference")
    parser.add_argument("--async_engine", action="store_true", help="Use async engine")
    parser.add_argument("--disable_async_output_proc", action="store_true", help="Disable async output processing")
    parser.add_argument("--num_scheduler_steps", type=int, default=10, help="Number of scheduler steps")
    parser.add_argument("--multi_modal", action="store_true", help="Run multi-modal inference")
    args = parser.parse_args()

    run_inference(
        args.prompts_json,
        measure_perf=args.measure_perf,
        perf_prompt_len=args.perf_prompt_len,
        max_tokens=args.max_tokens,
        greedy_sampling=args.greedy_sampling,
        max_seqs_in_batch=args.max_seqs_in_batch,
        async_engine=args.async_engine,
        num_scheduler_steps=args.num_scheduler_steps,
        disable_async_output_proc=args.disable_async_output_proc,
        multi_modal=args.multi_modal,
    )
