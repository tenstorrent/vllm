import argparse
import os
import sys
import runpy

from vllm import ModelRegistry
from examples.offline_inference_tt import check_tt_model_supported

# Import and register models from tt-metal
old_llama_70b = False
if old_llama_70b:
    from models.demos.t3000.llama2_70b.tt.generator_vllm import TtLlamaForCausalLM
else:
    from models.demos.llama3.tt.generator_vllm import TtLlamaForCausalLM
from models.demos.llama3.tt.generator_vllm import TtMllamaForConditionalGeneration
ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaForCausalLM)
ModelRegistry.register_model("TTMllamaForConditionalGeneration", TtMllamaForConditionalGeneration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B", help="Model name")
    args = parser.parse_args()
    
    check_tt_model_supported(args.model)
    
    sys.argv.extend([
        "--block_size", "64",
        "--max_num_seqs", "32",
        "--max_model_len", "131072",
        "--max_num_batched_tokens", "131072",
        "--num_scheduler_steps", "10",
    ])
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()