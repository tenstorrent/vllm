import os
import sys
import runpy

from vllm import ModelRegistry

# Import and register model from tt-metal
from models.demos.t3000.llama2_70b.tt.llama_generation import TtLlamaModelForGeneration
ModelRegistry.register_model("TTLlamaForCausalLM", TtLlamaModelForGeneration)


def main():
    sys.argv.extend([
        "--model", "meta-llama/Meta-Llama-3.1-70B",
        "--block_size", "64",
        "--max_num_seqs", "32",
        "--max_model_len", "131072",
        "--max_num_batched_tokens", "131072",
        "--num_scheduler_steps", "10",
    ])
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()