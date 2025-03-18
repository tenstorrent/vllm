import argparse
import sys
import runpy

from offline_inference_tt import register_tt_models, check_tt_model_supported

register_tt_models()  # Import and register models from tt-metal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-70B", help="Model name")
    args = parser.parse_args()
    
    check_tt_model_supported(args.model)
    
    sys.argv.extend([
        "--model", args.model,
        "--block_size", "32",
        "--max_num_seqs", "1",
        "--max_model_len", "131072",
        "--max_num_batched_tokens", "131072",
        "--num_scheduler_steps", "10",
    ])
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()
