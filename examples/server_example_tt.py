import argparse
import sys
import runpy
import json

from offline_inference_tt import register_tt_models, check_tt_model_supported

register_tt_models()  # Import and register models from tt-metal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="Model name")
    parser.add_argument("--dispatch_core_axis", type=str, default=None, help="Dispatch core axis [row, col]")
    args, unknown_args = parser.parse_known_args()
    
    check_tt_model_supported(args.model)

    override_tt_config = {}
    dispatch_core_axis = args.dispatch_core_axis
    if dispatch_core_axis:
        if dispatch_core_axis.lower() not in ["row", "col"]:
            raise ValueError(f"Invalid dispatch_core_axis: {dispatch_core_axis}, must be 'row' or 'col'")
        override_tt_config["dispatch_core_axis"] = dispatch_core_axis.lower()
    
    sys.argv = [
        "dummy", # FlexibleArgumentParser skips the first argument for some reason
        "--model", args.model,
        "--block_size", "64",
        "--max_num_seqs", "32",
        "--max_model_len", "131072",
        "--max_num_batched_tokens", "131072",
        "--num_scheduler_steps", "10",
        "--override_tt_config", json.dumps(override_tt_config),
    ]
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()
