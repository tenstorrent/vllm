# SPDX-License-Identifier: Apache-2.0
import argparse
import runpy
import sys

from offline_inference_tt import check_tt_model_supported, register_tt_models

register_tt_models()  # Import and register models from tt-metal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        type=str,
                        default="meta-llama/Llama-3.1-70B-Instruct",
                        help="Model name")
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=32,
        help="Maximum number of sequences to be processed in a single iteration"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help=
        "Maximum length of a sequence (including prompt and generated text).")
    args, _ = parser.parse_known_args()

    check_tt_model_supported(args.model)

    sys.argv.extend([
        "--model",
        args.model,
        "--block_size",
        "64",
        "--max_num_seqs",
        str(args.max_num_seqs),
        "--num_scheduler_steps",
        "10",
    ])
    if args.max_model_len is not None:
        sys.argv.extend(["--max_model_len", str(args.max_model_len)])
    runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')


if __name__ == '__main__':
    main()
