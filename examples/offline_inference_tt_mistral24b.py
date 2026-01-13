#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Example: Mistral 3.1 24B multimodal inference on Tenstorrent hardware.

This script demonstrates how to run Mistral-Small-3.1-24B-Instruct-2503 
(a vision-language model) on TT hardware using vLLM.

Usage:
    # Text-only inference
    python examples/offline_inference_tt_mistral24b.py \\
        --prompt "Write a haiku about programming"
    
    # Vision + text inference
    python examples/offline_inference_tt_mistral24b.py \\
        --prompt "Describe this image in detail" \\
        --image-path /path/to/image.jpg
"""

import argparse
from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(
        description="Run Mistral 3.1 24B multimodal inference on TT hardware"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        help="Model name or path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        help="Path to image file (optional, for vision mode)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="Maximum number of sequences (batch size)"
    )
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    print(f"Device: TT (Tenstorrent)")
    print(f"Max model length: {args.max_model_len}")
    print(f"Max batch size: {args.max_num_seqs}")
    
    # Create LLM with TT device
    llm = LLM(
        model=args.model,
        device="tt",
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
    )
    
    # Prepare input
    if args.image_path:
        # Multimodal: text + image
        print(f"\nMode: Vision + Text")
        print(f"Image: {args.image_path}")
        
        image = Image.open(args.image_path).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]
        
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        outputs = llm.chat(messages, sampling_params=sampling_params)
    else:
        # Text-only
        print(f"\nMode: Text only")
        
        sampling_params = SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        
        outputs = llm.generate([args.prompt], sampling_params)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nPrompt: {args.prompt}")
    if args.image_path:
        print(f"Image: {args.image_path}")
    print(f"\nGenerated text:")
    print("-"*80)
    print(outputs[0].outputs[0].text)
    print("-"*80)
    print(f"\nTokens generated: {len(outputs[0].outputs[0].token_ids)}")
    print(f"Finish reason: {outputs[0].outputs[0].finish_reason}")


if __name__ == "__main__":
    main()

