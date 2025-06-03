# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import math
import os
import random
import time
import logging
import requests
import torch
from typing import Any, List, Optional
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# === Server-side tokenization utility classes ===

class SimplePromptClient:
    """Simplified client for server-side tokenization and health checks"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, auth_token: str = None):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.headers = {}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
    
    def wait_for_healthy(self, timeout: float = 30.0) -> bool:
        """Wait for server to become healthy"""
        health_url = f"{self.base_url}/health"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(health_url, headers=self.headers, timeout=5)
                if response.status_code == 200:
                    logger.info("Server is healthy and ready!")
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        
        logger.error(f"Server did not become healthy within {timeout} seconds")
        return False
    
    def entokenize(self, prompt: str, model: str) -> dict:
        """Server-side tokenization"""
        url = f"{self.base_url}/tokenize"
        payload = {"model": model, "prompt": prompt}
        
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Error: {response.status_code}, {response.text}"
            logger.error(f"Server-side tokenization failed: {error_msg}")
            return {"error": error_msg}
    
    def detokenize(self, tokens: List[int], model: str) -> dict:
        """Server-side detokenization"""
        url = f"{self.base_url}/detokenize"
        payload = {"model": model, "tokens": tokens}
        
        response = requests.post(url, headers=self.headers, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            error_msg = f"Error: {response.status_code}, {response.text}"
            logger.error(f"Server-side detokenization failed: {error_msg}")
            return {"error": error_msg}


# === Tokenization functions ===

def get_fallback_tokenizer(model_name: str, fallback_model: str = "gpt2"):
    """Get tokenizer with fallback if primary model fails"""
    try:
        logger.info(f"Loading tokenizer for model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Successfully loaded tokenizer with vocab size: {tokenizer.vocab_size}")
        return tokenizer, model_name
    except Exception as e:
        logger.warning(f"Failed to load primary tokenizer: {str(e)}")
        try:
            logger.info(f"Trying fallback tokenizer: {fallback_model}")
            tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            logger.info(f"Using fallback tokenizer with vocab size: {tokenizer.vocab_size}")
            return tokenizer, fallback_model
        except Exception as e2:
            logger.error(f"Failed to load fallback tokenizer: {str(e2)}")
            raise RuntimeError("Could not load any tokenizer")


def tokenize_encode_client(prompt: str, tokenizer: Any, max_length: Optional[int], truncation: bool = False) -> List[int]:
    """Encode a prompt to tokens using the client-side tokenizer"""
    return tokenizer.encode(
        prompt, add_special_tokens=False, truncation=truncation, max_length=max_length
    )


def tokenize_decode_client(encoded_prompt: List[int], tokenizer: Any) -> str:
    """Decode tokens back to a string using the client-side tokenizer"""
    return tokenizer.decode(encoded_prompt)


def tokenize_encode_server(prompt: str, model: str, max_length: Optional[int], client: SimplePromptClient, truncation: bool = False) -> List[int]:
    """Encode a prompt to tokens using the server-side tokenizer"""
    result = client.entokenize(prompt, model)
    if "error" in result:
        raise RuntimeError(f"Server tokenization failed: {result['error']}")
    
    tokens = result["tokens"]
    # Truncate tokens if max_length is provided and tokens exceed that length
    if truncation and max_length is not None and len(tokens) > max_length:
        tokens = tokens[:max_length]
    return tokens


def tokenize_decode_server(tokens: List[int], model: str, client: SimplePromptClient) -> str:
    """Decode tokens back to a string using the server-side tokenizer"""
    result = client.detokenize(tokens, model)
    if "error" in result:
        raise RuntimeError(f"Server detokenization failed: {result['error']}")
    return result["prompt"]


# === Main prompt generation function ===

def generate_stable_prompt_tokens(
    input_length: int, 
    max_length: int,
    model_name: str,
    server_tokenizer: bool = False,
    client: Optional[SimplePromptClient] = None,
    fallback_model: str = "gpt2",
    seed: Optional[int] = None
) -> List[int]:
    """
    Generate a stable sequence of tokens by creating random tokens, then decoding and re-encoding.
    
    Args:
        input_length: Target number of tokens to generate
        max_length: Maximum allowed token length
        model_name: Name of the model/tokenizer to use
        server_tokenizer: Whether to use server-side tokenization
        client: SimplePromptClient instance for server-side tokenization
        fallback_model: Fallback model to use if the primary model fails
        seed: Random seed for reproducibility
        
    Returns:
        A list of integer token IDs
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
    else:
        torch.manual_seed(random.randint(0, 128000))
    
    # Load tokenizer if using client-side tokenization
    if not server_tokenizer:
        tokenizer, actual_model = get_fallback_tokenizer(model_name, fallback_model)
        vocab_size = tokenizer.vocab_size
    else:
        if client is None:
            raise ValueError("Client must be provided for server-side tokenization")
        tokenizer = model_name  # Just pass the model name for server tokenization
        actual_model = model_name
        # Estimate vocab size - could be retrieved from server if available
        vocab_size = 32000  # Default estimate for LLM models
    
    # Generate random tokens
    token_ids = torch.randint(0, vocab_size, (input_length,)).tolist()
    
    # First decoding - convert tokens to text
    if server_tokenizer:
        prompt_text = tokenize_decode_server(token_ids, tokenizer, client)
    else:
        prompt_text = tokenize_decode_client(token_ids, tokenizer)
    
    # First encoding - convert text back to tokens with truncation
    if server_tokenizer:
        encoded_tokens = tokenize_encode_server(prompt_text, tokenizer, max_length, client, truncation=True)
    else:
        encoded_tokens = tokenize_encode_client(prompt_text, tokenizer, max_length=max_length, truncation=True)
    
    # Second decoding - convert tokens to text again
    if server_tokenizer:
        decoded_text = tokenize_decode_server(encoded_tokens, tokenizer, client)
    else:
        decoded_text = tokenize_decode_client(encoded_tokens, tokenizer)
    
    # Final encoding - convert text back to tokens with truncation
    if server_tokenizer:
        final_tokens = tokenize_encode_server(decoded_text, tokenizer, max_length, client, truncation=True)
    else:
        final_tokens = tokenize_encode_client(decoded_text, tokenizer, max_length=max_length, truncation=True)
    
    return final_tokens


def generate_cleaned_random_prompts(
    num_prompts: int,
    input_len: int,
    output_len: int,
    model_name: str,
    client: SimplePromptClient,
    seed: int = 42
) -> List[tuple[str, int, int, None]]:
    """
    Generate cleaned random prompts using server-side tokenization.
    
    Returns list of (prompt_text, prompt_len, expected_output_len, multi_modal_data) tuples.
    """
    logger.info(f"Generating {num_prompts} cleaned random prompts...")
    
    prompts = []
    
    # Generate prompts with a progress bar
    for i in range(num_prompts):
        # Generate stable tokens
        tokens = generate_stable_prompt_tokens(
            input_length=input_len,
            max_length=input_len,  # Keep it at the exact requested length
            model_name=model_name,
            server_tokenizer=True,
            client=client,
            seed=seed + i  # Different seed for each prompt
        )
        
        # Decode tokens back to text for the prompt
        prompt_text = tokenize_decode_server(tokens, model_name, client)
        
        # The actual token count might be slightly different due to cleaning
        prompt_len = len(tokens)
        
        # Create prompt tuple (prompt_text, prompt_len, expected_output_len, multi_modal_data)
        prompts.append((prompt_text, prompt_len, output_len, None))
    
    logger.info(f"Generated {len(prompts)} cleaned prompts")
    avg_prompt_len = sum(p[1] for p in prompts) / len(prompts)
    logger.info(f"Average prompt length: {avg_prompt_len:.1f} tokens")
    
    return prompts


# === Original benchmark utility functions ===

def convert_to_pytorch_benchmark_format(args: argparse.Namespace,
                                        metrics: dict[str, list],
                                        extra_info: dict[str, Any]) -> list:
    """
    Save the benchmark results in the format used by PyTorch OSS benchmark with
    on metric per record
    https://github.com/pytorch/pytorch/wiki/How-to-integrate-with-PyTorch-OSS-benchmark-database
    """
    records = []
    if not os.environ.get("SAVE_TO_PYTORCH_BENCHMARK_FORMAT", False):
        return records

    for name, benchmark_values in metrics.items():
        record = {
            "benchmark": {
                "name": "vLLM benchmark",
                "extra_info": {
                    "args": vars(args),
                },
            },
            "model": {
                "name": args.model,
            },
            "metric": {
                "name": name,
                "benchmark_values": benchmark_values,
                "extra_info": extra_info,
            },
        }

        tp = record["benchmark"]["extra_info"]["args"].get(
            "tensor_parallel_size")
        # Save tensor_parallel_size parameter if it's part of the metadata
        if not tp and "tensor_parallel_size" in extra_info:
            record["benchmark"]["extra_info"]["args"][
                "tensor_parallel_size"] = extra_info["tensor_parallel_size"]

        records.append(record)

    return records


class InfEncoder(json.JSONEncoder):

    def clear_inf(self, o: Any):
        if isinstance(o, dict):
            return {k: self.clear_inf(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [self.clear_inf(v) for v in o]
        elif isinstance(o, float) and math.isinf(o):
            return "inf"
        return o

    def iterencode(self, o: Any, *args, **kwargs) -> Any:
        return super().iterencode(self.clear_inf(o), *args, **kwargs)


def write_to_json(filename: str, records: list) -> None:
    with open(filename, "w") as f:
        json.dump(records, f, cls=InfEncoder)
