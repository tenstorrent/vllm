# Benchmark Utilities for vLLM

This directory contains utilities for advanced benchmarking of vLLM, including server-side tokenization and cleaned prompt generation.

## Features

- **Server-side tokenization**: More accurate token counting using the server's tokenizer
- **Cleaned prompt generation**: Stable, reproducible prompts through encode/decode cycles
- **Standard authentication**: Uses vLLM's standard environment variable authentication

## Additional Requirements

While most dependencies are already included in vLLM's common requirements, the following additional package may be needed:

```bash
pip install torch
```

- `torch`: Required for the CleanedPromptGenerator's random token generation (falls back to numpy if not available)

## Usage

These utilities are automatically used by `benchmark_serving.py` when:
- Using the `--dataset-name cleaned-random` option
- Enabling `--use-server-tokenization`

## Authentication

The utilities follow the exact same authentication pattern as the original vLLM benchmark code:

1. **Environment variables**: `AUTHORIZATION` or `OPENAI_API_KEY`
2. **Fallback**: Empty string (for open servers)

No command line arguments needed - uses only environment variables for authentication.

## Components

- `PromptClient`: Simplified client for server communication and tokenization
- `CleanedPromptGenerator`: Generator for stable, reproducible prompts 