#!/bin/bash

# Test script to send requests to the vLLM server

echo "Testing vLLM server at http://localhost:8000"
echo ""

# Test 1: Simple completion
echo "=== Test 1: Simple completion ==="
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B",
        "prompt": "San Francisco is a",
        "max_tokens": 32,
        "temperature": 1
    }'

echo -e "\n\n"

# Test 2: Chat completion
echo "=== Test 2: Chat completion ==="
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }'

echo -e "\n\n"

# Test 3: Multiple prompts
echo "=== Test 3: Story generation ==="
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B",
        "prompt": "Once upon a time in a distant galaxy,",
        "max_tokens": 64,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 10
    }'

echo -e "\n\nAll tests completed!"

