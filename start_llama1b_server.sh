#!/bin/bash

# Script to start vLLM server with Llama-3.2-1B model
# This server will be accessible at http://localhost:8000
# Based on the tt-metal README instructions

set -e

echo "==== Starting vLLM Server with Llama-3.2-1B ===="
echo ""

# Set default values
MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
# MESH_DEVICE="${MESH_DEVICE:-N150}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Device: $MESH_DEVICE"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo ""

echo "Starting vLLM server..."
echo "  - Server will be available at: http://localhost:$PORT"
echo "  - API endpoint: http://localhost:$PORT/v1"
echo "  - Health check: http://localhost:$PORT/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
# Using server_example_tt.py from the tt_metal examples
VLLM_RPC_TIMEOUT=100000\
    python examples/server_example_tt.py \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT"

