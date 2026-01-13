#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Start vLLM server with Mistral 3.1 24B on Tenstorrent hardware
#
# Usage:
#   bash examples/server_tt_mistral24b.sh
#
# Then test with:
#   curl http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
#       "messages": [{"role": "user", "content": "Hello!"}]
#     }'

MODEL="${MODEL:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"

echo "Starting vLLM server with Mistral 3.1 24B on TT hardware"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Max model length: $MAX_MODEL_LEN"
echo "Max batch size: $MAX_NUM_SEQS"
echo ""
echo "Server will be available at: http://localhost:$PORT"
echo "API docs at: http://localhost:$PORT/docs"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --device tt \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --limit-mm-per-prompt '{"image": 1}' \
    --trust-remote-code

