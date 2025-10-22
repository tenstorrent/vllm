#!/bin/bash

# Script to stop all running vLLM servers

set -e

echo "==== Stopping All vLLM Servers ===="
echo ""

# Method 1: Stop by PID file
if [ -f /tmp/vllm_pids.txt ]; then
    echo "Stopping servers from PID file..."
    while read pid; do
        if ps -p $pid > /dev/null 2>&1; then
            echo "  Stopping process $pid..."
            kill $pid 2>/dev/null || true
        fi
    done < /tmp/vllm_pids.txt
    rm -f /tmp/vllm_pids.txt
    echo "✓ Stopped servers from PID file"
fi

# Method 2: Stop by process name (backup method)
echo ""
echo "Checking for any remaining vLLM processes..."
if pgrep -f "server_example_tt.py" > /dev/null; then
    echo "  Found remaining vLLM processes, stopping..."
    pkill -f "server_example_tt.py" || true
    echo "✓ Stopped all vLLM processes"
else
    echo "✓ No remaining vLLM processes found"
fi

echo ""
echo "All vLLM servers stopped!"
echo ""
echo "To stop Open WebUI:"
echo "  docker stop open-webui-vllm-multi"

