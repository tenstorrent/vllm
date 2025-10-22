#!/bin/bash

# Script to start multiple vLLM servers for different models
# Each server runs on a different port and can be accessed by Open WebUI

set -e

echo "==== Starting Multiple vLLM Servers ===="
echo ""
echo "This script will help you start multiple vLLM model servers."
echo "Each server will run on a different port."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VLLM_DIR="/home/ttuser/aperezvicente/tt-vllm"
cd "$VLLM_DIR"

# Check if environment is set up
if [ -z "$TT_METAL_HOME" ]; then
    echo -e "${YELLOW}Setting up tt-metal environment...${NC}"
    export vllm_dir=$(pwd)
    source $vllm_dir/tt_metal/setup-metal.sh
    source $PYTHON_ENV_DIR/bin/activate
fi

echo -e "${GREEN}Environment ready!${NC}"
echo ""

# Function to start a model server
start_model_server() {
    local model=$1
    local port=$2
    local device=$3
    local log_file=$4
    
    echo -e "${YELLOW}Starting server for $model on port $port with device $device...${NC}"
    
    VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=$device \
        python examples/server_example_tt.py \
        --model "$model" \
        --host 0.0.0.0 \
        --port "$port" \
        > "$log_file" 2>&1 &
    
    local pid=$!
    echo -e "${GREEN}✓ Started $model (PID: $pid, Port: $port, Log: $log_file)${NC}"
    echo "$pid" >> /tmp/vllm_pids.txt
}

# Clean up old PID file
rm -f /tmp/vllm_pids.txt

echo "Available model configurations:"
echo ""
echo "1. Llama-3.2-1B-Instruct (N150, Port 8000)"
echo "2. Llama-3.2-3B-Instruct (N300, Port 8001)"
echo "3. Llama-3.1-8B-Instruct (N300, Port 8002)"
echo ""
echo "Enter the models you want to start (space-separated, e.g., '1 2 3'):"
read -p "Your choice: " choices

for choice in $choices; do
    case $choice in
        1)
            start_model_server "meta-llama/Llama-3.2-1B-Instruct" 8000 "N150" "/tmp/vllm_1b.log"
            ;;
        2)
            start_model_server "meta-llama/Llama-3.2-3B-Instruct" 8001 "N300" "/tmp/vllm_3b.log"
            ;;
        3)
            start_model_server "meta-llama/Llama-3.1-8B-Instruct" 8002 "N300" "/tmp/vllm_8b.log"
            ;;
        *)
            echo -e "${RED}Invalid choice: $choice${NC}"
            ;;
    esac
done

echo ""
echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}All selected servers are starting!${NC}"
echo -e "${GREEN}=====================================${NC}"
echo ""
echo "Server endpoints:"
echo "  - http://localhost:8000 (if selected)"
echo "  - http://localhost:8001 (if selected)"
echo "  - http://localhost:8002 (if selected)"
echo ""
echo "Log files:"
echo "  - /tmp/vllm_1b.log (if selected)"
echo "  - /tmp/vllm_3b.log (if selected)"
echo "  - /tmp/vllm_8b.log (if selected)"
echo ""
echo "To view logs:"
echo "  tail -f /tmp/vllm_1b.log"
echo ""
echo "To stop all servers:"
echo "  ./stop_multimodel_servers.sh"
echo ""
echo "To start Open WebUI with these models:"
echo "  docker-compose -f docker-compose-openwebui-multimodel.yaml up -d"
echo ""
echo "Wait about 30-60 seconds for servers to fully initialize..."

