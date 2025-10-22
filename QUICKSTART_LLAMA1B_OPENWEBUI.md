# Quick Start: Llama 1B Server + Open WebUI (No Login)

This guide walks you through setting up a vLLM server with Llama-3.2-1B and connecting it to Open WebUI without requiring authentication.

Based on the official vLLM documentation: https://docs.vllm.ai/en/latest/deployment/frameworks/open-webui.html

---

## 📋 Prerequisites

1. **tt-metal environment** must be properly set up (see [tt_metal/README.md](tt_metal/README.md))
2. **Hugging Face access** to Meta-Llama models (see [Accessing the Meta-Llama Hugging Face Models](tt_metal/README.md#accessing-the-meta-llama-hugging-face-models))
3. **Docker** installed (for Open WebUI)
4. **tt-metal device** available (N150, N300, T3K, or TG)

---

## 🚀 Step 1: Activate the Environment

First, activate your tt-metal + vLLM environment:

```bash
# Navigate to vLLM directory
cd /home/ttuser/aperezvicente/tt-vllm

# Set vLLM directory
export vllm_dir=$(pwd)

# Activate environment
source $vllm_dir/tt_metal/setup-metal.sh && source $PYTHON_ENV_DIR/bin/activate
```

---

## 🖥️ Step 2: Start the vLLM Server with Llama-3.2-1B

### Option A: Using the Quick Start Script (Recommended)

```bash
# Start the server on default port 8000
./start_llama1b_server.sh
```

### Option B: Manual Command

```bash
# For N150 device (most common for 1B model)
VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=N150 \
    python examples/server_example_tt.py \
    --model "meta-llama/Llama-3.2-1B" \
    --host 0.0.0.0 \
    --port 8000
```

### For Other Devices:

- **N300**: `MESH_DEVICE=N300`
- **T3K (QuietBox)**: `MESH_DEVICE=T3K`
- **TG (Galaxy)**: `MESH_DEVICE=TG`

**🔍 Verify the server is running:**

Open a new terminal and test:

```bash
# Check server health
curl http://localhost:8000/health

# List available models
curl http://localhost:8000/v1/models

# Test completions endpoint
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B",
        "prompt": "San Francisco is a",
        "max_tokens": 32,
        "temperature": 1
    }'
```

---

## 🌐 Step 3: Start Open WebUI (No Authentication)

**In a new terminal**, run:

### Option A: Using the Quick Start Script (Recommended)

```bash
cd /home/ttuser/aperezvicente/tt-vllm
./run_open_webui.sh
```

### Option B: Using Docker Compose

```bash
cd /home/ttuser/aperezvicente/tt-vllm
docker-compose -f docker-compose-openwebui.yaml up -d
```

### Option C: Manual Docker Command

```bash
docker run -d \
    --name open-webui-vllm \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
    -e WEBUI_AUTH=False \
    -e ENABLE_SIGNUP=False \
    -v open-webui:/app/backend/data \
    --restart always \
    ghcr.io/open-webui/open-webui:main
```

**🔍 Key Settings for No Authentication:**
- `WEBUI_AUTH=False` - Disables all authentication
- `ENABLE_SIGNUP=False` - Hides signup form
- No login page will appear!

---

## 🎉 Step 4: Access Open WebUI

1. **Open your browser** and navigate to:
   ```
   http://localhost:3000
   ```

2. **No login required!** You should see the chat interface immediately.

3. **Select the model** from the dropdown: `meta-llama/Llama-3.2-1B`

4. **Start chatting!** 🎊

---

## 🛠️ Troubleshooting

### Issue: Open WebUI can't connect to vLLM server

**Check if vLLM server is running:**
```bash
curl http://localhost:8000/v1/models
```

**If on Linux and Docker can't reach `host.docker.internal`, use:**
```bash
# Find your Docker bridge IP
ip addr show docker0

# Update the environment variable (usually 172.17.0.1)
docker run -d \
    --name open-webui-vllm \
    -p 3000:8080 \
    -e OPENAI_API_BASE_URL=http://172.17.0.1:8000/v1 \
    -e WEBUI_AUTH=False \
    ...
```

### Issue: Port already in use

**For vLLM server (port 8000):**
```bash
# Find process using the port
lsof -i :8000

# Kill the process if needed
kill -9 <PID>

# Or use a different port
PORT=8001 ./start_llama1b_server.sh
```

**For Open WebUI (port 3000):**
```bash
# Stop existing container
docker stop open-webui-vllm
docker rm open-webui-vllm

# Or use a different port
docker run -d -p 8080:8080 ...
# Then access at http://localhost:8080
```

### Issue: Still seeing a login page

**Verify WEBUI_AUTH is set:**
```bash
docker exec open-webui-vllm env | grep WEBUI_AUTH
```

Should show: `WEBUI_AUTH=False`

**If not, restart with correct settings:**
```bash
docker stop open-webui-vllm
docker rm open-webui-vllm
./run_open_webui.sh
```

### Issue: Model not loading or errors

**Check tt-metal environment variables:**
```bash
echo $TT_METAL_HOME
echo $PYTHON_ENV_DIR
```

**Ensure model weights are downloaded:**
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.2-1B')"
```

**Check device availability:**
```bash
# Should show your Tenstorrent device
tt-smi
```

---

## 📚 Useful Docker Commands

```bash
# View logs
docker logs -f open-webui-vllm

# Stop container
docker stop open-webui-vllm

# Start container
docker start open-webui-vllm

# Restart container
docker restart open-webui-vllm

# Remove container
docker rm open-webui-vllm

# Remove volume (delete all data)
docker volume rm open-webui
```

---

## 🎯 Advanced Configuration

### Custom TT Config Options

You can customize the vLLM server behavior with `--override_tt_config`:

```bash
VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=N150 \
    python examples/server_example_tt.py \
    --model "meta-llama/Llama-3.2-1B" \
    --override_tt_config '{"sample_on_device_mode": "all", "trace_mode": true}'
```

### Running with V1 Backend

```bash
VLLM_USE_V1=1 VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=N150 \
    python examples/server_example_tt.py \
    --model "meta-llama/Llama-3.2-1B" \
    --num_scheduler_steps 1
```

### Using Different Models

For **Llama-3.2-3B**:
```bash
MESH_DEVICE=N300 python examples/server_example_tt.py \
    --model "meta-llama/Llama-3.2-3B"
```

For **Llama-3.1-8B**:
```bash
MESH_DEVICE=N300 python examples/server_example_tt.py \
    --model "meta-llama/Llama-3.1-8B"
```

---

## 📊 Testing the Setup

### Test vLLM API directly:

```bash
# Simple completion
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B",
        "prompt": "Once upon a time",
        "max_tokens": 100,
        "temperature": 0.7
    }'

# Chat completion
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 100
    }'
```

### Test through Open WebUI:

1. Open http://localhost:3000
2. Type a message like "Hello, how are you?"
3. Press Enter
4. You should see a response from the Llama 1B model!

---

## 🔄 Full Restart Procedure

If you need to restart everything:

```bash
# 1. Stop Open WebUI
docker stop open-webui-vllm

# 2. Stop vLLM server (Ctrl+C in the terminal where it's running)

# 3. Restart vLLM server
cd /home/ttuser/aperezvicente/tt-vllm
export vllm_dir=$(pwd)
source $vllm_dir/tt_metal/setup-metal.sh && source $PYTHON_ENV_DIR/bin/activate
./start_llama1b_server.sh

# 4. Start Open WebUI (in new terminal)
docker start open-webui-vllm
# Or if removed:
./run_open_webui.sh
```

---

## 🎨 Features of Open WebUI

- ✅ **No Login Required** - Start chatting immediately
- 💬 **Multiple Conversations** - Create and manage different chats
- 📝 **Chat History** - All conversations are saved
- 🔄 **Model Switching** - Switch between models on the fly
- 📎 **Document Upload** - Analyze documents (if configured)
- 🎤 **Voice Input** - Talk to your model
- 🔧 **Customizable Prompts** - Save and reuse prompts
- 🎨 **Modern UI** - Beautiful, responsive interface

---

## 📖 Additional Resources

- **vLLM + tt-metal README**: [tt_metal/README.md](tt_metal/README.md)
- **Open WebUI Documentation**: [README_OPENWEBUI.md](README_OPENWEBUI.md)
- **Official vLLM Open WebUI Docs**: https://docs.vllm.ai/en/latest/deployment/frameworks/open-webui.html
- **Open WebUI GitHub**: https://github.com/open-webui/open-webui
- **vLLM Documentation**: https://docs.vllm.ai/

---

## 🎊 You're All Set!

You now have:
- ✅ vLLM server running with Llama-3.2-1B
- ✅ Open WebUI connected to your server
- ✅ No authentication required
- ✅ Ready to chat!

Enjoy your local AI chat experience! 🚀









