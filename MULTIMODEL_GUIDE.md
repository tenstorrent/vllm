# Running Multiple Models with vLLM and Open WebUI

This guide explains how to run multiple LLM models simultaneously and access them through Open WebUI.

---

## 📋 Overview

There are **3 main approaches** to serve multiple models:

### Approach 1: Multiple vLLM Servers (Recommended ✅)
- Run multiple vLLM server instances on different ports
- Each server serves one model
- Open WebUI connects to all servers
- **Pros**: Simple, isolated, independent
- **Cons**: Requires more memory, more processes

### Approach 2: LoRA Adapters
- Run one base model with multiple LoRA adapters
- All adapters share the base model weights
- **Pros**: Memory efficient
- **Cons**: Limited to LoRA-compatible tasks, adapters must be compatible with base model

### Approach 3: Model Switching
- Run one server, manually switch models by restarting
- **Pros**: Uses least resources
- **Cons**: Downtime during switching, not practical

**We'll focus on Approach 1** as it's the most practical.

---

## 🚀 Quick Start: Multiple Models

### Step 1: Start Multiple vLLM Servers

You can start servers **manually** or use the **automated script**.

#### Option A: Using the Automated Script (Easiest)

```bash
cd /home/ttuser/aperezvicente/tt-vllm
./start_multimodel_servers.sh
```

The script will prompt you to select which models to start:
- **1**: Llama-3.2-1B-Instruct (N150, Port 8000)
- **2**: Llama-3.2-3B-Instruct (N300, Port 8001)
- **3**: Llama-3.1-8B-Instruct (N300, Port 8002)

Example: Enter `1 2` to start both 1B and 3B models.

#### Option B: Manual Start (More Control)

**Terminal 1 - Llama 1B on Port 8000:**
```bash
cd /home/ttuser/aperezvicente/tt-vllm
export vllm_dir=$(pwd)
source $vllm_dir/tt_metal/setup-metal.sh
source $PYTHON_ENV_DIR/bin/activate

VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=N150 \
    python examples/server_example_tt.py \
    --model "meta-llama/Llama-3.2-1B-Instruct" \
    --host 0.0.0.0 \
    --port 8000
```

**Terminal 2 - Llama 3B on Port 8001:**
```bash
cd /home/ttuser/aperezvicente/tt-vllm
export vllm_dir=$(pwd)
source $vllm_dir/tt_metal/setup-metal.sh
source $PYTHON_ENV_DIR/bin/activate

VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=N300 \
    python examples/server_example_tt.py \
    --model "meta-llama/Llama-3.2-3B-Instruct" \
    --host 0.0.0.0 \
    --port 8001
```

**Terminal 3 - Llama 8B on Port 8002:**
```bash
cd /home/ttuser/aperezvicente/tt-vllm
export vllm_dir=$(pwd)
source $vllm_dir/tt_metal/setup-metal.sh
source $PYTHON_ENV_DIR/bin/activate

VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=N300 \
    python examples/server_example_tt.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --host 0.0.0.0 \
    --port 8002
```

### Step 2: Verify Servers are Running

```bash
# Check all servers
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8002/health

# List models on each server
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models
curl http://localhost:8002/v1/models
```

### Step 3: Start Open WebUI with Multiple Models

#### Option A: Using Docker Compose (Recommended)

```bash
cd /home/ttuser/aperezvicente/tt-vllm
docker-compose -f docker-compose-openwebui-multimodel.yaml up -d
```

#### Option B: Manual Docker Command

```bash
docker run -d \
    --name open-webui-vllm-multi \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -e OPENAI_API_BASE_URLS="http://host.docker.internal:8000/v1;http://host.docker.internal:8001/v1;http://host.docker.internal:8002/v1" \
    -e WEBUI_AUTH=False \
    -e ENABLE_SIGNUP=False \
    -v open-webui-multi:/app/backend/data \
    --restart always \
    ghcr.io/open-webui/open-webui:main
```

**🔑 Key Setting:**
- `OPENAI_API_BASE_URLS` (plural!) with semicolon-separated URLs
- Each URL points to a different vLLM server

### Step 4: Access Open WebUI

1. Open browser: http://localhost:3000
2. **No login required!**
3. Click the **model dropdown** at the top
4. You should see all available models:
   - `meta-llama/Llama-3.2-1B-Instruct`
   - `meta-llama/Llama-3.2-3B-Instruct`
   - `meta-llama/Llama-3.1-8B-Instruct`
5. Select a model and start chatting!

---

## 🎯 Configuration Details

### Environment Variables for Open WebUI

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENAI_API_BASE_URL` | Single API endpoint | `http://host.docker.internal:8000/v1` |
| `OPENAI_API_BASE_URLS` | Multiple API endpoints (semicolon-separated) | `http://host:8000/v1;http://host:8001/v1` |
| `OPENAI_API_NAMES` | Custom names for endpoints (optional) | `Fast-1B;Powerful-3B` |
| `WEBUI_AUTH` | Disable authentication | `False` |
| `ENABLE_SIGNUP` | Hide signup form | `False` |

### Model-Device Recommendations

| Model | Recommended Device | Min Memory | Port (Suggested) |
|-------|-------------------|------------|------------------|
| Llama-3.2-1B-Instruct | N150 | 4GB | 8000 |
| Llama-3.2-3B-Instruct | N300 | 8GB | 8001 |
| Llama-3.1-8B-Instruct | N300 or T3K | 16GB | 8002 |
| Llama-3.1-70B-Instruct | T3K or TG | 128GB | 8003 |

**Device Environment Variables:**
- `MESH_DEVICE=N150` - Single chip
- `MESH_DEVICE=N300` - Two chips
- `MESH_DEVICE=T3K` - QuietBox (8 chips)
- `MESH_DEVICE=TG` - Galaxy (32 chips)

---

## 🛠️ Management Commands

### View Server Logs

```bash
# If using automated script
tail -f /tmp/vllm_1b.log
tail -f /tmp/vllm_3b.log
tail -f /tmp/vllm_8b.log

# View all logs
tail -f /tmp/vllm_*.log
```

### Stop All Servers

```bash
# Using stop script
./stop_multimodel_servers.sh

# Or manually
pkill -f "server_example_tt.py"
```

### Stop/Start Open WebUI

```bash
# Stop
docker stop open-webui-vllm-multi

# Start
docker start open-webui-vllm-multi

# Remove
docker rm open-webui-vllm-multi

# View logs
docker logs -f open-webui-vllm-multi
```

### Check Running Processes

```bash
# Check vLLM servers
ps aux | grep server_example_tt.py

# Check ports in use
lsof -i :8000
lsof -i :8001
lsof -i :8002
lsof -i :3000

# Check Docker containers
docker ps | grep open-webui
```

---

## 🔧 Troubleshooting

### Issue: Open WebUI only shows one model

**Problem**: Only one model appears in the dropdown.

**Solution**: 
1. Check that you used `OPENAI_API_BASE_URLS` (plural) not `OPENAI_API_BASE_URL`
2. Verify all servers are running:
   ```bash
   curl http://localhost:8000/v1/models
   curl http://localhost:8001/v1/models
   ```
3. Restart Open WebUI:
   ```bash
   docker restart open-webui-vllm-multi
   ```

### Issue: "Chat template" error

**Problem**: Error about missing chat template.

**Solution**: Use **Instruct** models, not base models:
- ✅ `meta-llama/Llama-3.2-1B-Instruct`
- ❌ `meta-llama/Llama-3.2-1B` (base model, no chat template)

### Issue: Port already in use

**Problem**: `Address already in use` error.

**Solution**:
```bash
# Find process using the port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
python examples/server_example_tt.py --port 8003
```

### Issue: Docker can't reach servers

**Problem**: Open WebUI can't connect to vLLM servers.

**Solution on Linux**:
```bash
# Find Docker bridge IP
ip addr show docker0

# Use that IP instead (usually 172.17.0.1)
-e OPENAI_API_BASE_URLS="http://172.17.0.1:8000/v1;http://172.17.0.1:8001/v1"
```

### Issue: Out of memory

**Problem**: Server crashes with OOM error.

**Solution**:
1. Use smaller models on limited devices
2. Don't run too many models simultaneously
3. Check device memory:
   ```bash
   tt-smi
   ```

---

## 📊 Testing Multiple Models

### Test Each Server Individually

```bash
# Test 1B model
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }'

# Test 3B model
curl http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.2-3B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }'
```

### Test Through Open WebUI

1. Open http://localhost:3000
2. Click model dropdown
3. Select "Llama-3.2-1B-Instruct"
4. Send a message: "Hello!"
5. Switch to "Llama-3.2-3B-Instruct"
6. Send another message: "Hello!"
7. Compare responses!

---

## 🎨 Advanced: Custom Model Names in Open WebUI

You can give friendly names to your models:

```bash
docker run -d \
    --name open-webui-vllm-multi \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -e OPENAI_API_BASE_URLS="http://host.docker.internal:8000/v1;http://host.docker.internal:8001/v1" \
    -e OPENAI_API_NAMES="Lightning-Fast-1B;Smart-3B" \
    -e WEBUI_AUTH=False \
    -v open-webui-multi:/app/backend/data \
    ghcr.io/open-webui/open-webui:main
```

Now the dropdown will show:
- **Lightning-Fast-1B** (instead of meta-llama/Llama-3.2-1B-Instruct)
- **Smart-3B** (instead of meta-llama/Llama-3.2-3B-Instruct)

---

## 🔄 Alternative: Using LoRA Adapters

If you want to serve a base model with multiple fine-tuned variants:

```bash
VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=N300 \
    python examples/server_example_tt.py \
    --model "meta-llama/Llama-3.1-8B-Instruct" \
    --enable-lora \
    --lora-modules \
        sql-lora=/path/to/sql-lora \
        code-lora=/path/to/code-lora \
        chat-lora=/path/to/chat-lora \
    --host 0.0.0.0 \
    --port 8000
```

Then in your API requests:
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "sql-lora",
        "messages": [{"role": "user", "content": "SELECT * FROM users"}]
    }'
```

---

## 📚 References

- **vLLM Open WebUI Docs**: https://docs.vllm.ai/en/latest/deployment/frameworks/open-webui.html
- **vLLM Multi-LoRA**: https://docs.vllm.ai/en/latest/features/lora.html
- **Open WebUI GitHub**: https://github.com/open-webui/open-webui
- **Open WebUI Environment Variables**: https://docs.openwebui.com/getting-started/env-configuration

---

## 🎉 Summary

**To run multiple models:**

1. ✅ Start multiple vLLM servers on different ports
2. ✅ Use `OPENAI_API_BASE_URLS` (plural) with semicolon-separated URLs
3. ✅ Access Open WebUI and switch between models
4. ✅ No authentication required with `WEBUI_AUTH=False`

**Quick commands:**
```bash
# Start servers
./start_multimodel_servers.sh

# Start Open WebUI
docker-compose -f docker-compose-openwebui-multimodel.yaml up -d

# Stop servers
./stop_multimodel_servers.sh

# Stop Open WebUI
docker stop open-webui-vllm-multi
```

Enjoy your multi-model setup! 🚀


