# Quick Start: Open WebUI with vLLM (No Authentication)

This guide helps you quickly set up Open WebUI with your vLLM server **without requiring user login**.

Based on [official vLLM documentation](https://docs.vllm.ai/en/latest/deployment/frameworks/open-webui.html) with no-authentication modifications.

## Prerequisites

1. ✅ Docker installed
2. ✅ vLLM server running at `http://localhost:8000`

## Step 1: Ensure vLLM Server is Running

Your vLLM server should be started with `--host` and `--port` flags:

```bash
# Using the provided script
./run_server_llama_1b.sh
```

Or manually:
```bash
MESH_DEVICE=N150 python examples/server_example_tt.py \
    --model "meta-llama/Llama-3.2-1B" \
    --host 0.0.0.0 \
    --port 8000
```

Verify it's running:
```bash
curl http://localhost:8000/v1/models
```

## Step 2: Start Open WebUI (3 Options)

### Option A: Quick Start Script (Recommended)
```bash
./run_open_webui.sh
```

### Option B: Docker Command
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

### Option C: Docker Compose
```bash
docker-compose -f docker-compose-openwebui.yaml up -d
```

## Step 3: Access Open WebUI

Open your browser and navigate to:
```
http://localhost:3000
```

🎉 **No login required!** Start chatting immediately.

## Key Environment Variables

| Variable | Value | Description |
|----------|-------|-------------|
| `OPENAI_API_BASE_URL` | `http://host.docker.internal:8000/v1` | vLLM server endpoint |
| `WEBUI_AUTH` | `False` | **Disables authentication** |
| `ENABLE_SIGNUP` | `False` | Disables signup form |

## Using the Interface

1. At the top of the page, select your model (e.g., `meta-llama/Llama-3.2-1B`)
2. Type your message in the chat box
3. Press Enter or click Send
4. Enjoy your local AI assistant!

## Docker Commands

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

# Remove volume (deletes all data)
docker volume rm open-webui
```

## Troubleshooting

### Issue: Can't see my model

**Solution:** Check if vLLM server is running:
```bash
curl http://localhost:8000/v1/models
```

### Issue: Connection refused

**On Linux**, Docker's `host.docker.internal` might not work. Try:
```bash
# Find your host IP
ip addr show docker0 | grep inet

# Use the IP (usually 172.17.0.1)
docker run -d \
    --name open-webui-vllm \
    -p 3000:8080 \
    -e OPENAI_API_BASE_URL=http://172.17.0.1:8000/v1 \
    -e WEBUI_AUTH=False \
    -v open-webui:/app/backend/data \
    ghcr.io/open-webui/open-webui:main
```

Or use `--network host`:
```bash
docker run -d \
    --name open-webui-vllm \
    --network host \
    -e OPENAI_API_BASE_URL=http://localhost:8000/v1 \
    -e WEBUI_AUTH=False \
    -v open-webui:/app/backend/data \
    ghcr.io/open-webui/open-webui:main

# Access at http://localhost:8080 (note: port 8080, not 3000)
```

### Issue: Still seeing login page

Check if `WEBUI_AUTH` is set correctly:
```bash
docker exec open-webui-vllm env | grep WEBUI_AUTH
```

Should show: `WEBUI_AUTH=False`

If not, recreate the container:
```bash
docker stop open-webui-vllm
docker rm open-webui-vllm
./run_open_webui.sh
```

## Testing the Setup

Once both vLLM server and Open WebUI are running:

1. **Check vLLM server:** `curl http://localhost:8000/v1/models`
2. **Access Open WebUI:** Open browser to `http://localhost:3000`
3. **Send a test message:** Type "Hello!" and press Enter
4. **Verify response:** You should get a response from the AI model

## Advanced: Using with Different Models

To use a different model:

1. Stop the current vLLM server
2. Start with your desired model:
   ```bash
   MESH_DEVICE=N150 python examples/server_example_tt.py \
       --model "meta-llama/Llama-3.2-3B"
   ```
3. The model will automatically appear in Open WebUI's dropdown

## Resources

- 📚 [Official vLLM + Open WebUI Docs](https://docs.vllm.ai/en/latest/deployment/frameworks/open-webui.html)
- 🐙 [Open WebUI GitHub](https://github.com/open-webui/open-webui)
- 🐙 [vLLM GitHub](https://github.com/vllm-project/vllm)
- 📖 [Open WebUI Documentation](https://docs.openwebui.com/)

Enjoy your private, local AI chat interface! 🚀

