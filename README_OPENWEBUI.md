# Open WebUI Setup for vLLM (No Authentication)

This guide will help you set up Open WebUI to connect to your vLLM inference server without requiring user login.

## Prerequisites

1. Your vLLM server should be running at `http://localhost:8000`
2. Either Docker or Python/pip installed

## Method 1: Using Docker (Recommended)

### Quick Start
```bash
./run_open_webui.sh
```

### Or manually with Docker:
```bash
docker run -d \
    --name open-webui-vllm \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
    -e OPENAI_API_KEY=dummy \
    -e WEBUI_AUTH=False \
    -v open-webui:/app/backend/data \
    ghcr.io/open-webui/open-webui:main
```

### Or using Docker Compose:
```bash
docker-compose -f docker-compose-openwebui.yaml up -d
```

## Method 2: Using pip

```bash
# Install Open WebUI
pip install open-webui

# Set environment variables and run
export OPENAI_API_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=dummy
export WEBUI_AUTH=False
open-webui serve --port 3000
```

## Access the Web Interface

Once started, access Open WebUI at: **http://localhost:3000**

No login is required! You can start chatting immediately.

## Configuration

Key environment variables:
- `WEBUI_AUTH=False` - Disables authentication (no login required)
- `OPENAI_API_BASE_URL` - URL of your vLLM server
- `OPENAI_API_KEY` - Any dummy value works for local vLLM
- `ENABLE_SIGNUP=False` - Disables signup form
- `ENABLE_LOGIN_FORM=False` - Hides login form completely

## Managing the Docker Container

### View logs:
```bash
docker logs -f open-webui-vllm
```

### Stop the container:
```bash
docker stop open-webui-vllm
```

### Start the container:
```bash
docker start open-webui-vllm
```

### Remove the container:
```bash
docker rm open-webui-vllm
```

### Remove the volume (delete all data):
```bash
docker volume rm open-webui
```

## Troubleshooting

### Issue: Can't connect to vLLM server

**Check if vLLM server is running:**
```bash
curl http://localhost:8000/v1/models
```

**If using Docker on Linux, use:**
```bash
-e OPENAI_API_BASE_URL=http://172.17.0.1:8000/v1
```
Instead of `host.docker.internal`

### Issue: Port 3000 already in use

**Use a different port:**
```bash
docker run -d \
    --name open-webui-vllm \
    -p 8080:8080 \
    ...
```
Then access at http://localhost:8080

### Issue: Still seeing login page

Make sure `WEBUI_AUTH=False` is set. Check with:
```bash
docker exec open-webui-vllm env | grep WEBUI_AUTH
```

## Testing

Once Open WebUI is running, you can:
1. Open http://localhost:3000 in your browser
2. Start chatting immediately (no login needed)
3. Select your model from the dropdown (e.g., meta-llama/Llama-3.2-1B)
4. Start a conversation!

## Additional Features

Open WebUI supports:
- Multiple conversations
- Chat history
- Model switching
- Document upload and analysis
- Voice input/output
- Image generation (if configured)
- Prompt library
- And much more!

Enjoy your Open WebUI + vLLM setup! 🚀

