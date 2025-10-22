# Complete Setup Guide: tt-inference-server with Open WebUI

**A comprehensive guide for deploying Tenstorrent inference server with Open WebUI interface**

Version: 1.0  
Last Updated: October 22, 2025

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Architecture](#architecture)
4. [Part 1: tt-inference-server Setup](#part-1-tt-inference-server-setup)
5. [Part 2: Open WebUI Setup](#part-2-open-webui-setup)
6. [Part 3: Authentication Configuration](#part-3-authentication-configuration)
7. [Part 4: Verification & Testing](#part-4-verification--testing)
8. [Part 5: Multi-Model Setup](#part-5-multi-model-setup-optional)
9. [Troubleshooting](#troubleshooting)
10. [Production Deployment](#production-deployment)
11. [Reference](#reference)

---

## Overview

This guide walks you through setting up a complete LLM inference stack using:
- **tt-inference-server**: Tenstorrent's optimized vLLM server for TT hardware
- **Open WebUI**: Modern, user-friendly chat interface
- **JWT Authentication**: Secure API access
- **Docker**: Containerized deployment

### What You'll Build

```
┌─────────────────┐
│   User Browser  │
│  (localhost:3000)│
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────────────┐
│    Open WebUI           │
│    (Docker Container)   │
│    - Chat Interface     │
│    - No Auth Required   │
└────────┬────────────────┘
         │ HTTP + JWT Token
         ▼
┌──────────────────────────────┐
│  tt-inference-server         │
│  (Docker Container)          │
│  - vLLM Server               │
│  - Llama 3.2 1B/3B/8B        │
│  - JWT Authentication        │
│  - TT Hardware Accelerated   │
└──────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  Tenstorrent Hardware        │
│  (N150/N300/T3K/TG)          │
└──────────────────────────────┘
```

---

## Prerequisites

### Required Software

- ✅ **Docker** (20.10+)
  ```bash
  docker --version  # Should show Docker version 20.10.0 or higher
  ```

- ✅ **Python 3.8+** with PyJWT library
  ```bash
  python3 --version
  pip3 install PyJWT  # For JWT token generation
  ```

- ✅ **curl** and **jq** (for testing)
  ```bash
  sudo apt-get install curl jq
  ```

### Required Hardware

- ✅ **Tenstorrent Device**: N150, N300, T3K (QuietBox), or TG (Galaxy)
- ✅ **RAM**: Minimum 16GB (32GB+ recommended for larger models)
- ✅ **Disk**: 50GB+ free space for models and Docker images

### Required Access

- ✅ **Hugging Face Account**: For accessing Meta-Llama models
- ✅ **HF Token**: Created at https://huggingface.co/settings/tokens
- ✅ **Model Access**: Request access to meta-llama models at:
  - https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
  - https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct

### Directory Structure

```bash
/home/ttuser/aperezvicente/
├── tt-metal/              # TT Metal repository
├── tt-metal-apv/          # TT Metal APV
├── tt-vllm/               # vLLM with TT support (this repo)
└── tt-inference-server/   # Inference server repository
```

---

## Architecture

### Component Overview

| Component | Purpose | Port | Technology |
|-----------|---------|------|------------|
| **tt-inference-server** | LLM inference backend | 8000 | vLLM + TT Metal |
| **Open WebUI** | Chat interface | 3000 | FastAPI + Svelte |
| **JWT Auth** | API security | N/A | HS256 tokens |
| **Docker Network** | Container communication | N/A | bridge + host-gateway |

### Authentication Flow

```
1. Open WebUI Container Starts
   ├─ Reads OPENAI_API_KEYS env var
   └─ JWT Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

2. User Opens Browser → http://localhost:3000
   ├─ Auto-login (WEBUI_AUTH=False)
   └─ Models dropdown populated

3. User Sends Chat Message
   ├─ Open WebUI → tt-inference-server
   ├─ Headers: Authorization: Bearer <JWT_TOKEN>
   ├─ tt-inference-server validates JWT
   └─ Response streamed back to user
```

---

## Part 1: tt-inference-server Setup

### Step 1.1: Check Existing tt-inference-server

First, check if you already have a tt-inference-server running:

```bash
# Check for running containers
docker ps | grep tt-inference-server

# If found, check the configuration
docker inspect <container-name> | grep -E "JWT_SECRET|MODEL|PORT"
```

**If already running**, skip to [Part 2](#part-2-open-webui-setup).

### Step 1.2: Set Up Environment (if starting fresh)

```bash
# Navigate to tt-inference-server directory
cd /home/ttuser/aperezvicente/tt-inference-server

# Set Hugging Face token
export HF_TOKEN="your_hf_token_here"

# Verify token is set
echo $HF_TOKEN
```

### Step 1.3: Start tt-inference-server

Using the workflow system (recommended):

```bash
cd /home/ttuser/aperezvicente/tt-inference-server

# Run with Llama 3.2 1B Instruct
python run.py \
    --workflow docker_server \
    --model_id Llama-3.2-1B-Instruct \
    --impl_id tt-transformers \
    --device_mesh N150 \
    --service_port 8000
```

**Alternative models:**
- Llama 3.2 3B: `--model_id Llama-3.2-3B-Instruct --device_mesh N300`
- Llama 3.1 8B: `--model_id Llama-3.1-8B-Instruct --device_mesh N300`

### Step 1.4: Configure JWT Authentication

The tt-inference-server uses JWT authentication by default with:
- **JWT_SECRET**: `tenstorrent` (default)
- **Payload**: `{"team_id": "tenstorrent", "token_id": "debug-test"}`

To verify:

```bash
# Check if JWT_SECRET is set
docker ps --format '{{.Names}}' | grep tt-inference-server | xargs docker inspect | grep JWT_SECRET
```

**Expected output:**
```
"JWT_SECRET=tenstorrent"
```

### Step 1.5: Verify tt-inference-server

```bash
# Generate JWT token
export JWT_TOKEN=$(python3 -c "import jwt; import json; payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); token = jwt.encode(payload, 'tenstorrent', algorithm='HS256'); print(token)")

# Test health endpoint (no auth required)
curl http://localhost:8000/health

# Test models endpoint (requires auth)
curl -s http://localhost:8000/v1/models \
    -H "Authorization: Bearer $JWT_TOKEN" | jq .

# Test completion (requires auth)
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $JWT_TOKEN" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [{"role": "user", "content": "Hello! How are you?"}],
        "max_tokens": 50,
        "temperature": 0.7
    }' | jq .
```

**Expected responses:**
- Health: `"OK"` or similar
- Models: List containing `meta-llama/Llama-3.2-1B-Instruct`
- Completion: Chat response with tokens

---

## Part 2: Open WebUI Setup

### Step 2.1: Pull Open WebUI Docker Image

```bash
docker pull ghcr.io/open-webui/open-webui:main
```

### Step 2.2: Generate JWT Token

The JWT token is required for Open WebUI to authenticate with tt-inference-server:

```bash
# Generate token
JWT_TOKEN=$(python3 -c "import jwt; import json; payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); token = jwt.encode(payload, 'tenstorrent', algorithm='HS256'); print(token)")

# Save it for reference
echo "JWT Token: $JWT_TOKEN"
```

**Default Token** (if using default JWT_SECRET='tenstorrent'):
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0ZWFtX2lkIjoidGVuc3RvcnJlbnQiLCJ0b2tlbl9pZCI6ImRlYnVnLXRlc3QifQ.SZcxmsrkk8uxsa1-u7Rzia4C5-yZh0CGHBvkmDJyoh8
```

### Step 2.3: Start Open WebUI Container

```bash
# Remove any existing instance
docker stop open-webui-vllm 2>/dev/null || true
docker rm open-webui-vllm 2>/dev/null || true

# Start Open WebUI with proper configuration
docker run -d \
    --name open-webui-vllm \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -e OPENAI_API_BASE_URLS="http://host.docker.internal:8000/v1" \
    -e OPENAI_API_KEYS="$JWT_TOKEN" \
    -e WEBUI_AUTH=False \
    -e ENABLE_SIGNUP=False \
    -v open-webui:/app/backend/data \
    --restart always \
    ghcr.io/open-webui/open-webui:main
```

**Important Configuration:**
- `OPENAI_API_BASE_URLS` (plural): For multiple servers
- `OPENAI_API_KEYS` (plural): Corresponding JWT tokens
- `WEBUI_AUTH=False`: Auto-login without password
- `ENABLE_SIGNUP=False`: Disable user registration
- `host.docker.internal`: Allows container to reach host ports

### Step 2.4: Wait for Startup

```bash
# Wait for container to be healthy
echo "Waiting for Open WebUI to start..."
sleep 15

# Check status
docker ps | grep open-webui-vllm

# View logs
docker logs open-webui-vllm --tail 20
```

### Step 2.5: Using the Convenience Script

Alternatively, use the provided script:

```bash
cd /home/ttuser/aperezvicente/tt-vllm

# Make script executable (if not already)
chmod +x run_open_webui.sh

# Run it
./run_open_webui.sh
```

The script automatically:
- ✅ Generates JWT token
- ✅ Stops any existing containers
- ✅ Configures all environment variables
- ✅ Starts Open WebUI

---

## Part 3: Authentication Configuration

### Understanding JWT Authentication

The tt-inference-server uses JWT (JSON Web Tokens) for API authentication:

```python
# JWT Generation (Python)
import jwt
import json

payload = {"team_id": "tenstorrent", "token_id": "debug-test"}
secret = "tenstorrent"  # From JWT_SECRET env var
token = jwt.encode(payload, secret, algorithm="HS256")
```

### JWT Components

| Component | Value | Description |
|-----------|-------|-------------|
| **Secret** | `tenstorrent` | Signing key from JWT_SECRET |
| **Algorithm** | HS256 | HMAC SHA-256 |
| **Payload** | `{"team_id": "tenstorrent", "token_id": "debug-test"}` | Token claims |

### Where JWT is Used

1. **tt-inference-server startup** (`run_vllm_api_server.py`):
   ```python
   if os.getenv("JWT_SECRET"):
       encoded_jwt = get_encoded_api_key(jwt_secret)
       os.environ["VLLM_API_KEY"] = encoded_jwt
   ```

2. **Every API request**:
   ```bash
   curl http://localhost:8000/v1/models \
       -H "Authorization: Bearer <JWT_TOKEN>"
   ```

3. **Open WebUI configuration**:
   ```bash
   -e OPENAI_API_KEYS="<JWT_TOKEN>"
   ```

### Custom JWT Secret (Optional)

To use a custom secret:

1. **Modify tt-inference-server**:
   ```bash
   # When starting the server, set:
   -e JWT_SECRET="your-custom-secret"
   ```

2. **Generate custom token**:
   ```bash
   python3 -c "import jwt; import json; \
   payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); \
   token = jwt.encode(payload, 'your-custom-secret', algorithm='HS256'); \
   print(token)"
   ```

3. **Update Open WebUI**:
   ```bash
   -e OPENAI_API_KEYS="<new-token>"
   ```

---

## Part 4: Verification & Testing

### Test 1: Container Health Check

```bash
# Check both containers are running
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | \
    grep -E "NAMES|open-webui|tt-inference"
```

**Expected output:**
```
NAMES                          STATUS                  PORTS
open-webui-vllm                Up 2 minutes (healthy)  0.0.0.0:3000->8080/tcp
tt-inference-server-xxxxx      Up 15 minutes           0.0.0.0:8000->8000/tcp
```

### Test 2: tt-inference-server API

```bash
# Generate token
JWT_TOKEN=$(python3 -c "import jwt; import json; payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); token = jwt.encode(payload, 'tenstorrent', algorithm='HS256'); print(token)")

# Test health
curl http://localhost:8000/health

# List models
curl -s http://localhost:8000/v1/models \
    -H "Authorization: Bearer $JWT_TOKEN" | jq '.data[].id'

# Test chat completion
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $JWT_TOKEN" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }' | jq '.choices[0].message.content'
```

### Test 3: Open WebUI Connection

```bash
# Test from inside Open WebUI container
docker exec open-webui-vllm curl -s \
    http://host.docker.internal:8000/v1/models \
    -H "Authorization: Bearer $JWT_TOKEN" | jq '.data[].id'
```

**Expected:** List of available models

### Test 4: Web Interface

1. **Open browser**: http://localhost:3000

2. **Verify auto-login**: You should be logged in automatically (no password prompt)

3. **Check model dropdown**: At the top of the page, click the model dropdown

4. **Expected models**:
   - `meta-llama/Llama-3.2-1B-Instruct` (or whichever model you started)

5. **Test chat**:
   - Type: "Hello! Can you help me?"
   - Press Enter
   - You should see a response within seconds

### Test 5: Automated Test Script

Use the provided test script:

```bash
cd /home/ttuser/aperezvicente/tt-vllm
chmod +x test_openwebui_connection.sh
./test_openwebui_connection.sh
```

---

## Part 5: Multi-Model Setup (Optional)

### Overview

You can run multiple vLLM servers on different ports and have Open WebUI access all of them:

```
Open WebUI (port 3000)
    ├─→ tt-inference-server-1 (port 8000) → Llama 1B
    ├─→ tt-inference-server-2 (port 8001) → Llama 3B
    └─→ tt-inference-server-3 (port 8002) → Llama 8B
```

### Step 5.1: Start Multiple Servers

**Terminal 1 - Llama 1B:**
```bash
python run.py \
    --workflow docker_server \
    --model_id Llama-3.2-1B-Instruct \
    --impl_id tt-transformers \
    --device_mesh N150 \
    --service_port 8000
```

**Terminal 2 - Llama 3B:**
```bash
python run.py \
    --workflow docker_server \
    --model_id Llama-3.2-3B-Instruct \
    --impl_id tt-transformers \
    --device_mesh N300 \
    --service_port 8001
```

### Step 5.2: Configure Open WebUI for Multiple Models

```bash
# Generate JWT token
JWT_TOKEN=$(python3 -c "import jwt; import json; payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); token = jwt.encode(payload, 'tenstorrent', algorithm='HS256'); print(token)")

# Stop existing Open WebUI
docker stop open-webui-vllm && docker rm open-webui-vllm

# Start with multiple endpoints
docker run -d \
    --name open-webui-vllm \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -e OPENAI_API_BASE_URLS="http://host.docker.internal:8000/v1;http://host.docker.internal:8001/v1" \
    -e OPENAI_API_KEYS="$JWT_TOKEN;$JWT_TOKEN" \
    -e WEBUI_AUTH=False \
    -e ENABLE_SIGNUP=False \
    -v open-webui:/app/backend/data \
    --restart always \
    ghcr.io/open-webui/open-webui:main
```

**Key points:**
- URLs and keys are semicolon-separated: `url1;url2;url3`
- Number of URLs must match number of keys
- All models will appear in the dropdown

### Step 5.3: Use the Multi-Model Script

```bash
cd /home/ttuser/aperezvicente/tt-vllm

# Interactive script to start multiple servers
./start_multimodel_servers.sh

# Then start Open WebUI
docker-compose -f docker-compose-openwebui-multimodel.yaml up -d
```

---

## Troubleshooting

### Issue 1: Models Not Showing in Open WebUI

**Symptoms:**
- Open WebUI loads but model dropdown is empty
- No models listed

**Diagnosis:**
```bash
# Check Open WebUI logs
docker logs open-webui-vllm | grep -i "error\|unauthorized"

# Test connection manually
JWT_TOKEN=$(python3 -c "import jwt; import json; payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); token = jwt.encode(payload, 'tenstorrent', algorithm='HS256'); print(token)")

docker exec open-webui-vllm curl -s \
    http://host.docker.internal:8000/v1/models \
    -H "Authorization: Bearer $JWT_TOKEN"
```

**Solutions:**

1. **Verify JWT token is set**:
   ```bash
   docker inspect open-webui-vllm | jq '.[0].Config.Env[] | select(contains("OPENAI_API_KEYS"))'
   ```

2. **Regenerate and restart**:
   ```bash
   cd /home/ttuser/aperezvicente/tt-vllm
   docker stop open-webui-vllm && docker rm open-webui-vllm
   ./run_open_webui.sh
   ```

3. **Clear Open WebUI data** (if settings are cached):
   ```bash
   docker volume rm open-webui
   # Then restart Open WebUI
   ```

4. **In browser**: Go to Settings → Connections → Verify/Refresh OpenAI connection

### Issue 2: 401 Unauthorized Error

**Symptoms:**
- Error in tt-inference-server logs: `"GET /v1/models HTTP/1.1" 401 Unauthorized`
- Open WebUI shows "Connection failed"

**Diagnosis:**
```bash
# Check if JWT_SECRET is set on server
docker ps --format '{{.Names}}' | grep tt-inference | \
    xargs docker inspect | grep JWT_SECRET

# Test with and without token
curl http://localhost:8000/v1/models  # Should fail
curl http://localhost:8000/v1/models -H "Authorization: Bearer $JWT_TOKEN"  # Should work
```

**Solutions:**

1. **Verify JWT_SECRET matches between server and client**:
   ```bash
   # Check server
   docker inspect tt-inference-server-* | grep JWT_SECRET
   
   # Regenerate token with same secret
   python3 -c "import jwt; import json; payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); token = jwt.encode(payload, 'tenstorrent', algorithm='HS256'); print(token)"
   ```

2. **Update Open WebUI with correct token**:
   ```bash
   # Stop and restart with new token
   docker stop open-webui-vllm && docker rm open-webui-vllm
   # Use new JWT_TOKEN
   ./run_open_webui.sh
   ```

### Issue 3: Container Can't Reach tt-inference-server

**Symptoms:**
- Error: "Cannot connect to host.docker.internal"
- Connection timeout

**Diagnosis:**
```bash
# Test from inside container
docker exec open-webui-vllm ping -c 3 host.docker.internal
docker exec open-webui-vllm curl -s http://host.docker.internal:8000/health
```

**Solutions:**

1. **On Linux**: Use Docker bridge IP instead:
   ```bash
   # Find Docker bridge IP
   ip addr show docker0 | grep "inet\b" | awk '{print $2}' | cut -d/ -f1
   # Usually 172.17.0.1
   
   # Restart Open WebUI with bridge IP
   docker stop open-webui-vllm && docker rm open-webui-vllm
   docker run -d \
       --name open-webui-vllm \
       -p 3000:8080 \
       -e OPENAI_API_BASE_URLS="http://172.17.0.1:8000/v1" \
       -e OPENAI_API_KEYS="$JWT_TOKEN" \
       ...
   ```

2. **Use Docker network** (advanced):
   ```bash
   # Create custom network
   docker network create tt-network
   
   # Connect both containers
   docker network connect tt-network tt-inference-server-*
   docker network connect tt-network open-webui-vllm
   
   # Use container name as hostname
   -e OPENAI_API_BASE_URLS="http://tt-inference-server-xxxxx:8000/v1"
   ```

### Issue 4: Port Already in Use

**Symptoms:**
- Error: "Bind for 0.0.0.0:3000 failed: port is already allocated"

**Solutions:**

1. **Find and stop conflicting container**:
   ```bash
   docker ps | grep 3000
   docker stop <container-name>
   ```

2. **Use different port**:
   ```bash
   docker run -d \
       --name open-webui-vllm \
       -p 8080:8080 \  # Use port 8080 instead
       ...
   # Access at http://localhost:8080
   ```

### Issue 5: Chat Template Error

**Symptoms:**
- Error: "chat template is no longer allowed"
- 400 Bad Request on chat completions

**Solution:**
- Use **Instruct** models, not base models:
  - ✅ `meta-llama/Llama-3.2-1B-Instruct`
  - ❌ `meta-llama/Llama-3.2-1B` (base model)

### Issue 6: Out of Memory

**Symptoms:**
- tt-inference-server crashes
- CUDA/TT out of memory errors

**Solutions:**

1. **Use smaller model**:
   - N150: Llama 1B only
   - N300: Llama 1B, 3B
   - T3K: Llama 1B, 3B, 8B, 70B

2. **Reduce batch size**:
   ```bash
   python run.py ... --max_num_seqs 8  # Default is 16
   ```

3. **Check device**:
   ```bash
   tt-smi  # View device status
   ```

### Issue 7: Slow Response Times

**Diagnosis:**
```bash
# Check server logs for performance metrics
docker logs tt-inference-server-* | grep "throughput\|tokens/s"
```

**Solutions:**

1. **Enable trace mode** (after warmup):
   - Wait for first few requests to complete
   - Subsequent requests should use traced operations

2. **Check device utilization**:
   ```bash
   tt-smi  # Should show active operations
   ```

3. **Verify model is loaded**:
   ```bash
   docker logs tt-inference-server-* | grep "loaded successfully"
   ```

---

## Production Deployment

### Security Best Practices

1. **Change JWT_SECRET**:
   ```bash
   # Generate secure random secret
   JWT_SECRET=$(openssl rand -base64 32)
   
   # Start server with custom secret
   -e JWT_SECRET="$JWT_SECRET"
   
   # Generate corresponding token
   python3 -c "import jwt; import json; \
   payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"prod\"}'); \
   token = jwt.encode(payload, '$JWT_SECRET', algorithm='HS256'); \
   print(token)"
   ```

2. **Enable Open WebUI Authentication**:
   ```bash
   # Remove WEBUI_AUTH=False
   # Users will need to create accounts
   docker run -d \
       --name open-webui-vllm \
       -p 3000:8080 \
       -e ENABLE_SIGNUP=True \  # Allow user registration
       ...
   ```

3. **Use HTTPS**:
   - Deploy behind nginx with SSL certificates
   - Use Let's Encrypt for free certificates

4. **Restrict Network Access**:
   ```bash
   # Bind only to localhost
   -p 127.0.0.1:3000:8080 \
   -p 127.0.0.1:8000:8000
   
   # Use with reverse proxy (nginx/traefik)
   ```

### Performance Tuning

1. **Batch Size**:
   ```bash
   --max_num_seqs 32  # Increase for higher throughput
   ```

2. **Context Length**:
   ```bash
   --max_model_len 8192  # Adjust based on use case
   ```

3. **KV Cache**:
   ```bash
   --gpu_memory_utilization 0.9  # Increase GPU memory for cache
   ```

### Monitoring

1. **Container Health**:
   ```bash
   # Add health checks
   docker run -d \
       --health-cmd="curl -f http://localhost:8000/health || exit 1" \
       --health-interval=30s \
       --health-timeout=10s \
       --health-retries=3 \
       ...
   ```

2. **Prometheus Metrics**:
   ```bash
   # vLLM exposes metrics at /metrics
   curl http://localhost:8000/metrics
   ```

3. **Logging**:
   ```bash
   # Centralize logs
   docker logs -f tt-inference-server-* >> /var/log/tt-inference.log
   docker logs -f open-webui-vllm >> /var/log/open-webui.log
   ```

### Backup & Recovery

1. **Backup Open WebUI data**:
   ```bash
   # Backup volume
   docker run --rm -v open-webui:/data -v $(pwd):/backup \
       ubuntu tar czf /backup/open-webui-backup.tar.gz /data
   ```

2. **Restore data**:
   ```bash
   # Restore volume
   docker run --rm -v open-webui:/data -v $(pwd):/backup \
       ubuntu tar xzf /backup/open-webui-backup.tar.gz -C /
   ```

### High Availability

1. **Load Balancing**:
   - Run multiple tt-inference-server instances
   - Use nginx/haproxy for load balancing
   - Configure Open WebUI with all endpoints

2. **Auto-Restart**:
   ```bash
   --restart unless-stopped  # Auto-restart on failure
   ```

3. **Health Checks**:
   ```bash
   # Monitor and auto-recover
   #!/bin/bash
   while true; do
       if ! curl -f http://localhost:8000/health; then
           docker restart tt-inference-server-*
       fi
       sleep 60
   done
   ```

---

## Reference

### Quick Command Reference

```bash
# Start tt-inference-server
python run.py --workflow docker_server --model_id Llama-3.2-1B-Instruct \
    --impl_id tt-transformers --device_mesh N150 --service_port 8000

# Generate JWT Token
JWT_TOKEN=$(python3 -c "import jwt; import json; payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); token = jwt.encode(payload, 'tenstorrent', algorithm='HS256'); print(token)")

# Start Open WebUI
./run_open_webui.sh

# Test Connection
curl -s http://localhost:8000/v1/models -H "Authorization: Bearer $JWT_TOKEN" | jq .

# View Logs
docker logs -f tt-inference-server-*
docker logs -f open-webui-vllm

# Stop Everything
docker stop open-webui-vllm tt-inference-server-*
docker rm open-webui-vllm tt-inference-server-*
```

### File Locations

| File | Path | Purpose |
|------|------|---------|
| Run script | `/home/ttuser/aperezvicente/tt-vllm/run_open_webui.sh` | Start Open WebUI |
| Test script | `/home/ttuser/aperezvicente/tt-vllm/test_openwebui_connection.sh` | Verify setup |
| Auth guide | `/home/ttuser/aperezvicente/tt-vllm/TT_INFERENCE_SERVER_AUTH_GUIDE.md` | JWT details |
| Multi-model guide | `/home/ttuser/aperezvicente/tt-vllm/MULTIMODEL_GUIDE.md` | Multiple models |
| This guide | `/home/ttuser/aperezvicente/tt-vllm/COMPLETE_SETUP_GUIDE.md` | Complete setup |

### Environment Variables Reference

#### tt-inference-server

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | - | Hugging Face access token |
| `JWT_SECRET` | `tenstorrent` | JWT signing secret |
| `SERVICE_PORT` | `8000` | API server port |
| `MESH_DEVICE` | - | TT device (N150/N300/T3K/TG) |
| `TT_METAL_HOME` | - | TT Metal installation path |
| `MODEL_WEIGHTS_PATH` | - | Model weights directory |

#### Open WebUI

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_BASE_URLS` | - | Semicolon-separated API URLs |
| `OPENAI_API_KEYS` | - | Semicolon-separated JWT tokens |
| `WEBUI_AUTH` | `True` | Enable/disable authentication |
| `ENABLE_SIGNUP` | `True` | Allow user registration |
| `DEFAULT_MODELS` | - | Default model selection |

### Port Reference

| Service | Port | Protocol | Access |
|---------|------|----------|--------|
| tt-inference-server | 8000 | HTTP | API endpoints |
| Open WebUI | 3000 | HTTP | Web interface |
| Ollama (optional) | 11434 | HTTP | Ollama API |

### Model Reference

| Model | Size | Device | Context | Use Case |
|-------|------|--------|---------|----------|
| Llama-3.2-1B-Instruct | 1B | N150 | 128K | Fast, lightweight |
| Llama-3.2-3B-Instruct | 3B | N300 | 128K | Balanced |
| Llama-3.1-8B-Instruct | 8B | N300/T3K | 128K | High quality |
| Llama-3.1-70B-Instruct | 70B | T3K/TG | 128K | Production |

### API Endpoints

#### tt-inference-server

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | No | Health check |
| `/v1/models` | GET | Yes | List models |
| `/v1/completions` | POST | Yes | Text completion |
| `/v1/chat/completions` | POST | Yes | Chat completion |
| `/v1/embeddings` | POST | Yes | Embeddings |
| `/metrics` | GET | No | Prometheus metrics |

#### Open WebUI

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | Browser | Web interface |
| `/api/models` | GET | API | List models |
| `/api/chat` | POST | API | Chat endpoint |

### Useful Docker Commands

```bash
# Container management
docker ps                                    # List running containers
docker ps -a                                 # List all containers
docker stop <container>                      # Stop container
docker start <container>                     # Start container
docker restart <container>                   # Restart container
docker rm <container>                        # Remove container
docker logs -f <container>                   # Follow logs
docker exec -it <container> bash             # Enter container

# Volume management
docker volume ls                             # List volumes
docker volume inspect <volume>               # Inspect volume
docker volume rm <volume>                    # Remove volume

# Network management
docker network ls                            # List networks
docker network inspect bridge                # Inspect network
docker network create <network>              # Create network
docker network connect <network> <container> # Connect container

# Image management
docker images                                # List images
docker pull <image>                          # Pull image
docker rmi <image>                           # Remove image

# System cleanup
docker system prune -a                       # Remove unused data
docker volume prune                          # Remove unused volumes
```

### Support Resources

- **tt-metal Documentation**: https://github.com/tenstorrent/tt-metal
- **vLLM Documentation**: https://docs.vllm.ai/
- **Open WebUI Documentation**: https://docs.openwebui.com/
- **Docker Documentation**: https://docs.docker.com/

### Troubleshooting Resources

1. **tt-inference-server logs**:
   ```bash
   docker logs tt-inference-server-* | grep -i "error\|warning\|exception"
   ```

2. **Open WebUI logs**:
   ```bash
   docker logs open-webui-vllm | grep -i "error\|warning\|exception"
   ```

3. **Network connectivity**:
   ```bash
   docker exec open-webui-vllm curl -v http://host.docker.internal:8000/health
   ```

4. **JWT validation**:
   ```bash
   python3 -c "import jwt; token='YOUR_TOKEN'; print(jwt.decode(token, 'tenstorrent', algorithms=['HS256']))"
   ```

---

## Summary Checklist

### Initial Setup ✅

- [ ] Docker installed and running
- [ ] Python 3.8+ with PyJWT installed
- [ ] Tenstorrent device available
- [ ] HF token configured
- [ ] Model access granted

### tt-inference-server ✅

- [ ] Server running on port 8000
- [ ] JWT_SECRET configured
- [ ] Model loaded successfully
- [ ] Health endpoint responding
- [ ] Auth working (401 without token, 200 with token)

### Open WebUI ✅

- [ ] Container running on port 3000
- [ ] JWT token configured in OPENAI_API_KEYS
- [ ] WEBUI_AUTH=False set
- [ ] Can reach tt-inference-server from container
- [ ] Browser accessible at http://localhost:3000

### Verification ✅

- [ ] Auto-login working (no password prompt)
- [ ] Models appearing in dropdown
- [ ] Chat messages getting responses
- [ ] Response latency acceptable
- [ ] No errors in logs

---

## Conclusion

You now have a complete LLM inference stack running with:
- ✅ **tt-inference-server**: Optimized TT hardware inference
- ✅ **Open WebUI**: Modern chat interface
- ✅ **JWT Authentication**: Secure API access
- ✅ **No login**: Instant access for development

**Next steps:**
- Customize the UI and behavior
- Add more models (see Multi-Model Setup)
- Deploy to production (see Production Deployment)
- Integrate with applications via API

**Access your setup:**
```
🌐 Web Interface: http://localhost:3000
🔌 API Endpoint: http://localhost:8000
📚 Documentation: This guide!
```

Enjoy your Tenstorrent-powered LLM inference! 🚀

---

**Document Version**: 1.0  
**Last Updated**: October 22, 2025  
**Author**: Generated for tt-vllm project  
**License**: Apache 2.0

