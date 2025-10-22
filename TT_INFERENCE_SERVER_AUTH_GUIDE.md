# tt-inference-server Authentication Guide

## Problem

When trying to connect Open WebUI to the tt-inference-server, you may see that **models are not appearing** in the dropdown. This is because the tt-inference-server requires **JWT authentication**.

## Symptoms

```bash
curl http://localhost:8000/v1/models
# Returns: {"error": "Unauthorized"}
```

In the tt-inference-server logs, you'll see:
```
INFO:     172.17.0.x:xxxxx - "GET /v1/models HTTP/1.1" 401 Unauthorized
```

## Root Cause

The tt-inference-server uses JWT (JSON Web Token) authentication when `JWT_SECRET` environment variable is set. By default in the container, `JWT_SECRET=tenstorrent`.

## Solution

### 1. Generate the JWT Token

The JWT token is generated using:
- **Secret**: `tenstorrent` (from `JWT_SECRET` env var)
- **Payload**: `{"team_id": "tenstorrent", "token_id": "debug-test"}`
- **Algorithm**: HS256

```bash
python3 -c "import jwt; import json; payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); token = jwt.encode(payload, 'tenstorrent', algorithm='HS256'); print(token)"
```

**Output:**
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0ZWFtX2lkIjoidGVuc3RvcnJlbnQiLCJ0b2tlbl9pZCI6ImRlYnVnLXRlc3QifQ.SZcxmsrkk8uxsa1-u7Rzia4C5-yZh0CGHBvkmDJyoh8
```

### 2. Test with curl

```bash
export JWT_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0ZWFtX2lkIjoidGVuc3RvcnJlbnQiLCJ0b2tlbl9pZCI6ImRlYnVnLXRlc3QifQ.SZcxmsrkk8uxsa1-u7Rzia4C5-yZh0CGHBvkmDJyoh8"

# List models
curl http://localhost:8000/v1/models -H "Authorization: Bearer $JWT_TOKEN" | jq .

# Test completion
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $JWT_TOKEN" \
    -d '{
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }'
```

### 3. Configure Open WebUI

When starting Open WebUI, you must provide the JWT token via `OPENAI_API_KEY`:

```bash
docker run -d \
    --name open-webui-vllm \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
    -e OPENAI_API_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0ZWFtX2lkIjoidGVuc3RvcnJlbnQiLCJ0b2tlbl9pZCI6ImRlYnVnLXRlc3QifQ.SZcxmsrkk8uxsa1-u7Rzia4C5-yZh0CGHBvkmDJyoh8" \
    -e WEBUI_AUTH=False \
    -e ENABLE_SIGNUP=False \
    -v open-webui:/app/backend/data \
    --restart always \
    ghcr.io/open-webui/open-webui:main
```

### 4. Using the Updated Script

The `run_open_webui.sh` script now automatically generates the JWT token:

```bash
cd /home/ttuser/aperezvicente/tt-vllm
./run_open_webui.sh
```

## How It Works

### tt-inference-server Code Flow

1. **Environment Check** (`run_vllm_api_server.py:212-226`):
   ```python
   jwt_secret = os.getenv("JWT_SECRET")
   if jwt_secret:
       logger.info("JWT_SECRET is set: HTTP requests to vLLM API require bearer token")
       encoded_api_key = get_encoded_api_key(jwt_secret)
       if encoded_api_key is not None:
           os.environ["VLLM_API_KEY"] = encoded_api_key
   else:
       logger.warning("JWT_SECRET is not set: HTTP requests will not require authorization")
   ```

2. **Token Generation** (`utils/vllm_run_utils.py:55-60`):
   ```python
   def get_encoded_api_key(jwt_secret):
       if jwt_secret is None:
           return None
       json_payload = json.loads('{"team_id": "tenstorrent", "token_id":"debug-test"}')
       encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
       return encoded_jwt
   ```

3. **vLLM Server Authentication**:
   The generated JWT token is set as `VLLM_API_KEY` environment variable, which vLLM's OpenAI API server uses to validate incoming requests.

## Checking Server Configuration

### Check if JWT_SECRET is Set

```bash
# Method 1: Inspect container
docker inspect tt-inference-server-* | jq '.[0].Config.Env[] | select(contains("JWT"))'

# Method 2: Check logs at startup
docker logs tt-inference-server-* | grep JWT
```

**Expected output:**
```
JWT_SECRET is set: HTTP requests to vLLM API require bearer token in 'Authorization' header
```

### Check Available Models

```bash
# Without auth (will fail if JWT_SECRET is set)
curl http://localhost:8000/v1/models

# With auth
JWT_TOKEN=$(python3 -c "import jwt; import json; payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); token = jwt.encode(payload, 'tenstorrent', algorithm='HS256'); print(token)")
curl http://localhost:8000/v1/models -H "Authorization: Bearer $JWT_TOKEN"
```

## Troubleshooting

### Issue: Still getting "Unauthorized"

**Check 1**: Verify JWT_SECRET value
```bash
docker inspect tt-inference-server-* | jq '.[0].Config.Env[] | select(contains("JWT"))'
```

**Check 2**: Verify your token is correct
```bash
# Decode your token to check payload
python3 -c "import jwt; token='YOUR_TOKEN_HERE'; print(jwt.decode(token, 'tenstorrent', algorithms=['HS256']))"
```

**Check 3**: Check Open WebUI is using the key
```bash
docker inspect open-webui-vllm | jq '.[0].Config.Env[] | select(contains("API_KEY"))'
```

### Issue: Models not showing in Open WebUI

**Check 1**: Verify Open WebUI can reach the server
```bash
docker logs open-webui-vllm | grep -i "error\|unauthorized"
```

**Check 2**: Restart Open WebUI with correct API key
```bash
docker stop open-webui-vllm
docker rm open-webui-vllm
./run_open_webui.sh
```

### Issue: "Invalid JWT format"

Make sure you're using the full token including all three parts (header.payload.signature):
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0ZWFtX2lkIjoidGVuc3RvcnJlbnQiLCJ0b2tlbl9pZCI6ImRlYnVnLXRlc3QifQ.SZcxmsrkk8uxsa1-u7Rzia4C5-yZh0CGHBvkmDJyoh8
```

## Disabling Authentication (Not Recommended for Production)

If you want to disable authentication entirely, you can restart the tt-inference-server **without** the `JWT_SECRET` environment variable:

```bash
# Stop current server
docker stop tt-inference-server-*

# Start new server without JWT_SECRET
# (You would need to modify the docker run command or workflow config)
```

**Note:** This is not recommended for production deployments.

## Custom JWT Secret

To use a different JWT secret:

1. **Set custom JWT_SECRET** when starting tt-inference-server
2. **Generate new token** with your custom secret:
   ```bash
   python3 -c "import jwt; import json; payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); token = jwt.encode(payload, 'YOUR_CUSTOM_SECRET', algorithm='HS256'); print(token)"
   ```
3. **Update Open WebUI** with the new token

## Summary

✅ **Problem**: tt-inference-server requires JWT authentication  
✅ **Default JWT_SECRET**: `tenstorrent`  
✅ **Default Token**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...`  
✅ **Solution**: Set `OPENAI_API_KEY` in Open WebUI with the JWT token  
✅ **Quick Start**: Use `./run_open_webui.sh` (now includes auth)

## References

- **JWT Generation Code**: `/home/ttuser/aperezvicente/tt-inference-server/utils/vllm_run_utils.py:55-60`
- **Authentication Check**: `/home/ttuser/aperezvicente/tt-inference-server/vllm-tt-metal-llama3/src/run_vllm_api_server.py:212-226`
- **vLLM API Server**: vLLM's built-in OpenAI-compatible API server with authentication support


