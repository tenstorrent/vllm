# Open WebUI Setup with tt-inference-server

Quick links to documentation and setup scripts for running Open WebUI with Tenstorrent inference server.

---

## 🚀 Quick Start (5 minutes)

### Step 1: Start tt-inference-server
```bash
cd /home/ttuser/aperezvicente/tt-inference-server

# Start server with Llama 1B (or your preferred model)
python run.py \
    --workflow docker_server \
    --model_id Llama-3.2-1B-Instruct \
    --impl_id tt-transformers \
    --device_mesh N150 \
    --service_port 8000
```

### Step 2: Start Open WebUI
```bash
cd /home/ttuser/aperezvicente/tt-vllm

# Run the setup script (handles JWT auth automatically)
./run_open_webui.sh
```

### Step 3: Open Browser
```
http://localhost:3000
```

✅ **Done!** No login required, model should appear in dropdown.

---

## 📚 Documentation

| Document | Description | When to Use |
|----------|-------------|-------------|
| **[COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)** | 📖 **START HERE** - Full setup from scratch | New setup, production deployment |
| **[TT_INFERENCE_SERVER_AUTH_GUIDE.md](TT_INFERENCE_SERVER_AUTH_GUIDE.md)** | 🔐 JWT authentication details | Troubleshooting auth issues |
| **[MULTIMODEL_GUIDE.md](MULTIMODEL_GUIDE.md)** | 🔄 Running multiple models | Want to use 2+ models simultaneously |
| **[QUICKSTART_LLAMA1B_OPENWEBUI.md](QUICKSTART_LLAMA1B_OPENWEBUI.md)** | ⚡ Quick Llama 1B setup | Fast single-model setup |

---

## 🛠️ Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| **`run_open_webui.sh`** | Start Open WebUI with auth | `./run_open_webui.sh` |
| **`test_openwebui_connection.sh`** | Verify everything works | `./test_openwebui_connection.sh` |
| **`start_multimodel_servers.sh`** | Start multiple vLLM servers | `./start_multimodel_servers.sh` |
| **`stop_multimodel_servers.sh`** | Stop all vLLM servers | `./stop_multimodel_servers.sh` |

---

## 🔍 Troubleshooting

### Models not showing?

1. **Check tt-inference-server is running:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Test authentication:**
   ```bash
   JWT_TOKEN=$(python3 -c "import jwt; import json; payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); token = jwt.encode(payload, 'tenstorrent', algorithm='HS256'); print(token)")
   
   curl http://localhost:8000/v1/models \
       -H "Authorization: Bearer $JWT_TOKEN"
   ```

3. **Check Open WebUI logs:**
   ```bash
   docker logs open-webui-vllm --tail 50
   ```

4. **Restart with fresh data:**
   ```bash
   docker stop open-webui-vllm
   docker rm open-webui-vllm
   docker volume rm open-webui
   ./run_open_webui.sh
   ```

5. **Run diagnostic script:**
   ```bash
   ./test_openwebui_connection.sh
   ```

### 401 Unauthorized errors?

- **Issue**: JWT token mismatch
- **Solution**: See [TT_INFERENCE_SERVER_AUTH_GUIDE.md](TT_INFERENCE_SERVER_AUTH_GUIDE.md)

### Can't reach server from container?

- **Issue**: Network connectivity
- **Solution**: Try using Docker bridge IP (172.17.0.1) instead of host.docker.internal

---

## 📦 What Gets Installed

### Docker Containers

```bash
# View running containers
docker ps

# Expected output:
# - open-webui-vllm (port 3000)
# - tt-inference-server-* (port 8000)
```

### Docker Volumes

```bash
# View volumes
docker volume ls | grep open-webui

# Backup data
docker run --rm -v open-webui:/data -v $(pwd):/backup \
    ubuntu tar czf /backup/open-webui-backup.tar.gz /data
```

---

## 🎯 Common Use Cases

### Single Model (Llama 1B)
→ See [QUICKSTART_LLAMA1B_OPENWEBUI.md](QUICKSTART_LLAMA1B_OPENWEBUI.md)

### Multiple Models (1B, 3B, 8B)
→ See [MULTIMODEL_GUIDE.md](MULTIMODEL_GUIDE.md)

### Production Deployment
→ See [COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md#production-deployment)

### Custom Authentication
→ See [TT_INFERENCE_SERVER_AUTH_GUIDE.md](TT_INFERENCE_SERVER_AUTH_GUIDE.md#custom-jwt-secret)

---

## 🔗 Architecture

```
┌────────────────┐
│  Browser       │  http://localhost:3000
│  (User)        │
└───────┬────────┘
        │
        ▼
┌────────────────────┐
│  Open WebUI        │  Auto-login, No auth
│  (Docker)          │  Port 3000
└───────┬────────────┘
        │ JWT Token
        ▼
┌────────────────────────┐
│  tt-inference-server   │  JWT Auth Required
│  (Docker)              │  Port 8000
└───────┬────────────────┘
        │
        ▼
┌────────────────────────┐
│  TT Hardware           │  N150/N300/T3K/TG
│  (Accelerator)         │
└────────────────────────┘
```

---

## 📋 Requirements

- ✅ Docker (20.10+)
- ✅ Python 3.8+ with PyJWT
- ✅ Tenstorrent device (N150/N300/T3K/TG)
- ✅ HF Token with Meta-Llama access
- ✅ 16GB+ RAM (32GB+ recommended)
- ✅ 50GB+ free disk space

---

## 🆘 Support

### Check Logs
```bash
# tt-inference-server
docker logs tt-inference-server-* --tail 100

# Open WebUI
docker logs open-webui-vllm --tail 100
```

### Verify Configuration
```bash
# Check JWT is set
docker inspect open-webui-vllm | jq '.[0].Config.Env[] | select(contains("OPENAI_API_KEYS"))'

# Check server JWT secret
docker inspect tt-inference-server-* | jq '.[0].Config.Env[] | select(contains("JWT_SECRET"))'
```

### Clean Slate
```bash
# Stop everything
docker stop open-webui-vllm tt-inference-server-* 2>/dev/null

# Remove containers
docker rm open-webui-vllm tt-inference-server-* 2>/dev/null

# Remove data (optional - deletes chat history)
docker volume rm open-webui

# Start fresh
./run_open_webui.sh
```

---

## 🎓 Learn More

- **vLLM Official Docs**: https://docs.vllm.ai/
- **Open WebUI Docs**: https://docs.openwebui.com/
- **tt-metal GitHub**: https://github.com/tenstorrent/tt-metal
- **Hugging Face Models**: https://huggingface.co/meta-llama

---

## ✅ Verification Checklist

After setup, verify:

- [ ] `docker ps` shows both containers running
- [ ] `curl http://localhost:8000/health` returns OK
- [ ] `curl http://localhost:3000` returns webpage
- [ ] Browser opens http://localhost:3000 without login
- [ ] Model appears in dropdown
- [ ] Chat message gets response
- [ ] No errors in `docker logs`

---

## 📝 Quick Reference

### Ports
- **3000**: Open WebUI (web interface)
- **8000**: tt-inference-server (API)

### Default JWT
- **Secret**: `tenstorrent`
- **Token**: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0ZWFtX2lkIjoidGVuc3RvcnJlbnQiLCJ0b2tlbl9pZCI6ImRlYnVnLXRlc3QifQ.SZcxmsrkk8uxsa1-u7Rzia4C5-yZh0CGHBvkmDJyoh8`

### Useful Commands
```bash
# Start
./run_open_webui.sh

# Test
./test_openwebui_connection.sh

# Logs
docker logs -f open-webui-vllm

# Stop
docker stop open-webui-vllm

# Remove
docker rm open-webui-vllm

# Restart
docker restart open-webui-vllm
```

---

**Happy chatting! 🎉**

*For detailed information, see [COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md)*


