# Connecting Open WebUI to vLLM Server in Docker

This guide covers different scenarios for connecting Open WebUI to a vLLM server that's running in a Docker container.

## Scenario 1: vLLM in Docker, Open WebUI Outside Docker (Recommended)

This is the **simplest approach** - your vLLM server is in a container exposing port 8000, and you run Open WebUI directly on your host machine.

### Requirements
- vLLM container is running and exposing port 8000: `-p 8000:8000`

### Setup

```bash
./run_open_webui_external.sh
```

Or manually:

```bash
# Install Open WebUI
pip3 install open-webui

# Set environment variables
export OPENAI_API_BASE_URL=http://localhost:8000/v1
export WEBUI_AUTH=False
export ENABLE_SIGNUP=False

# Run Open WebUI
open-webui serve --port 3000
```

### How it works
- vLLM container exposes port 8000 to host → accessible at `localhost:8000`
- Open WebUI runs on host → connects to `localhost:8000`
- Access Open WebUI at: http://localhost:3000

---

## Scenario 2: Both vLLM and Open WebUI in Separate Docker Containers

When both are in Docker containers, they need to communicate through Docker networking.

### Option A: Automatic Script (Recommended)

```bash
./run_open_webui_docker_network.sh
```

This script will:
1. Ask for your vLLM container name
2. Detect the Docker network it's using
3. Start Open WebUI on the same network
4. Configure proper connectivity

### Option B: Manual Setup

```bash
# 1. Find your vLLM container name
docker ps

# 2. Find its network
docker inspect <vllm-container-name> --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}'

# 3. Start Open WebUI on the same network
docker run -d \
    --name open-webui-vllm \
    --network <network-name> \
    -p 3000:8080 \
    -e OPENAI_API_BASE_URL=http://<vllm-container-name>:8000/v1 \
    -e WEBUI_AUTH=False \
    -e ENABLE_SIGNUP=False \
    -v open-webui:/app/backend/data \
    ghcr.io/open-webui/open-webui:main
```

**Key point:** Use the container name (not `localhost`) in the API URL since they're on the same Docker network.

### Option C: Docker Compose

If you want to manage both containers together:

```bash
docker-compose -f docker-compose-both-containers.yaml up -d
```

You'll need to modify the `docker-compose-both-containers.yaml` file to match your vLLM container configuration.

---

## Scenario 3: vLLM in Docker with Host Network Mode

If your vLLM container uses `--network host`:

```bash
# vLLM is using host network
docker run --network host ... vllm-server

# Open WebUI can also use host network
docker run -d \
    --name open-webui-vllm \
    --network host \
    -e OPENAI_API_BASE_URL=http://localhost:8000/v1 \
    -e WEBUI_AUTH=False \
    -v open-webui:/app/backend/data \
    ghcr.io/open-webui/open-webui:main

# Access at http://localhost:8080 (default port when using host network)
```

---

## Troubleshooting

### Issue: "Connection refused" or "Cannot connect to vLLM"

**Check 1: Is vLLM container running?**
```bash
docker ps | grep vllm
```

**Check 2: Is port 8000 exposed?**
```bash
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep vllm
# Should show something like: 0.0.0.0:8000->8000/tcp
```

**Check 3: Can you reach vLLM from the host?**
```bash
curl http://localhost:8000/v1/models
```

**Check 4: Are containers on the same network? (for Docker-to-Docker)**
```bash
# Get vLLM network
docker inspect <vllm-container> --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}'

# Get Open WebUI network
docker inspect open-webui-vllm --format='{{range $k, $v := .NetworkSettings.Networks}}{{$k}}{{end}}'

# They should match!
```

**Check 5: Test connectivity between containers**
```bash
# From Open WebUI container, try to reach vLLM
docker exec open-webui-vllm curl http://<vllm-container-name>:8000/v1/models
```

### Issue: vLLM container doesn't expose port 8000

If your vLLM container wasn't started with `-p 8000:8000`, you have two options:

**Option 1: Recreate the vLLM container with port mapping**
```bash
docker stop <vllm-container>
docker rm <vllm-container>
# Restart with -p 8000:8000 ...
```

**Option 2: Use Docker networking (containers on same network)**
Use the script: `./run_open_webui_docker_network.sh`

### Issue: Open WebUI shows "No models available"

This usually means Open WebUI can't reach the vLLM API.

**Debug steps:**
```bash
# Check Open WebUI logs
docker logs open-webui-vllm

# Check what URL Open WebUI is trying to use
docker exec open-webui-vllm env | grep OPENAI_API_BASE_URL

# Test the connection from inside Open WebUI container
docker exec open-webui-vllm curl http://<api-url>/v1/models
```

---

## Quick Reference

| Setup | vLLM Access URL | Notes |
|-------|----------------|-------|
| **Open WebUI outside Docker** | `http://localhost:8000/v1` | Simplest, vLLM must expose port 8000 |
| **Both in Docker, same network** | `http://<vllm-container-name>:8000/v1` | Use container name, not localhost |
| **Both with host network** | `http://localhost:8000/v1` | Access Open WebUI at port 8080 |
| **Docker Compose** | `http://vllm-server:8000/v1` | Use service name from compose file |

---

## Recommended Setup

For most users, we recommend:

1. **Keep vLLM in Docker** (for isolation and easy management)
2. **Run Open WebUI outside Docker** (simpler, easier debugging)
3. **Ensure vLLM exposes port 8000**: `-p 8000:8000`
4. **Use the script**: `./run_open_webui_external.sh`

This gives you the best of both worlds - containerized vLLM with easy-to-access Open WebUI.

---

## Need Help?

- Check logs: `docker logs <container-name>`
- List networks: `docker network ls`
- Inspect network: `docker network inspect <network-name>`
- Test connectivity: `docker exec <container> curl <url>`

Good luck! 🚀

