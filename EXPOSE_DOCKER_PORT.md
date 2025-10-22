# How to Expose Docker Container Port to Localhost

## Problem
Your vLLM server is running in a Docker container, but the port is not accessible from localhost.

## Solution: Expose Port When Starting Container

### Option 1: Start New Container with Port Mapping (Recommended)

When you start your vLLM container, use the `-p` flag to map the container port to localhost:

```bash
docker run -p 8000:8000 [other options] your-vllm-image

# Full example:
docker run -d \
    --name vllm-server \
    -p 8000:8000 \
    -v /workspace/tt-metal-apv:/tt-metal \
    -v /workspace/tt-vllm:/vllm \
    your-vllm-image \
    python examples/server_example_tt.py --model meta-llama/Llama-3.2-1B --host 0.0.0.0 --port 8000
```

**Format:** `-p <host-port>:<container-port>`
- `8000:8000` means container port 8000 → localhost port 8000
- `3000:8000` means container port 8000 → localhost port 3000

### Option 2: If Container is Already Running

You **cannot** add port mapping to a running container. You must:

1. **Stop the container:**
   ```bash
   docker stop <container-name>
   ```

2. **Commit the container to an image (to save your changes):**
   ```bash
   docker commit <container-name> vllm-server-saved
   ```

3. **Remove the old container:**
   ```bash
   docker rm <container-name>
   ```

4. **Start new container with port mapping:**
   ```bash
   docker run -d \
       --name vllm-server \
       -p 8000:8000 \
       [other original options] \
       vllm-server-saved \
       [original command]
   ```

### Option 3: Use Host Network Mode

Instead of port mapping, use host networking:

```bash
docker run --network host [other options] your-vllm-image
```

**With host network:**
- Container uses host's network directly
- No need for `-p` flag
- Server runs on `localhost:8000` automatically
- Easier but less isolated

**Example:**
```bash
docker run -d \
    --name vllm-server \
    --network host \
    -v /workspace/tt-metal-apv:/tt-metal \
    -v /workspace/tt-vllm:/vllm \
    your-vllm-image \
    python examples/server_example_tt.py --model meta-llama/Llama-3.2-1B --host 0.0.0.0 --port 8000
```

## Verification

After starting the container with port mapping, verify:

```bash
# Check port mapping
docker ps --format "table {{.Names}}\t{{.Ports}}"

# Should show something like:
# NAMES          PORTS
# vllm-server    0.0.0.0:8000->8000/tcp

# Test the connection
curl http://localhost:8000/v1/models
```

## Finding Your Current Container Setup

To see how your container was started:

```bash
# List running containers with ports
docker ps

# Get full command that started the container
docker inspect <container-name> --format='{{.Config.Cmd}}'

# Get all port mappings
docker port <container-name>

# Get complete container config
docker inspect <container-name>
```

## Common Scenarios

### Scenario 1: Container Started Without Port Mapping

```bash
# Current (no port mapping):
docker run -d --name vllm-server my-image

# Fix: Stop, remove, restart with port mapping
docker stop vllm-server
docker commit vllm-server vllm-server-backup
docker rm vllm-server
docker run -d --name vllm-server -p 8000:8000 vllm-server-backup
```

### Scenario 2: Container Started with docker-compose

Edit your `docker-compose.yml`:

```yaml
services:
  vllm-server:
    image: your-vllm-image
    ports:
      - "8000:8000"  # Add this line
    # ... other config
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

### Scenario 3: Already Inside Container

If you're currently running commands inside the container, you need to:

1. **Exit the container:**
   ```bash
   exit
   ```

2. **From the host, check current container:**
   ```bash
   docker ps
   ```

3. **Restart with port mapping** (see Option 2 above)

## Quick Reference Commands

```bash
# Start container with port 8000 exposed
docker run -p 8000:8000 ...

# Start with host network (no port mapping needed)
docker run --network host ...

# Check if port is exposed
docker port <container-name>

# Check what ports container is listening on (from inside)
docker exec <container-name> netstat -tulpn

# Test connection from host
curl http://localhost:8000/v1/models
```

## Troubleshooting

### Issue: Port already in use

```bash
# Find what's using port 8000
sudo lsof -i :8000
# or
sudo netstat -tulpn | grep 8000

# Stop the process or use different port
docker run -p 8001:8000 ...  # Map to different host port
```

### Issue: Connection refused

**Check 1: Is the server running inside the container?**
```bash
docker exec <container-name> curl http://localhost:8000/v1/models
```

**Check 2: Is server bound to 0.0.0.0 (not 127.0.0.1)?**
```bash
# Server must be started with --host 0.0.0.0
python server.py --host 0.0.0.0 --port 8000
```

**Check 3: Is port actually mapped?**
```bash
docker port <container-name>
# Should show: 8000/tcp -> 0.0.0.0:8000
```

### Issue: Can't recreate container

If you can't recreate the container, use Docker Compose or save a script:

```bash
# Save your docker run command to a script
cat > start_vllm.sh << 'EOF'
#!/bin/bash
docker run -d \
    --name vllm-server \
    -p 8000:8000 \
    -v /workspace/tt-metal-apv:/tt-metal \
    -v /workspace/tt-vllm:/vllm \
    your-image \
    python examples/server_example_tt.py --model meta-llama/Llama-3.2-1B --host 0.0.0.0
EOF

chmod +x start_vllm.sh
```

Now you can easily restart with: `./start_vllm.sh`

---

## Summary

**If starting fresh:** Use `-p 8000:8000` when running docker
**If already running:** Stop, commit, remove, restart with port mapping
**Easiest option:** Use `--network host` (less isolated but simpler)

Once the port is exposed, you can connect Open WebUI to `http://localhost:8000`! 🚀

