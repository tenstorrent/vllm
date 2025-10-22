# Quick Fix: Expose Docker Port to Localhost

## Your Situation
- vLLM server running in Docker container
- Port 8000 NOT accessible from localhost
- Need to expose it to connect Open WebUI

## Quick Solutions

### Solution 1: Restart Container with Port Mapping ⭐

```bash
# 1. Check your container name
docker ps

# 2. Stop and backup
docker stop <container-name>
docker commit <container-name> vllm-backup

# 3. Remove old container
docker rm <container-name>

# 4. Start with port exposed
docker run -d \
    --name <container-name> \
    -p 8000:8000 \
    vllm-backup
```

### Solution 2: Use Host Network (Easier)

```bash
# 1. Stop and backup
docker stop <container-name>
docker commit <container-name> vllm-backup
docker rm <container-name>

# 2. Start with host network
docker run -d \
    --name <container-name> \
    --network host \
    vllm-backup
```

### Solution 3: Use Docker-to-Docker Networking

Don't expose port - connect containers directly:

```bash
./run_open_webui_docker_network.sh
```

## Use Our Helper Script

```bash
./check_vllm_container.sh
```

This script will:
- Find your vLLM container
- Check if port is exposed
- Give you exact commands to fix it

## Verify It Works

```bash
curl http://localhost:8000/v1/models
```

Should return your model list!

## Then Run Open WebUI

Once port is exposed:

```bash
./run_open_webui.sh
```

Access at: http://localhost:3000
