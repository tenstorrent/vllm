#!/bin/bash

# Script to check your vLLM Docker container and help expose the port

echo "==== vLLM Docker Container Checker ===="
echo ""

# Find containers that might be vLLM
echo "🔍 Looking for running containers..."
CONTAINERS=$(docker ps --format "{{.Names}}")

if [ -z "$CONTAINERS" ]; then
    echo "❌ No running containers found!"
    echo ""
    echo "Start your vLLM container with port mapping:"
    echo "  docker run -p 8000:8000 [options] your-image"
    exit 1
fi

echo "📦 Running containers:"
docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
echo ""

# Ask user which container is vLLM
echo "Which container is running vLLM?"
echo "(Enter the container name from above)"
read -p "Container name: " CONTAINER_NAME

if [ -z "$CONTAINER_NAME" ]; then
    echo "❌ No container name provided"
    exit 1
fi

# Check if container exists
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ Container '$CONTAINER_NAME' not found or not running"
    exit 1
fi

echo ""
echo "✅ Checking container: $CONTAINER_NAME"
echo ""

# Check port mappings
echo "📡 Port mappings:"
PORTS=$(docker port $CONTAINER_NAME 2>/dev/null)
if [ -z "$PORTS" ]; then
    echo "❌ NO PORTS EXPOSED!"
    echo ""
    echo "This container is not exposing any ports to localhost."
    echo ""
    echo "🔧 To fix this, you need to recreate the container:"
    echo ""
    echo "1. Get the current container's command:"
    docker inspect $CONTAINER_NAME --format='{{.Config.Cmd}}' | head -5
    echo ""
    echo "2. Stop and backup the container:"
    echo "   docker stop $CONTAINER_NAME"
    echo "   docker commit $CONTAINER_NAME ${CONTAINER_NAME}-backup"
    echo "   docker rm $CONTAINER_NAME"
    echo ""
    echo "3. Start with port mapping:"
    echo "   docker run -d --name $CONTAINER_NAME -p 8000:8000 ${CONTAINER_NAME}-backup"
    echo ""
    echo "OR use host network mode:"
    echo "   docker run -d --name $CONTAINER_NAME --network host ${CONTAINER_NAME}-backup"
    echo ""
    
    read -p "Would you like me to show the full container config? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Full container configuration:"
        docker inspect $CONTAINER_NAME
    fi
    
else
    echo "✅ Ports exposed:"
    echo "$PORTS"
    echo ""
    
    # Check if port 8000 is mapped
    if echo "$PORTS" | grep -q "8000"; then
        echo "✅ Port 8000 is exposed!"
        
        # Try to connect
        echo ""
        echo "🔍 Testing connection to localhost:8000..."
        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "✅ SUCCESS! vLLM server is accessible at http://localhost:8000"
            echo ""
            echo "You can now run Open WebUI:"
            echo "  ./run_open_webui_external.sh"
            echo ""
            echo "Available models:"
            curl -s http://localhost:8000/v1/models | python3 -m json.tool 2>/dev/null || echo "  (Could not parse models list)"
        else
            echo "⚠️  Port is exposed but server is not responding"
            echo ""
            echo "Possible issues:"
            echo "1. Server inside container is not running"
            echo "2. Server is bound to 127.0.0.1 instead of 0.0.0.0"
            echo ""
            echo "Check server inside container:"
            echo "  docker exec $CONTAINER_NAME curl http://localhost:8000/v1/models"
        fi
    else
        echo "⚠️  Port 8000 is NOT exposed to localhost"
        echo ""
        echo "Available ports: $PORTS"
        echo ""
        echo "You need to recreate the container with -p 8000:8000"
        echo "See instructions above."
    fi
fi

echo ""
echo "🔍 Container network mode:"
NETWORK_MODE=$(docker inspect $CONTAINER_NAME --format='{{.HostConfig.NetworkMode}}')
echo "  $NETWORK_MODE"

if [ "$NETWORK_MODE" = "host" ]; then
    echo ""
    echo "✅ Container is using host network mode"
    echo "   Server should be accessible at http://localhost:8000"
    echo ""
    echo "Testing connection..."
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "✅ SUCCESS! Server is accessible"
    else
        echo "❌ Server not responding. Check if it's running inside the container."
        echo "   docker exec $CONTAINER_NAME ps aux | grep python"
    fi
fi

echo ""
echo "==== Summary ===="
echo ""
echo "Container: $CONTAINER_NAME"
echo "Network: $NETWORK_MODE"
echo "Ports: ${PORTS:-None}"
echo ""
echo "Next steps:"
echo "  - If port 8000 is accessible: ./run_open_webui_external.sh"
echo "  - If port not exposed: Recreate container with -p 8000:8000"
echo "  - Alternative: Use Docker-to-Docker networking: ./run_open_webui_docker_network.sh"
echo ""

