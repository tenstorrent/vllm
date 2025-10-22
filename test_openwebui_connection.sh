#!/bin/bash

echo "==== Testing Open WebUI Connection to tt-inference-server ===="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Generate JWT token
echo "Generating JWT token..."
JWT_TOKEN=$(python3 -c "import jwt; import json; payload = json.loads('{\"team_id\": \"tenstorrent\", \"token_id\":\"debug-test\"}'); token = jwt.encode(payload, 'tenstorrent', algorithm='HS256'); print(token)")
echo -e "${GREEN}✓ JWT token generated${NC}"
echo ""

# Test 1: Check if tt-inference-server is running
echo "Test 1: Checking tt-inference-server..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✓ tt-inference-server is running on port 8000${NC}"
else
    echo -e "${RED}✗ tt-inference-server is NOT running${NC}"
    echo "  Start it first!"
    exit 1
fi
echo ""

# Test 2: Check if we can access models with JWT
echo "Test 2: Fetching models from tt-inference-server..."
MODELS_RESPONSE=$(curl -s http://localhost:8000/v1/models -H "Authorization: Bearer $JWT_TOKEN")
if echo "$MODELS_RESPONSE" | grep -q "meta-llama"; then
    echo -e "${GREEN}✓ Successfully fetched models${NC}"
    echo "$MODELS_RESPONSE" | jq '.data[].id'
else
    echo -e "${RED}✗ Failed to fetch models${NC}"
    echo "Response: $MODELS_RESPONSE"
    exit 1
fi
echo ""

# Test 3: Check if Open WebUI is running
echo "Test 3: Checking Open WebUI..."
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Open WebUI is running on port 3000${NC}"
else
    echo -e "${RED}✗ Open WebUI is NOT running${NC}"
    echo "  Run: ./run_open_webui.sh"
    exit 1
fi
echo ""

# Test 4: Check if Open WebUI can reach tt-inference-server from inside container
echo "Test 4: Testing connection from Open WebUI container..."
CONTAINER_TEST=$(docker exec open-webui-vllm curl -s http://host.docker.internal:8000/v1/models -H "Authorization: Bearer $JWT_TOKEN" 2>&1)
if echo "$CONTAINER_TEST" | grep -q "meta-llama"; then
    echo -e "${GREEN}✓ Open WebUI container CAN reach tt-inference-server${NC}"
else
    echo -e "${RED}✗ Open WebUI container CANNOT reach tt-inference-server${NC}"
    echo "Response: $CONTAINER_TEST"
fi
echo ""

# Test 5: Check Open WebUI environment variables
echo "Test 5: Checking Open WebUI configuration..."
echo "OPENAI_API_BASE_URLS:"
docker inspect open-webui-vllm | jq -r '.[0].Config.Env[] | select(contains("OPENAI_API_BASE_URLS"))'
echo "OPENAI_API_KEYS:"
docker inspect open-webui-vllm | jq -r '.[0].Config.Env[] | select(contains("OPENAI_API_KEYS"))' | head -c 50
echo "..."
echo ""

echo "======================================"
echo -e "${GREEN}All tests passed!${NC}"
echo "======================================"
echo ""
echo "📍 Access Open WebUI at: ${YELLOW}http://localhost:3000${NC}"
echo ""
echo "If models still don't show:"
echo "1. Open http://localhost:3000 in your browser"
echo "2. You should be auto-logged in (no password needed)"
echo "3. Look for the model dropdown at the top"
echo "4. If empty, go to Settings → Connections → Verify OpenAI connection"
echo ""


