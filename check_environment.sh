#!/bin/bash

# Script to check if vLLM environment is properly configured

echo "==== vLLM Environment Checker ===="
echo ""

# Check current directory
echo "📁 Current directory: $(pwd)"
echo ""

# Check critical environment variables
echo "🔍 Environment Variables:"
echo ""

check_var() {
    local var_name=$1
    local var_value="${!var_name}"
    if [ -z "$var_value" ]; then
        echo "  ❌ $var_name: NOT SET"
        return 1
    else
        echo "  ✅ $var_name: $var_value"
        return 0
    fi
}

SUCCESS=0

check_var "VLLM_TARGET_DEVICE" || SUCCESS=1
check_var "TT_METAL_HOME" || SUCCESS=1
check_var "PYTHONPATH" || SUCCESS=1
check_var "PYTHON_ENV_DIR" || SUCCESS=1

echo ""
echo "🐍 Python Information:"
echo "  Python: $(which python)"
echo "  Version: $(python --version 2>&1)"

echo ""
echo "📦 Package Checks:"

# Check if vLLM is importable
if python -c "import vllm" 2>/dev/null; then
    VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "unknown")
    echo "  ✅ vLLM installed (version: $VERSION)"
else
    echo "  ❌ vLLM not installed or not importable"
    SUCCESS=1
fi

# Check if torch is importable
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo "  ✅ PyTorch installed (version: $TORCH_VERSION)"
else
    echo "  ❌ PyTorch not installed"
    SUCCESS=1
fi

# Check if TT models can be registered
echo ""
echo "🔧 TensTorrent Setup:"
if python -c "import sys; sys.path.insert(0, '/workspace/tt-vllm/examples'); from offline_inference_tt import register_tt_models; register_tt_models()" 2>/dev/null; then
    echo "  ✅ TensTorrent models can be registered"
else
    echo "  ❌ Failed to register TensTorrent models"
    echo "     This might be okay if PYTHONPATH isn't set yet"
fi

echo ""
echo "📂 Directory Checks:"

if [ -d "/workspace/tt-metal-apv" ]; then
    echo "  ✅ tt-metal-apv directory exists"
else
    echo "  ❌ tt-metal-apv directory not found"
    SUCCESS=1
fi

if [ -d "/workspace/tt-vllm" ]; then
    echo "  ✅ tt-vllm directory exists"
else
    echo "  ❌ tt-vllm directory not found"
    SUCCESS=1
fi

if [ -f "/workspace/tt-metal-apv/env_vars_setup.sh" ]; then
    echo "  ✅ env_vars_setup.sh found"
else
    echo "  ❌ env_vars_setup.sh not found"
    SUCCESS=1
fi

if [ -f "/workspace/tt-vllm/tt_metal/setup-metal.sh" ]; then
    echo "  ✅ setup-metal.sh found"
else
    echo "  ❌ setup-metal.sh not found"
    SUCCESS=1
fi

echo ""
echo "==== Summary ===="
echo ""

if [ $SUCCESS -eq 0 ]; then
    echo "✅ Environment looks good!"
    echo ""
    echo "You can run:"
    echo "  ./run_inference_fixed.sh"
    echo "  or"
    echo "  MESH_DEVICE=N150 python examples/offline_inference_tt.py --model meta-llama/Llama-3.2-1B"
else
    echo "❌ Environment has issues!"
    echo ""
    echo "To fix, run:"
    echo ""
    echo "  export TT_METAL_HOME=/workspace/tt-metal-apv"
    echo "  export vllm_dir=/workspace/tt-vllm"
    echo "  source \$TT_METAL_HOME/env_vars_setup.sh"
    echo "  source \$vllm_dir/tt_metal/setup-metal.sh"
    echo ""
    echo "Then check again with: ./check_environment.sh"
fi

echo ""

