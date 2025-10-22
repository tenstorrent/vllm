# Troubleshooting vLLM Inference Errors

## Error: "Device string must not be empty"

This error occurs when `VLLM_TARGET_DEVICE` is not set properly.

### Solution: Set up the environment correctly

```bash
# Make sure you're in the correct directory and source the setup scripts
cd /workspace/tt-vllm
export vllm_dir=$(pwd)
export TT_METAL_HOME=/workspace/tt-metal-apv

# Source tt-metal environment
source $TT_METAL_HOME/env_vars_setup.sh

# Source vllm environment (sets VLLM_TARGET_DEVICE=tt)
source $vllm_dir/tt_metal/setup-metal.sh

# Verify environment
echo "VLLM_TARGET_DEVICE=$VLLM_TARGET_DEVICE"
echo "TT_METAL_HOME=$TT_METAL_HOME"
echo "PYTHONPATH=$PYTHONPATH"

# Now run inference
MESH_DEVICE=N150 python examples/offline_inference_tt.py --model "meta-llama/Llama-3.2-1B"
```

## Inside Docker Container

If you're running inside a Docker container, you need to ensure the environment is set up **every time** you enter the container:

```bash
# Add to your container startup or run manually
export TT_METAL_HOME=/workspace/tt-metal-apv
export vllm_dir=/workspace/tt-vllm

source $TT_METAL_HOME/env_vars_setup.sh
source $vllm_dir/tt_metal/setup-metal.sh

# Verify
env | grep -E "(VLLM_TARGET_DEVICE|TT_METAL_HOME|PYTHONPATH)"
```

## Error: "Failed to import from vllm._C"

This indicates vLLM wasn't built properly for the TensTorrent target device.

### Solution: Rebuild vLLM

```bash
cd /workspace/tt-vllm

# Set up environment first
export TT_METAL_HOME=/workspace/tt-metal-apv
source $TT_METAL_HOME/env_vars_setup.sh
source tt_metal/setup-metal.sh

# Install/rebuild vLLM
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

## Testing with Supported Models

According to the tt-metal README, supported models include:
- Llama-3.1-8B
- Llama-3.2-1B (recommended for testing)
- Llama-3.2-3B
- Llama-3.1-70B
- Qwen2.5-7B
- Qwen2.5-72B

GLM-4.5-Air may not be supported on TensTorrent hardware yet.

### Test with a known working model:

```bash
# Source environment
cd /workspace/tt-vllm
export vllm_dir=$(pwd)
export TT_METAL_HOME=/workspace/tt-metal-apv
source $TT_METAL_HOME/env_vars_setup.sh
source $vllm_dir/tt_metal/setup-metal.sh

# Run with Llama-3.2-1B (small and well-tested)
MESH_DEVICE=N150 python examples/offline_inference_tt.py \
    --model "meta-llama/Llama-3.2-1B"
```

## Common Issues & Solutions

### Issue 1: PyTorch version mismatch
```
ERROR! Intel® Extension for PyTorch* needs to work with PyTorch 2.6.*, but PyTorch 2.7.1+cpu is found
```

**Solution:** This warning can usually be ignored for TensTorrent devices, or reinstall with correct PyTorch version.

### Issue 2: No platform detected
```
INFO: No platform detected, vLLM is running on UnspecifiedPlatform
```

**Solution:** Make sure `VLLM_TARGET_DEVICE=tt` is set before running.

### Issue 3: Running in Docker without proper setup

If you're inside the Docker container, create an alias or script:

```bash
# Inside container, create setup alias
cat >> ~/.bashrc << 'EOF'
alias setup_vllm='
export TT_METAL_HOME=/workspace/tt-metal-apv
export vllm_dir=/workspace/tt-vllm
source $TT_METAL_HOME/env_vars_setup.sh
source $vllm_dir/tt_metal/setup-metal.sh
echo "✅ vLLM environment ready"
'
EOF

source ~/.bashrc

# Now just run:
setup_vllm
```

## Quick Debug Commands

```bash
# Check if vLLM is installed
python -c "import vllm; print(vllm.__version__)"

# Check environment variables
env | grep -E "(VLLM|TT_METAL|MESH_DEVICE|PYTHONPATH)"

# Check if TensTorrent models are registered
python -c "from vllm import ModelRegistry; print(ModelRegistry.get_supported_archs())"

# Test basic import
python -c "from examples.offline_inference_tt import register_tt_models; register_tt_models(); print('✅ TT models registered')"
```

## Full Working Example

```bash
#!/bin/bash
# Save this as run_inference_fixed.sh

set -e

# Setup environment
export TT_METAL_HOME=/workspace/tt-metal-apv
export vllm_dir=/workspace/tt-vllm

echo "Setting up environment..."
cd $TT_METAL_HOME
source env_vars_setup.sh

cd $vllm_dir
source tt_metal/setup-metal.sh

echo "Environment variables:"
echo "  VLLM_TARGET_DEVICE=$VLLM_TARGET_DEVICE"
echo "  TT_METAL_HOME=$TT_METAL_HOME"
echo "  PYTHONPATH=$PYTHONPATH"

echo ""
echo "Running inference..."
MESH_DEVICE=N150 python examples/offline_inference_tt.py \
    --model "meta-llama/Llama-3.2-1B"

echo "Done!"
```

Make it executable and run:
```bash
chmod +x run_inference_fixed.sh
./run_inference_fixed.sh
```

