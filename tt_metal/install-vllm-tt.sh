export VLLM_TARGET_DEVICE="tt"
uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match