VLLM_TARGET_DEVICE=empty uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install -e "plugins/vllm-tt-plugin[runtime]" --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
