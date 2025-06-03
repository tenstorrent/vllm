
## vLLM and tt-metal Branches
Git-checkout the following branches in each repo separately:
- vLLM branch: [dev](https://github.com/tenstorrent/vllm/tree/dev) (last verified commit: [f1cb301](https://github.com/tenstorrent/vllm/tree/f1cb30141b662650353d40a330366c50b83b5a98))
- tt-metal branch: [main](https://github.com/tenstorrent/tt-metal) (last verified commit: [v0.57.0-rc44](https://github.com/tenstorrent/tt-metal/tree/v0.57.0-rc44))

## Environment Creation

**To create the vLLM+tt-metal environment (first time):**
1. Set tt-metal environment variables (see INSTALLING.md in tt-metal repo)
2. From the main vLLM directory, run:
    ```sh
    export vllm_dir=$(pwd)
    source $vllm_dir/tt_metal/setup-metal.sh
    ```
3. From the main tt-metal directory, build and create the environment as usual:
    ```sh
    ./build_metal.sh && ./create_venv.sh
    source $PYTHON_ENV_DIR/bin/activate
    ```
4. Install vLLM:
    ```sh
    pip3 install --upgrade pip
    cd $vllm_dir && pip install -e .
    ```

**To activate the vLLM+tt-metal environment (after the first time):**
1. Ensure `$vllm_dir` contains the path to vLLM and run:
    ```sh
    source $vllm_dir/tt_metal/setup-metal.sh && source $PYTHON_ENV_DIR/bin/activate
    ```

## Accessing the Meta-Llama Hugging Face Models

To run Meta-Llama-3.1/3.2, it is required to have access to the model on Hugging Face. To gain access:
1. Request access on Hugging Face:
    - Llama-3.1: [https://huggingface.co/meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)
    - Llama-3.2: [https://huggingface.co/meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
    - Llama-3.2-Vision: [https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
2. Once you have received access, create and copy your access token from the settings tab on Hugging Face.
3. Run this code in python and paste your access token:
    ```python
    from huggingface_hub import login
    login()
    ```

## Preparing the tt-metal models

1. Ensure that `$PYTHONPATH` contains the path to tt-metal (should already have been done when installing tt-metal)
2. For the desired model, follow the setup instructions (if any) for the corresponding tt-metal demo. E.g. For Llama-3.1/3.2 and Qwen-2.5, follow the [demo instructions](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers) for preparing the weights and environment variables, and install any extra requirements (e.g. `pip install -r models/tt_transformers/requirements.txt`).

## Running the offline inference example

### Llama-3.1/3.2 (1B, 3B, 8B, 70B) and Qwen-2.5 (7B, 72B) Text Models

To generate tokens (Llama70B) for sample prompts (with batch size 32):
```sh
MESH_DEVICE=T3K WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py
```

To measure performance (Llama70B) for a single batch of 32 prompts (with the default prompt length of 128 tokens):
```sh
MESH_DEVICE=T3K WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py --measure_perf
```

**Note 1**: Custom TT options can be set using `--override_tt_config` with a json string, e.g. `--override_tt_config '{"sample_on_device_mode": "all"}'`, however these shouldn't be used unless the model supports them (most currently do not). Supported parameters are:
- `sample_on_device_mode`: ["all", "decode_only"]
- `trace_region_size`: [default: 23887872]
- `worker_l1_size`
- `fabric_config`: ["DISABLED", "FABRIC_1D", "FABRIC_2D", "CUSTOM"]
- `dispatch_core_axis`: ["row", "col"]
- `data_parallel`: [default: 1]

**Note 2 (Llama70B)**: To run Llama70B on Galaxy, set `MESH_DEVICE=TG` and do not set `WH_ARCH_YAML=...`.

**Note 3 (Llama70B)**: By default, this will run the newer tt-metal implementation of Llama70B from the [tt_transformers demo](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers). To run other implementations use the `TT_LLAMA_TEXT_VER` environment variable:
- `"llama3_subdevices"` for the Llama TG implementation
- `"llama2_70b"` for the old Llama implementation

**Note 4 (Other Models)**: By default, the inference example will run with Llama-3.1-70B. To run with other Llama models, or Qwen-2.5, ensure that the apprioriate environment variables are set as per the [demo instructions](https://github.com/tenstorrent/tt-metal/tree/main/models/tt_transformers), then set `MESH_DEVICE=<device>` (valid options for `<device>` are `N150`, `N300`, `T3K`, or `TG`) and one of the following:
- Llama-3.1-8B: `--model "meta-llama/Llama-3.1-8B"`
- Llama-3.2-1B: `--model "meta-llama/Llama-3.2-1B"`
- Llama-3.2-3B: `--model "meta-llama/Llama-3.2-3B"`
- Qwen-2.5-7B: `--model "Qwen/Qwen2.5-7B"` (currently only supported on N300)
- Qwen-2.5-72B: `--model "Qwen/Qwen2.5-72B"` (currently only supported on T3K)
- DeepSeek-R1-Distill-Llama-70B: `--model "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"`

Command line to run Llama70B on TG is:
```sh
MESH_DEVICE=TG LLAMA_DIR=<path to weights> TT_LLAMA_TEXT_VER="llama3_subdevices" python examples/offline_inference_tt.py --model "meta-llama/Llama-3.1-70B-Instruct" --override_tt_config '{"dispatch_core_axis": "col", "sample_on_device_mode": "all", "fabric_config": "FABRIC_1D", "worker_l1_size": 1344544, "trace_region_size": 62000000}'
```

### Llama-3.2 (11B and 90B) Vision models

To generate tokens for sample prompts:
```sh
MESH_DEVICE=N300 WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py --model "meta-llama/Llama-3.2-11B-Vision-Instruct" --multi_modal --max_seqs_in_batch 16 --num_repeat_prompts 8
```

To measure performance for a single batch (with the default prompt length of 128 tokens):
```sh
MESH_DEVICE=N300 WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py --model "meta-llama/Llama-3.2-11B-Vision-Instruct" --measure_perf --multi_modal --max_seqs_in_batch 16
```

**Note**: To run on T3000, set `MESH_DEVICE=T3K` and `--max_seqs_in_batch 32`.

**Note**: Running `90B` model, set `MESH_DEVICE=T3K` and `--max_seqs_in_batch 4`:

```sh
MESH_DEVICE=T3K WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/offline_inference_tt.py --model "meta-llama/Llama-3.2-90B-Vision-Instruct" --multi_modal --max_seqs_in_batch 4
```

## Running the server example

To start up the server:
```sh
VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=T3K WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml python examples/server_example_tt.py
```

**Note**: By default, the server will run with Llama-3.1-70B-Instruct. To run with other models, set `MESH_DEVICE` and `--model` as described in [Running the offline inference example](#running-the-offline-inference-example).

**Note**: Custom TT options can be set using `--override_tt_config` as described in [Running the offline inference example](#running-the-offline-inference-example).

To send a request to the server:
```sh
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{ "model": "meta-llama/Llama-3.1-70B-Instruct", "prompt": "San Francisco is a", "max_tokens": 32, "temperature": 1, "top_p": 0.9, "top_k": 10 }'
```

### Llama-3.2 (11B and 90B) Vision models

First, start the server following the instructions above with the correct model through `--model`. 

Second, generate a prompt json, e.g.,
```python
import base64
import json

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Path to your image
image_path = "pasta.jpeg"

# Getting the base64 string
base64_image = encode_image(image_path)

payload = {
    "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is for dinner?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    "max_tokens": 128,
    "temperature": 1,
    "top_p": 0.9,
    "top_k": 10
}

# Save to a JSON file
with open("server-instruct-mm-prompt.json", "w") as json_file:
    json.dump(payload, json_file, indent=4)
```

Finally, send a request to the server:
```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" --data-binary @server-instruct-mm-prompt.json
```
