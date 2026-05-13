# vLLM TT Plugin

Tenstorrent backend plugin for vLLM.

`vllm-tt-plugin` integrates Tenstorrent hardware into vLLM using the standard
plugin mechanism. Install it alongside vLLM and, when `ttnn` is importable,
TT hardware is automatically available as a vLLM platform.

The plugin is self-contained: model registration, platform detection, request
validation, scheduling, worker execution, model loading, async decode, gathered
data-parallel execution, and `tt-run` / MPI launch orchestration all live here.
Nothing TT-specific needs to touch vLLM core.

## Package layout

```text
plugins/vllm-tt-plugin/
+-- src/vllm_tt_plugin/
|   +-- entrypoints.py       # vLLM plugin entry points
|   +-- platform.py          # TTPlatform and config validation
|   +-- model_registry.py    # TT model architecture registration
|   +-- worker.py            # TT worker implementation
|   +-- model_runner.py      # TT model execution bridge
|   +-- scheduler.py         # TT scheduling policy
|   +-- engine.py            # TT engine core and DP engine processes
|   +-- launcher.py          # tt-run / MPI launch integration
|   +-- loader.py            # TT model loader
|   +-- input_batch.py       # TT input-batch representation
|   +-- async_decode.py      # Decode overlap helpers
|   +-- config.py            # TT plugin config access
+-- docs/tt_metal/           # TT runtime setup and operator notes
+-- examples/                # Offline and OpenAI-server examples
+-- tests/tt/                # Server-facing TT plugin tests
```

## Install

The normal developer path is to create or activate a tt-metal environment, then
install vLLM and this plugin from the repository root:

```bash
source plugins/vllm-tt-plugin/docs/install-vllm-tt.sh
```

To install or refresh only the plugin package with its runtime dependencies:

```bash
uv pip install -e "plugins/vllm-tt-plugin[runtime]" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match
```

If the active tt-metal environment already owns the runtime dependencies, use a
no-dependency editable install:

```bash
uv pip install -e plugins/vllm-tt-plugin --no-deps
```

The plugin requires Python `>=3.10,<3.14` and depends on `vllm`. The `runtime`
extra pins the CPU PyTorch stack and TT-side Python dependencies used by the
current integration.

## Verify plugin discovery

The editable install registers two vLLM entry points:

| Entry point group | Name | Target |
| --- | --- | --- |
| `vllm.general_plugins` | `tt_model_registry` | `vllm_tt_plugin.entrypoints:register` |
| `vllm.platform_plugins` | `tt` | `vllm_tt_plugin.entrypoints:platform_plugin` |

`platform_plugin()` returns `vllm_tt_plugin.platform.TTPlatform` only when
`ttnn` is importable. This keeps ordinary vLLM environments from accidentally
selecting the TT platform.

Quick checks:

```bash
python -c "import vllm_tt_plugin; print(vllm_tt_plugin.__file__)"
python -c "import ttnn; print('ttnn available')"
```

If `VLLM_PLUGINS` is set, it must allow both TT entry points:

```bash
export VLLM_PLUGINS=tt,tt_model_registry
```

## Quick start

Run offline generation:

```bash
MESH_DEVICE=T3K \
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model meta-llama/Llama-3.1-70B-Instruct
```

Run the OpenAI-compatible server:

```bash
VLLM_RPC_TIMEOUT=100000 MESH_DEVICE=T3K \
python plugins/vllm-tt-plugin/examples/server_example_tt.py \
  --model meta-llama/Llama-3.1-70B-Instruct
```

Send a completion request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "prompt": "San Francisco is a",
    "max_tokens": 32,
    "temperature": 1,
    "top_p": 0.9,
    "top_k": 10
  }'
```

See `docs/tt_metal/README.md` for full model-specific setup, Hugging Face
access notes, multimodal examples, benchmarking commands, and multi-host launch
examples.

## Runtime architecture

`TTPlatform.check_and_update_config()` is the main handoff from vLLM into the TT
runtime. It validates configuration, registers TT model architectures, and
selects the TT-owned runtime classes through vLLM's existing extension points:

| vLLM config field | TT implementation |
| --- | --- |
| `parallel_config.worker_cls` | `vllm_tt_plugin.worker.TTWorker` |
| `parallel_config.engine_core_cls` | `vllm_tt_plugin.engine.TTEngineCore` |
| `parallel_config.engine_core_proc_cls` | `vllm_tt_plugin.engine.TTEngineCoreProc` |
| `parallel_config.dp_engine_core_proc_cls` | `vllm_tt_plugin.engine.TTDPEngineCoreProc` |
| `parallel_config.engine_core_launcher_cls` | `vllm_tt_plugin.launcher.TTCoreEngineLauncher` |
| `scheduler_config.scheduler_cls` | `vllm_tt_plugin.scheduler.TTScheduler` |

The execution model matches TT hardware characteristics:

- A TT step is either prefill-only or decode-only.
- Chunked prefill is not used.
- Async scheduling overlaps decode submission with host-side scheduling when
  the model declares support.
- Gathered DP collects local rank inputs, executes across the TT mesh, and
  scatters outputs back to the participating ranks.
- Multi-host execution uses `tt-run` / MPI while vLLM sees a normal
  engine-client handshake.

For a deeper walk-through of the scheduling and execution model, read
`docs/tt_metal/SCHEDULING.md`.

## Configuration

TT options are passed through vLLM's generic plugin namespace, keeping them
cleanly separated from the core vLLM CLI:

```bash
--plugin-config '{"tt": {"sample_on_device_mode": "all"}}'
```

Plugin code reads this through `vllm_tt_plugin.config.get_tt_config()`, which
returns `vllm_config.plugin_config["tt"]`.

Common options:

| Key | Purpose |
| --- | --- |
| `sample_on_device_mode` | Select on-device sampling mode, currently `all` or `decode_only` when supported by the model. |
| `trace_mode` | Control TT tracing: `all`, `decode_only`, or `none`. Default: `all`. |
| `enable_model_warmup` | Warm up the model before the server reports healthy. Default: `true`. |
| `input_queue_batching_delay` | Short idle delay in seconds to allow more requests to coalesce before TT execution. Default: `0.002`. |
| `optimizations` | Select model/runtime optimization profile, such as `accuracy` or `performance`. |
| `register_test_models` | Register non-production TT test models for infrastructure tests. Default: `false`. |
| `rank_binding` | Rank-binding YAML used for `tt-run` / MPI launches. |
| `mpi_args` | MPI launch arguments, for example host and rankfile settings. |
| `extra_ttrun_args` | Additional raw arguments passed to `tt-run`. |
| `config_pkl_dir` | Shared directory used to pass launch config to remote hosts. |
| `env_passthrough` | Environment variable names or glob patterns propagated to remote hosts. |

Example multi-host launch across two Galaxy systems with DP=2:

```bash
MESH_DEVICE="(8,8)" \
python -u plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model <MODEL_NAME> \
  --data_parallel_size 2 \
  --async_engine \
  --plugin-config '{
    "tt": {
      "rank_binding": "<TT_METAL>/tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml",
      "mpi_args": "--host <HOST1>,<HOST2> --map-by rankfile:file=/etc/mpirun/rankfile",
      "extra_ttrun_args": "--tcp-interface cnx1",
      "config_pkl_dir": "<SHARED_TMP_DIR>",
      "fabric_config": "FABRIC_1D",
      "fabric_reliability_mode": "RELAXED_INIT",
      "env_passthrough": ["VLLM_*", "MESH_DEVICE"]
    }
  }'
```

## Supported model families

The plugin registers TT-prefixed model architectures backed by tt-metal model
implementations. Current families:

- Llama 3.1 / 3.2 text models (`TTLlamaForCausalLM`)
- Llama 3.2 vision models (`TTMllamaForConditionalGeneration`)
- Qwen 2.5 and Qwen 3 text models (`TTQwen2ForCausalLM`, `TTQwen3ForCausalLM`)
- Qwen 2.5-VL and Qwen 3-VL vision-language models
- Mistral and Mistral 3 multimodal models
- Gemma 3 multimodal models
- DeepSeek V3 (`TTDeepseekV3ForCausalLM`)
- GPT-OSS 20B / 120B (`TTGptOssForCausalLM`)

Model availability, supported device shapes, max sequence limits, and required
environment variables are documented per model in `docs/tt_metal/README.md` and
in the corresponding tt-metal model demos.

## Operational constraints

`TTPlatform` rejects or adjusts unsupported feature combinations early, giving a
clear error before anything reaches the device:

- Tensor parallel and pipeline parallel execution are not supported.
- Speculative decoding is not currently supported.
- LoRA is not currently supported.
- Chunked prefill is disabled.
- Prompt logprobs are rejected at request validation time.
- Prefix caching is enabled only for models that declare TT support for it.
- Async decode overlap is enabled only for models that declare the capability.

These are TT runtime characteristics, not vLLM plugin API limitations.

## Testing

The plugin ships server-facing tests under `tests/tt`. Start a vLLM server with
a TT model, then run:

```bash
pytest plugins/vllm-tt-plugin/tests/tt -v \
  --tt-server-url=http://localhost:8000 \
  --tt-model-name=meta-llama/Llama-3.1-8B-Instruct
```

Tests cover request isolation, sampling behavior, penalties, logprobs,
host-only parameter handling, and TT utility helpers.

## Benchmarking

Offline benchmarking via the example script:

```bash
MESH_DEVICE=T3K \
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --measure_perf
```

Client-server benchmarking with `vllm bench serve`:

```bash
vllm bench serve \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 128 \
  --num-prompts 32 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el
```

For prefix-cache experiments, use prompts with shared prefixes or pass
`--random-prefix-len` to `vllm bench serve`.

## Status

This plugin currently lives alongside a vLLM fork that adds a small set of
generic plugin mechanism extensions not yet present in upstream vLLM. Once those
extensions are part of upstream, this plugin will be fully self-contained and
will work against stock vLLM without any fork.

The extensions needed are:

- `VllmConfig.plugin_config` — a namespaced dict that lets any plugin pass
  backend-specific configuration without touching the core CLI.
- `ParallelConfig.engine_core_cls` — lets a platform plugin select an
  alternative `EngineCore` implementation.
- `ParallelConfig.engine_core_proc_cls` — lets a platform plugin select an
  alternative `EngineCoreProc` implementation.
- `ParallelConfig.dp_engine_core_proc_cls` — lets a platform plugin select an
  alternative data-parallel engine process implementation.
- `ParallelConfig.engine_core_launcher_cls` — lets a platform plugin own the
  engine launch and handshake topology.

## Development notes

- Normal Python changes under `src/vllm_tt_plugin/` take effect after
  restarting the Python or vLLM process.
- Reinstall the plugin when package metadata or entry points change, such as
  edits to `pyproject.toml`.
- TT-specific options belong under the `tt` key in `--plugin-config`, not in
  the vLLM CLI namespace.
- Model capability declarations (`model_capabilities` dict on the model class)
  are the preferred way to gate features like async decode and prefix caching,
  rather than hard-coded model-name checks.
