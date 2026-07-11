Here's how the two repos fit together.

## TLDR

`/localdev/gwang/vllm_duo/vllm` is Tenstorrent's **fork** of vLLM (`origin = tenstorrent/vllm`, `upstream = vllm-project/vllm`). Almost nothing TT-specific lives in vLLM core. Instead, all the glue is a self-contained **out-of-tree plugin** at `plugins/vllm-tt-plugin/`. On the tt-metal side, each model exposes a thin `generator_vllm.py` **bridge class** on top of its existing `Generator`/`Transformer` implementation.

The boundary between them is **duck-typed, not a shared base class**. The plugin never imports tt-metal model classes directly — it resolves them by **string import paths** registered into vLLM's `ModelRegistry`, then calls a fixed set of methods (`initialize_vllm_model`, `prefill_forward`, `decode_forward`, `allocate_kv_cache*`, `get_max_tokens_all_users`, …) and reads capabilities via `getattr`. The **wire format is torch tensors** in both directions; `ttnn` (the tt-metal runtime) is the only hard dependency, and its mere importability is what activates the whole backend.

```
┌─────────────────────── vllm repo (TT fork) ───────────────────────┐
│  vllm core (≈ upstream)                                            │
│  plugins/vllm-tt-plugin/  ← ALL the glue                           │
│    entrypoints → platform → worker → model_runner → loader         │
│      builds TTModelInput (torch tensors) ─────────────┐            │
└───────────────────────────────────────────────────────┼──────────┘
                                                          │ duck-typed call
                        registered via string import path │ (torch in/out)
┌───────────────────── tt-metal repo ─────────────────────▼──────────┐
│  models/**/generator_vllm.py   ← thin bridge (the contract)        │
│  models/tt_transformers/tt/generator.py  (Generator: prefill/decode)│
│  models/**/model.py (Transformer)  →  ttnn  →  mesh device          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1. Discovery & activation

The plugin registers two standard vLLM entry points in `plugins/vllm-tt-plugin/pyproject.toml`:

```28:32:plugins/vllm-tt-plugin/pyproject.toml
[project.entry-points."vllm.general_plugins"]
tt_model_registry = "vllm_tt_plugin.entrypoints:register"

[project.entry-points."vllm.platform_plugins"]
tt = "vllm_tt_plugin.entrypoints:platform_plugin"
```

`platform_plugin()` gates the entire backend on whether the tt-metal runtime is importable — this is the "am I on TT hardware" test:

```50:59:plugins/vllm-tt-plugin/src/vllm_tt_plugin/entrypoints.py
def platform_plugin() -> str | None:
    """Return the TT platform class when TT runtime libraries are present."""
    try:
        import ttnn  # noqa: F401
    except Exception as exc:
        logger.debug("TT plugin platform is not available because: %s", exc)
        return None
    ...
    return "vllm_tt_plugin.platform.TTPlatform"
```

Build-time detail that matters: base vLLM is built with `VLLM_TARGET_DEVICE=empty` (no CUDA), and the plugin runs *inside the tt-metal python_env*, which already owns torch/transformers/numpy. So the plugin's only declared dependency is `tblib` — everything else comes from the tt-metal environment. That's why the two repos share one venv.

## 2. The central handoff: `TTPlatform.check_and_update_config`

This is the single most important function on the vLLM side (`platform.py`). vLLM calls it once the config is assembled, and it rewrites `VllmConfig` to swap TT-owned implementations into vLLM's extension points:

```293:298:plugins/vllm-tt-plugin/README.md
| `parallel_config.worker_cls` | `vllm_tt_plugin.worker.TTWorker` |
| `parallel_config.engine_core_cls` | `vllm_tt_plugin.engine.TTEngineCore` |
| `parallel_config.engine_core_proc_cls` | `vllm_tt_plugin.engine.TTEngineCoreProc` |
| `parallel_config.dp_engine_core_proc_cls` | `vllm_tt_plugin.engine.TTDPEngineCoreProc` |
| `parallel_config.engine_core_launcher_cls` | `vllm_tt_plugin.launcher.TTCoreEngineLauncher` |
| `scheduler_config.scheduler_cls` | `TTScheduler` or `TTLaneCoordinator` |
```

In the same hook it also: forces off chunked prefill, asserts no TP/PP/speculative/LoRA, clamps `max_logprobs` to 20 (device does top-32, OpenAI API allows 20), registers the TT models, prepends `"TT"` to the HF architecture name, and reads model capabilities to decide sampling/async/prefix-caching. Those `parallel_config.engine_core_*` fields are the *only* additions the fork makes to vLLM core that aren't upstream yet (see README "Status") — everything else is plugin-side.

## 3. Model registration: string paths, not imports

`register_tt_models()` maps HF arch names to tt-metal import paths as **strings**, so vLLM core never has a compile-time dependency on tt-metal:

```244:256:plugins/vllm-tt-plugin/src/vllm_tt_plugin/entrypoints.py
    # Llama3.1/3.2 - Text
    _register_model_if_missing(ModelRegistry, "TTLlamaForCausalLM", path_llama_text)
    ...
    # Qwen2.5 - Text
    path_qwen_text = "models.tt_transformers.tt.generator_vllm:QwenForCausalLM"
    _register_model_if_missing(ModelRegistry, "TTQwen2ForCausalLM", path_qwen_text)
```

The `"TT"` prefixing in `check_and_update_config` (`arch_names[i] = "TT" + arch_names[i]`) means a normal `meta-llama/Llama-3.1-70B` request whose HF config says `LlamaForCausalLM` gets routed to `TTLlamaForCausalLM` → `models.tt_transformers.tt.generator_vllm:LlamaForCausalLM`. Env vars like `TT_LLAMA_TEXT_VER` switch which tt-metal implementation backs a family (e.g. `tt_transformers` vs the single-mesh `llama3_70b_galaxy` generator).

## 4. The interface contract (the part you care about)

A tt-metal model class registered above must implement a fixed duck-typed API. This *is* the interface. On the tt-metal side these are the `generator_vllm.py` bridge classes subclassing `Generator`:

**a) Construction — a classmethod factory.** The loader calls it (never `__init__` directly):

```38:45:plugins/vllm-tt-plugin/src/vllm_tt_plugin/loader.py
        model = model_class.initialize_vllm_model(
            model_config.hf_config,
            device_config.device,      # the opened ttnn mesh device
            max_batch_size,
            max_seq_len=model_config.max_model_len,
            tt_data_parallel=tt_data_parallel,
            optimizations=optimizations,
        )
```

On the tt-metal side that builds the real `Transformer`(s), one per DP submesh, and wraps them in the `Generator`:

```660:696:models/tt_transformers/tt/generator_vllm.py
    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, ...):
        ...
        tt_model, model_args = initialize_vllm_text_transformer(hf_config, tt_data_parallel, mesh_device, ...)
        return cls(tt_model, model_args, mesh_device)
```

**b) Forward — `prefill_forward` / `decode_forward`.** The plugin calls these with a kwargs dict of torch tensors. Prefill (`model_runner.py`):

```2328:2335:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
        kwargs = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": self.kv_caches,
            "enable_trace": self.trace_mode in ["all"],
            "prompt_lens": model_input.prompt_lens,
            "start_pos": model_input.input_positions,
        }
```

The bridge usually just forwards to the base `Generator` (`generator_vllm.py`):

```702:706:models/tt_transformers/tt/generator_vllm.py
    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)
```

The base `Generator.__init__` docstring states the boundary design principle explicitly — the contract is torch-in/torch-out:

```84:86:models/tt_transformers/tt/generator.py
        For bringup, make this class general to any backend implementation, as long as it takes torch tensors and returns torch tensors.
```

**c) KV cache allocation.** vLLM computes *shapes*, tt-metal allocates the *actual DRAM tensors*. Legacy uniform models expose `allocate_kv_cache(shape, dtype, num_layers)`; hybrid models expose `allocate_kv_cache_per_layer(specs)`. The runner picks whichever exists:

```444:445:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
        if hasattr(self.model, "allocate_kv_cache_per_layer"):
            return self.model.allocate_kv_cache_per_layer(per_layer_specs)
```

**d) Capabilities — a class attribute dict** read via `getattr`. This is how the plugin gates features without model-name checks:

```629:633:models/tt_transformers/tt/generator_vllm.py
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }
```

`check_and_update_config` reads these to enable/disable on-device sampling, async decode overlap, and prefix caching.

**e) KV sizing hook — `get_max_tokens_all_users(...)` classmethod.** Because the TT backend doesn't run vLLM's memory-profiling pass, it asks the model how big the KV cache should be, then overrides vLLM's block count directly:

```531:537:plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py
        max_tokens_all_users = model_class.get_max_tokens_all_users(
            model_name=model_config.model,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            max_model_len=model_config.max_model_len,
            max_num_seqs=get_tt_per_lane_max_num_seqs(vllm_config),
        )
```

```298:300:plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py
        num_tt_blocks = get_num_available_blocks_tt(self.vllm_config, self.num_devices)
        self.cache_config.num_gpu_blocks_override = num_tt_blocks
        return 1 << 64   # dummy "available memory"
```

**f) Optional hybrid-attention hook — `get_kv_cache_spec(vllm_config)` classmethod.** Its mere presence opts a model into upstream's hybrid KV cache manager (Gemma3/4, GPT-OSS). The plugin only sends `page_tables_per_layer` to classes that expose it. `HybridAttentionForCausalLM` in `generator_vllm.py` is the base that provides it.

## 5. The data-plane objects

Everything crossing the boundary per step is packed into two frozen dataclasses in `model_input.py`. `TTModelInput` carries the tensors; note the fields are all torch tensors / plain lists — no ttnn types leak up to vLLM:

```64:89:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_input.py
class TTModelInput:
    input_tokens: torch.Tensor
    input_positions: torch.Tensor
    prompt_lens: list[int] | None
    block_tables: torch.Tensor
    block_tables_per_group: list[torch.Tensor]
    block_tables_per_layer: list[torch.Tensor] | None
    unpadded_batch_size: int | list[int]
    tt_sampling_params: TTSamplingParams
    multi_modal_kwargs: dict[str, Any]
    ...
```

Key design point on **paging**: vLLM owns the *block/page tables* (which logical block maps to which physical KV block) and hands them across as torch tensors; tt-metal owns the *physical KV tensors* in device DRAM. `prompt_lens is None` is the sentinel that distinguishes a decode step from a prefill step throughout the runner.

## 6. Execution lifecycle (end to end)

1. **`TTWorker.init_device`** — re-runs `check_and_update_config` in the worker subprocess, then opens the ttnn mesh device (only on local DP rank 0). `MESH_DEVICE` env maps names → grids (`T3K`→`(1,8)`, `TG`→`(8,4)`, …) in `worker.py:get_mesh_grid`. Fabric config is set *before* opening the mesh.
2. **`load_model`** → `TTModelLoader` → `initialize_vllm_model` (weights loaded, `Transformer`s built per submesh).
3. **KV cache** — `get_kv_cache_spec` (shapes) → `determine_available_memory` overrides block count → `initialize_kv_cache` → `allocate_kv_cache*` allocates ttnn DRAM tensors.
4. **`compile_or_warm_up_model`** → `warmup_model` does two-phase warmup: compile all op variants, then capture ttnn traces (`model_runner.py:2952`). TT uses trace capture/replay instead of torch.compile (note `simple_compile_backend = "eager"` — the tt-metal triton is incompatible with inductor).
5. **Per step**: `execute_model(scheduler_output)` → `build_model_input` produces a `TTModelInput` → `prefill_forward`/`decode_forward` runs on device → returns **`None`** and defers sampling. The engine then calls **`sample_tokens(grammar_output)`**, which pops the pending forward and produces the `ModelRunnerOutput`.

That **forward-then-sample split** is deliberate: it lets the engine compute the grammar bitmask (and, in async decode, overlap the device readback) while the device is busy:

```331:343:plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py
    def execute_model(self, scheduler_output):
        """... Returns ``None``: the forward leaves a pending sampler that the
        engine finalizes via ``sample_tokens``. ..."""
        return self.model_runner.execute_model(scheduler_output)
```

## 7. Async decode overlap — the readback contract

The most TT-specific piece of the forward interface is the split decode submission, used for host/device overlap. Models that set `supports_async_decode` implement `decode_forward(..., read_from_device=False)` returning on-device handles, plus `read_decode_output(..., async_read=True)` which issues non-blocking `.cpu()` reads and returns ttnn events:

```205:217:plugins/vllm-tt-plugin/docs/SCHEDULING.md
1. Submit decode work with `decode_forward(..., read_from_device=False)`.
2. Ask the model to start host readback with `read_decode_output(..., async_read=True)`.
3. Keep the returned read events with the submission record.
4. Later, during finalization, wait on those read events with `ttnn.event_synchronize(...)`.
5. Only after those events complete, convert the decode output into normal host tensors and sampling results.
```

This is the one place ttnn handles/events legitimately cross back up into the plugin (held opaquely in `async_decode.py`), because the whole point is to defer the blocking read.

## 8. On-device sampling

Sampling can happen either on host (vLLM's normal `Sampler` + logits processors) or **on device**. When enabled and the batch qualifies (`check_perform_device_sampling`), the plugin passes `TTSamplingParams` (temperature/top_k/top_p/penalties/seed as lists) into `prefill_forward`/`decode_forward`, and the device returns sampled *tokens* instead of logits. "Compat sampling" falls back to the host path whenever a request needs something the device can't do (min_p, bad_words, logit_bias, structured output, prompt_logprobs, etc.) — see `TTPlatform.compat_sampling_required`.

## 9. Scheduling & parallelism specializations

- **`TTScheduler`** (subclass of vLLM's `AsyncScheduler`) enforces TT's execution model: every step is **all-prefill or all-decode**, never mixed, and no chunked prefill. It exposes `set_forced_mode` so DP ranks can globally agree on prefill vs decode.
- **Parallelism is expressed through the model, not vLLM's TP/PP.** TP/PP are asserted off; the model shards itself across the mesh internally. Data parallelism comes in three flavors: non-DP, **gathered multi-process DP** (gather per-rank inputs → execute merged batch on rank 0 → scatter outputs), and **single-process lane-DP** for Galaxy generators, where `--data_parallel_size N` is transparently rewritten into N in-process lanes driven by `TTLaneCoordinator` (`platform.py:_convert_galaxy_gather_dp_to_lanes`). `create_submeshes` in `generator.py` does the actual mesh partition.

---

## Why the interface is designed this way

- **Plugin-in-vLLM, bridge-in-tt-metal, joined by strings.** vLLM core stays ~upstream; tt-metal adds only thin `generator_vllm.py` wrappers over its existing `Generator`. Neither repo imports the other's classes directly — registration is by import-path string and feature detection is by `getattr`/`hasattr`. This is what lets the plugin claim it will eventually run against *stock* upstream vLLM once the five `engine_core_*` config fields land upstream.
- **torch tensors as the wire format.** The `Generator` explicitly promises "takes torch tensors and returns torch tensors." ttnn types stay below the bridge, except for the deliberate async-decode handle/event leak needed for overlap.
- **Capabilities over model-name checks.** `model_capabilities`, `get_kv_cache_spec`, and `get_max_tokens_all_users` push model-specific policy *into the model class*, so the plugin logic stays generic.
- **Config via one generic namespace.** All TT knobs ride in vLLM's `additional_config["tt"]` (read through `config.get_tt_config`), so no vLLM CLI surface changes are needed.

---

## Llama 3 8B Deep Dive

This section uses Llama 3 8B as the concrete model, specifically the normal
`tt_transformers` path:

```text
HF arch:      LlamaForCausalLM
TT arch:      TTLlamaForCausalLM
TT class:     models.tt_transformers.tt.generator_vllm:LlamaForCausalLM
Base runtime: models.tt_transformers.tt.generator:Generator
```

For this model, the important class is `LlamaForCausalLM` in
`models/tt_transformers/tt/generator_vllm.py`. It declares:

```627:633:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/generator_vllm.py
class LlamaForCausalLM(Generator):
    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }
```

That means Llama 3 8B is one of the better-supported TT vLLM models: it can use
prefix caching, it can use the TT async decode path, and it can sample on device
when the request batch stays inside the device-supported sampling subset.

### Platform mapping: N150, N300, T3K

The hardware platform is selected by `MESH_DEVICE`; `TTWorker` maps it to a
ttnn mesh shape:

```723:733:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py
    mesh_grid_dict = {
        "N150": (1, 1),
        ...
        "N300": (1, 2),
        ...
        "T3K": (1, 8),
```

For Llama 3 8B, the bridge has one special N150 rule:

```671:681:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/generator_vllm.py
        if (
            ("3.1-8B" in hf_model_name or "3.2-11B" in hf_model_name)
            and mesh_device.get_num_devices() == 1
            and is_wormhole_b0()
        ):
            MAX_PROMPT_LEN = 32768
            if max_seq_len > MAX_PROMPT_LEN:
                raise ValueError(
                    f"TT-LLama8B and TT-Llama11B do not support max_model_len greater than {MAX_PROMPT_LEN} on N150 "
```

So the practical matrix is:

| Platform | Mesh | Local DP shape for Llama 3 8B | Notes |
| --- | --- | --- | --- |
| `N150` | `(1, 1)` | `DP=1` only for local single-host serving | `max_model_len` must be <= `32768` on Wormhole N150. |
| `N300` | `(1, 2)` | `DP=1` uses both devices; `DP=2` splits into two one-device submeshes | `DP=2` effectively runs two N150-sized replicas, so the one-device KV/token budget rule applies per replica. |
| `T3K` | `(1, 8)` | `DP=1`, `DP=2`, `DP=4`, `DP=8` are the natural local splits | DP must divide the 8-device mesh; `DP=8` again gives one device per replica. |

The exact split is done inside tt-metal, not vLLM core:

```2897:2907:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/generator.py
    num_rows, num_cols = _mesh_shape_tuple(mesh_device.shape)
    num_devices = num_rows * num_cols
    assert num_devices % data_parallel == 0, f"Unsupported device split: {num_devices} devices, {data_parallel} groups"
    ...
    return mesh_device.create_submeshes(ttnn.MeshShape(1, num_devices // data_parallel))
```

For N150/N300/T3K, ignore Galaxy lane-DP. Llama 3 8B uses gathered DP when
`--data_parallel_size > 1`; it does not trigger `TTLaneCoordinator`, because
that conversion is only for the Galaxy single-execute generators.

### How local DP actually works

There are two distinct meanings of "DP" in this integration:

1. vLLM DP ranks: multiple scheduler/engine ranks, each with its own request
   queue and scheduler state.
2. tt-metal DP submeshes: the single opened mesh is split into `DP` submeshes,
   each running a replica of the model.

For local gathered DP, only `data_parallel_rank_local == 0` opens the TT mesh and
loads the model:

```136:162:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py
        local_dp_rank = self.parallel_config.data_parallel_rank_local
        # Open mesh only on local DP rank 0 (device ranks).
        if local_dp_rank == 0:
            self.mesh_device = open_mesh_device(
                get_tt_config(self.vllm_config), self.trace_mode, local_dp_rank
            )
            ...
        ...
        # Only local DP rank 0 (device rank) loads the model
        if self.parallel_config.data_parallel_rank_local == 0:
            self.model_runner.load_model()
```

The model loader passes the effective TT data parallel size into the tt-metal
bridge:

```35:45:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/loader.py
        tt_data_parallel = get_tt_data_parallel_size(vllm_config)
        max_batch_size = get_tt_max_batch_size(vllm_config)

        model = model_class.initialize_vllm_model(
            model_config.hf_config,
            device_config.device,
            max_batch_size,
            max_seq_len=model_config.max_model_len,
            tt_data_parallel=tt_data_parallel,
            optimizations=optimizations,
        )
```

Then `initialize_vllm_text_transformer()` creates one `Transformer` per submesh:

```346:380:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/generator_vllm.py
    submesh_devices = create_submeshes(mesh_device, tt_data_parallel)
    ...
    for i, submesh in enumerate(submesh_devices):
        tt_model_i = Transformer(
            args=model_args[i],
            mesh_device=submesh,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=model_args[i].weight_cache_path(dtype),
            use_paged_kv_cache=True,
        )
        tt_model.append(tt_model_i)
```

So for `MESH_DEVICE=T3K --data_parallel_size 4`, vLLM has four DP ranks, but
the device-owning process builds four tt-metal `Transformer` handles over four
`1x2` submeshes. Each logical rank's input is gathered into one merged batch,
the merged batch runs on the four submeshes, and the sampled tokens are scattered
back to the vLLM ranks.

The per-rank batch contract stays sane:

- vLLM user flag `--max_num_seqs B` is per DP rank.
- For gathered DP, `get_tt_max_batch_size()` returns `B * DP`.
- The tt-metal bridge divides that by `tt_data_parallel`, so each submesh still
  gets `B`.

### Gathered-DP control flow

`TTDPEngineCoreProc` owns gathered DP. Every loop starts with rank alignment and
global negotiation, because all ranks must execute the same batch mode:

```324:338:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/engine.py
    def _dp_negotiate_forced_mode(self) -> TTSchedulingMode:
        has_running = bool(getattr(self.scheduler, "running", []))
        has_waiting = bool(getattr(self.scheduler, "waiting", False))
        ...
        dist.all_reduce(intent_tensor, op=dist.ReduceOp.MAX, group=self.dp_group)
        forced_mode = TTSchedulingMode.from_prefill_intent(int(intent_tensor.item()))
```

If any rank wants prefill, everyone schedules prefill-only. Otherwise everyone
decodes. This matters because TT cannot run mixed prefill+decode batches.

The gather path is:

1. Each rank calls `build_dp_model_input()`.
2. Decode inputs are tensor-packed into fixed-shape `int_inputs` and
   `float_inputs`.
3. Prefill inputs are gathered as Python objects.
4. Rank 0, and any other device-owning MPI rank, receives the merged payload.
5. Device rank calls `concat_and_execute_dp()`.
6. Output tokens are packed as `[world, per_rank_batch, 1]`.
7. Results scatter back to every vLLM DP rank.

Decode is optimized to avoid object gathers for the hot path:

```1373:1402:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
        # Pack into flattened tensors to reduce number of collectives.
        # B = max batch size, W = max_num_blocks_per_req, G = num kv_cache_groups.
        # Layout includes one block_table block per group (G*B*W ints) so
        # hybrid models can carry per-group routing through DP gather; for
        # the legacy single-group case G == 1 and the layout is byte-
        # identical to the pre-hybrid format.
        block_tables_packed = torch.cat(
            [
                bt[:, :max_blocks_decode_batch].contiguous().view(-1)
                for bt in block_tables_per_group
            ],
            dim=0,
        )
```

For Llama 3 8B, `G == 1`, so decode gather carries exactly one page table per
rank. Hybrid models use the same wire layout with `G > 1`.

### Async decode controller

For Llama 3 8B, async decode is supported, but it only hits the fast path under
specific conditions:

- `--async-scheduling` must be enabled.
- `trace_mode` must not be `"none"`.
- The model must support async decode (`supports_async_decode=True`).
- The batch must be decode-only.
- The padded decode layout must be stable (`reset_batch=False`).
- Sampling must stay on device.
- No structured outputs, penalties, bad words, allowed token masks, logprobs,
  custom logits processors, or host-only sampling requirements.

The controller checks that directly:

```249:291:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py
    def steady_decode_scheduler_invariants_met(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
    ) -> bool:
        ...
        if is_prompt or runner._decode_layout_changed_since_last_decode:
            return False
        ...
        if not input_batch.no_penalties:
            return False
        if not input_batch.no_allowed_token_ids:
            return False
        if input_batch.sampling.bad_words_token_ids:
            return False
        ...
        return runner.check_perform_device_sampling(
            is_decode=True,
            has_structured_outputs=False,
        )
```

The low-level async contract is:

```622:635:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py
        tt_out = runner.model.decode_forward(
            **kwargs,
            **enc_dec_kwargs,
            enable_trace=enable_trace,
            read_from_device=read_from_device,
        )
        read_events = None
        if async_read:
            if hasattr(runner.model, "read_decode_output"):
                tt_out, read_events = cast(
                    tuple[Any, list[Any]],
                    runner.model.read_decode_output(tt_out, async_read=True),
                )
```

And finalization waits the TT read events:

```663:675:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py
        if submission.read_events is not None:
            for read_event in submission.read_events:
                ttnn.event_synchronize(read_event)
            tt_out = submission.tt_out
        ...
            tt_out = runner.model.process_decode_output_host(
                tt_out,
                is_tokens=submission.perform_device_sampling,
            )
```

This is not "run arbitrary batches async." It is narrower: submit decode,
start an async host read, let the engine do scheduler work while the read is in
flight, then finalize exactly once. The wrapper has an idempotent lock because
either the vLLM executor output thread or the runner's drain path may get there
first:

```102:110:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/async_decode.py
    def ensure_finalized(self) -> Any:
        if self._finalized:
            return self._cached_output
        with self._finalize_lock:
            if not self._finalized:
                self._cached_output = self._get_output_impl()
                self._finalized = True
                self._completion_event.set()
        return self._cached_output
```

For gathered DP, the same `TTAsyncDecodeController` is used, but the outer loop
is more conservative. `step_dp_with_batch_queue()` now finalizes the previous
gathered result before submitting the next device execution, because submitting
the next decode reads `input_batch.token_ids_cpu`, and that state is only correct
after applying the previous sampled tokens:

```411:421:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/engine.py
        # Always finalize the previous step before submitting the next one.
        #
        # The submit reads ``input_batch.token_ids_cpu`` to build the decode
        # input for the next step; that table is only updated once
        # ``apply_dp_execution_result`` runs inside ``_finalize_previous``.
        ...
        finalize_before_submit = prev_handle is not None
```

So for Llama 3 8B DP, async decode still helps by deferring readback and letting
some rank coordination/scheduling happen while previous work is in flight, but
it is not as aggressively pipelined as single-rank non-DP steady decode.

### Hybrid-KV page-table routing

For Llama 3 8B, this is mostly a "not active" story.

Llama 3 8B does **not** inherit `HybridAttentionForCausalLM`, does not expose
`get_kv_cache_spec`, and therefore the TT worker uses the default single-group
KV spec:

```247:283:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py
    def _build_default_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Single-layer spec used by the legacy non-hybrid path. Downstream
        sizing is overridden via ``cache_config.num_gpu_blocks_override``.
        """
        ...
        return {"foo": attn_spec}
```

With one KV cache group:

- `self._layer_to_group_idx` stays `None`.
- `TTModelInput.block_tables_per_layer` stays `None`.
- The plugin passes only the legacy `page_table` kwarg.
- tt-metal broadcasts that same page table to every attention layer.

The fallback is explicit in `Transformer.forward()`:

```901:917:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/model.py
            # vLLM hybrid kv-cache-groups: each attention layer gets its own
            # paged pool (sliding-window vs full-attention have different
            # block counts). When ``page_tables_per_layer`` is None we fall
            # back to broadcasting the single ``page_table`` to every layer
            # — byte-equivalent to the pre-hybrid path used by every legacy
            # caller (demos, unit tests, non-hybrid vLLM bridges).
            layer_page_table = page_tables_per_layer[i] if page_tables_per_layer is not None else page_table
```

The hybrid path matters for Gemma/GPT-OSS style models with mixed full/sliding
attention. There, the model class exposes `get_kv_cache_spec`, the plugin builds
multiple `block_tables_per_group`, expands them to per-layer tables, and passes
`page_tables_per_layer` into the model. tt-metal then keeps persistent per-layer
ttnn page-table tensors so traced replay sees stable device addresses:

```667:711:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/model.py
    def _page_tables_to_ttnn(self, page_tables_per_layer):
        """Resolve a per-layer list of ``torch.Tensor`` page tables to a
        list of *persistent* ttnn device tensors (allocate-only).
        ...
                persistent.append(
                    ttnn.from_torch(
                        pt,
                        device=self.mesh_device,
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=self._page_table_mesh_mapper(pt.shape[0]),
                    )
                )
```

For Llama 3 8B, if you are debugging page tables, do not chase the hybrid
per-layer path unless someone changed the model class to expose
`get_kv_cache_spec`. The active path is single `page_table`.

### `tt-run` / MPI

For N150/N300/T3K on a single host, you normally do **not** need `tt-run`/MPI.
Set `MESH_DEVICE` and run vLLM normally; local DP is handled by vLLM DP ranks
plus tt-metal submeshes.

`tt-run`/MPI becomes relevant when the TT mesh or DP device ranks span hosts.
The plugin owns that launch through `TTCoreEngineLauncher`.

The activation knob is `additional_config["tt"]["rank_binding"]`:

```186:218:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/launcher.py
def parse_tt_mpi_params(vllm_config: VllmConfig) -> tuple[str | None, set[int]]:
    ...
    rank_binding_file = tt_config.get("rank_binding")
    ...
        mpi_world = len(rb.get("rank_bindings", []))
        ...
        # Only the first DP rank in each MPI segment owns a TT device process.
        # The other DP ranks stay local and participate as non-device ranks.
        dp_size_per_mpi_rank = dp_size // mpi_world
        device_dp_ranks = {i * dp_size_per_mpi_rank for i in range(mpi_world)}
        non_device_dp_ranks = {i for i in range(dp_size) if i not in device_dp_ranks}
```

The local launcher serializes the full `VllmConfig` to a shared directory,
creates a temporary rank-binding YAML with selected environment variables
injected, and launches:

```297:315:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/launcher.py
    # tt-run launches one Python engine entrypoint per MPI rank.
    cmd = ["tt-run"]
    cmd.extend(normalized_extra_ttrun_args)
    cmd.extend(["--rank-binding", tmp_rb_path])
    if mpi_args:
        cmd.extend(["--mpi-args", mpi_args])
    cmd.extend(
        [
            sys.executable,
            "-m",
            "vllm_tt_plugin.launcher",
```

The remote entrypoint runs under MPI, derives its DP rank from the MPI rank, and
starts a normal vLLM engine core:

```360:388:/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/launcher.py
    has_mpi = "OMPI_COMM_WORLD_SIZE" in os.environ or "PMI_SIZE" in os.environ
    mpi_rank = int(
        os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0"))
    )
    ...
    pc.data_parallel_rank = mpi_rank * segment
    pc.data_parallel_rank_local = 0
    assert pc.distributed_executor_backend == "uni", (
        "TT MPI must be used with uniproc executor backend"
    )
```

For Llama 3 8B, the thing to watch is divisibility and mesh shape:

- Local T3K: `--data_parallel_size` should divide 8 because submeshes are carved
  out of the local `(1, 8)` mesh.
- Local N300: `DP=2` is the natural split.
- Local N150: `DP=1` is the natural/supported split.
- Multi-host MPI: `data_parallel_size` must be divisible by the number of
  device MPI ranks in the rank-binding file. Whether a given N150/N300/T3K
  multi-host shape works depends on the distributed mesh that tt-metal exposes
  under `tt-run`; vLLM's launcher just stages config, assigns DP ranks, and
  keeps collectives aligned.

### Concrete run shapes

These are the shapes I would use as starting points:

```bash
# N150, single replica. Must cap context on Wormhole N150.
MESH_DEVICE=N150 \
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max_model_len 32768 \
  --additional-config '{"tt": {"sample_on_device_mode": "decode_only"}}'
```

```bash
# N300, two local DP replicas, one device each.
MESH_DEVICE=N300 \
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data_parallel_size 2 \
  --max_model_len 32768 \
  --async-scheduling \
  --additional-config '{"tt": {"sample_on_device_mode": "decode_only"}}'
```

```bash
# T3K, four local DP replicas, two devices each.
MESH_DEVICE=T3K \
python plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --data_parallel_size 4 \
  --async-scheduling \
  --additional-config '{"tt": {"sample_on_device_mode": "decode_only"}}'
```

For max throughput on T3K, also try `--data_parallel_size 2` and `8`. `DP=8`
maximizes independent replicas but gives each replica only one device, which
brings back the N150-style context/token budget. `DP=2` gives each replica four
devices, which may be better if the workload is longer-context or more
compute-bound.

### Installing vLLM with tt-metal and starting a server

The install flow is intentionally asymmetric: tt-metal owns the Python
environment and hardware runtime, while this vLLM fork installs into that same
environment as an `empty` target plus the TT plugin. Do not install the PyPI
`vllm` wheel into a separate env for this path.

From a built tt-metal checkout:

```bash
export TT_METAL_HOME=/localdev/gwang/vllm_duo/tt-metal-too
export VLLM_DIR=/localdev/gwang/vllm_duo/vllm

cd "$TT_METAL_HOME"
source python_env/bin/activate
export PYTHONPATH="$TT_METAL_HOME:$VLLM_DIR:${PYTHONPATH:-}"

cd "$VLLM_DIR"
source plugins/vllm-tt-plugin/docs/install-vllm-tt.sh
```

That script is deliberately small:

```bash
VLLM_TARGET_DEVICE=empty uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install -e plugins/vllm-tt-plugin
```

`VLLM_TARGET_DEVICE=empty` is a build-time choice only. At runtime the TT
platform activates when `ttnn` is importable and the plugin entry points are
registered:

```bash
python -c "import ttnn; print('ttnn ok')"
python -c "import vllm_tt_plugin; print(vllm_tt_plugin.__file__)"
```

If `VLLM_PLUGINS` is set in the environment, it must allow both TT entry
points:

```bash
export VLLM_PLUGINS=tt,tt_model_registry
```

For Llama 3.1 8B, use the same model ID for both vLLM and tt-metal's
`HF_MODEL`. Setting `TT_CACHE_PATH` is optional but keeps TTNN weight caches out
of the HF hub cache:

```bash
cd /localdev/gwang/vllm_duo/vllm
source /localdev/gwang/vllm_duo/tt-metal-too/python_env/bin/activate

export PYTHONPATH="/localdev/gwang/vllm_duo/tt-metal-too:/localdev/gwang/vllm_duo/vllm:${PYTHONPATH:-}"
export HF_MODEL=meta-llama/Llama-3.1-8B-Instruct
export TT_LLAMA_TEXT_VER=tt_transformers
export VLLM_RPC_TIMEOUT=300000
export TT_CACHE_PATH=/mnt/MLPerf/huggingface/tt_cache/meta-llama--Llama-3.1-8B-Instruct
```

The validated server entrypoint is the TT plugin wrapper, not plain
`vllm serve`, because it injects TT-friendly defaults such as `--block_size 64`:

```bash
MESH_DEVICE=T3K \
python plugins/vllm-tt-plugin/examples/server_example_tt.py \
  --model "$HF_MODEL" \
  --async-scheduling \
  --additional-config '{"tt": {"fabric_config": "FABRIC_1D", "sample_on_device_mode": "all", "trace_region_size": 85000000}}'
```

For an N150 smoke test, cap context explicitly because the Llama 8B bridge
rejects longer `max_model_len` on one Wormhole device:

```bash
MESH_DEVICE=N150 \
python plugins/vllm-tt-plugin/examples/server_example_tt.py \
  --model "$HF_MODEL" \
  --max_model_len 32768 \
  --max_num_seqs 1 \
  --additional-config '{"tt": {"sample_on_device_mode": "decode_only"}}'
```

Basic health and completion checks:

```bash
curl -sf http://localhost:8000/health

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "San Francisco is a",
    "max_tokens": 32,
    "temperature": 1,
    "top_p": 0.9,
    "top_k": 10
  }'
```

Once the server is up, the plugin's server-facing tests target that running
OpenAI-compatible endpoint:

```bash
pytest plugins/vllm-tt-plugin/tests/tt -v \
  --tt-server-url=http://localhost:8000 \
  --tt-model-name=meta-llama/Llama-3.1-8B-Instruct
```

The important failure modes are usually environment mismatches: `ttnn` not
importable, tt-metal not on `PYTHONPATH`, PyPI vLLM replacing the local `empty`
build, missing Hugging Face access for Meta Llama weights, or `MESH_DEVICE`/DP
combinations that do not divide the physical mesh.

### Reported Llama 3.1 8B vLLM server perf from PRs

The numbers below are the latest exact Llama 3.1 8B vLLM results I found in
`tenstorrent/vllm` PR descriptions or attached PR screenshots. They are server
benchmarks unless noted otherwise; do not mix them directly with tt-metal
standalone `PERF.md` demo numbers.

| Device | Source | Setup | Reported result |
| --- | --- | --- | --- |
| `N150` | [tenstorrent/vllm#337](https://github.com/tenstorrent/vllm/pull/337) | `vllm bench serve`, `meta-llama/Llama-3.1-8B-Instruct`, random input len `2`, output len `1024`, max concurrency `32`, prompts `320`; comparing async decode scheduling off vs on. | Async improved decode cadence but hurt admission latency: mean TPOT `55.58 -> 50.76 ms`, mean ITL `55.53 -> 50.71 ms`, P99 ITL `130.09 -> 58.2 ms`; mean TTFT worsened `2985.9 -> 3505.14 ms`, P99 TTFT worsened `3422.45 -> 6042.57 ms`. |
| `N150` | [tenstorrent/vllm#171](https://github.com/tenstorrent/vllm/pull/171) | Llama 3.1 8B on `N150`, host-side sampling vs force-enabled compatibility sampling. | Regular host-side sampling measured `24.0 t/s/u`; force-enabled compatibility sampling measured `22.8 t/s/u` (~5% slower). |
| `N300` | [tenstorrent/vllm#346](https://github.com/tenstorrent/vllm/pull/346) | `vllm bench serve`, `meta-llama/Llama-3.1-8B-Instruct`, random input len `2`, output len `256`, max concurrency `32`, prompts `80`; server config used `sample_on_device_mode=decode_only` and `trace_mode=decode_only`; comparing no async vs async branch. | Async improved both TTFT and TPOT in this run: mean TTFT `1546 -> 1431 ms`, P99 TTFT `1984 -> 1796 ms`, mean TPOT `63.84 -> 60.89 ms`, P99 TPOT `71.28 -> 68.26 ms`. |
| `T3K` | No exact PR-reported server result found | PRs mention T3K bring-up/testing for Llama 8B, but I did not find an exact `vllm bench serve` result for Llama 3.1 8B on `T3K` in PR text or screenshots. | Treat the T3K commands above as the bring-up shape; collect a fresh `vllm bench serve` result before using T3K in a perf comparison. |

One non-server result is still useful for prefix-cache expectations:
[tenstorrent/vllm#272](https://github.com/tenstorrent/vllm/pull/272) reports
`benchmark_prefix_caching.py` on Llama 3.1 8B `N300` with input lengths
`128:1024`: `prefill_sec` improved as prompt reuse increased, from `0.243`
at repeat count `1`, to `0.152` at `2`, `0.106` at `4`, and `0.091` at `8`.

---

## Additional Llama 3 8B Feature Deep Dives

This section keeps Llama 3 8B as the focal model and separates:

- what vLLM owns,
- what the TT plugin owns,
- what the tt-metal bridge/model must implement,
- and which vLLM features are currently disabled or only partially supported.

The default Llama 3 8B route is still:

```text
HF arch:      LlamaForCausalLM
TT arch:      TTLlamaForCausalLM
TT class:     models.tt_transformers.tt.generator_vllm:LlamaForCausalLM
Base runtime: models.tt_transformers.tt.generator:Generator
```

The registration path is selected in the plugin:

```227:245:plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py
    llama_text_version = os.getenv("TT_LLAMA_TEXT_VER", "tt_transformers")
    if llama_text_version == "tt_transformers":
        path_llama_text = "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    ...
    _register_model_if_missing(ModelRegistry, "TTLlamaForCausalLM", path_llama_text)
```

The Llama bridge opts into the three important vLLM-facing capabilities:

```627:633:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/generator_vllm.py
class LlamaForCausalLM(Generator):
    # Class-level capabilities
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }
```

### Prefix caching

For Llama 3 8B, prefix caching is supported.

The TT plugin enables prefix caching only when the resolved TT model declares
`supports_prefix_caching`, and disables it for sliding-window models:

```646:671:plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py
        if vllm_config.cache_config.enable_prefix_caching:
            supports_prefix_caching = (
                model_capabilities.get("supports_prefix_caching", False)
                if model_capabilities
                else False
            )

            if not supports_prefix_caching:
                vllm_config.cache_config.enable_prefix_caching = False
                ...
            else:
                uses_sliding_window = (
                    vllm_config.model_config.get_sliding_window() is not None
                )
                if uses_sliding_window:
                    vllm_config.cache_config.enable_prefix_caching = False
```

The division of labor is:

- vLLM owns prefix-cache lookup, block reuse, block IDs, and
  `num_computed_tokens`.
- The TT plugin forwards the current `page_table` (cached/reused prefix blocks
  plus newly allocated blocks) plus `start_pos=input_batch.num_computed_tokens_cpu`.
- tt-metal Llama must execute only the uncached suffix while addressing the full
  KV cache correctly.

vLLM's prefix-cache manager finds full cached blocks and returns the number of
tokens already computed:

```164:204:vllm/v1/core/kv_cache_manager.py
    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        ...
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0
        ...
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
        ...
        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens
```

The TT runner then turns vLLM's cached-token count into `start_pos` for prefill:

```917:925:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
            # num_computed_tokens for each request is the input position
            # (=computed previously and cached)
            input_positions = input_batch.num_computed_tokens_cpu[req_indices]
            # Prefill length in tokens for each request:
            # - For new requests: equals prompt length.
            # - For resumed-from-preemption requests: includes any generated
            #   output tokens so far, so we can replay the full sequence to
            #   rebuild KV after preemption freed the cache blocks.
            prompt_lens = input_batch.num_tokens[req_indices]
```

That is passed to tt-metal as `start_pos`:

```2328:2335:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
        kwargs = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": self.kv_caches,
            "enable_trace": self.trace_mode in ["all"],
            "prompt_lens": model_input.prompt_lens,
            "start_pos": model_input.input_positions,
        }
```

The Llama bridge simply forwards to the base generator:

```702:706:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/generator_vllm.py
    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)
```

The actual prefix-cache behavior lives in `Generator.prefill_forward_text`.
It treats `tokens` and `prompt_lens` as full prompt state, derives the cached
prefix length from `start_pos`, and pads only the uncached suffix:

```528:610:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/generator.py
    def prefill_forward_text(
        self,
        tokens: torch.Tensor,  # All tokens, including the cached ones
        ...
        prompt_lens=None,  # Full prompt lengths, including the cached ones
        ...
        start_pos: list[int] = None,  # Cached prefixes lengths
        ...
        num_cached_per_user = [int(n) for n in start_pos] if start_pos is not None else [0] * len(prompt_lens)
        ...
        prefill_seq_lens = [
            get_padded_prefill_len(seq_len - num_cached)
            for seq_len, num_cached in zip(prompt_lens, num_cached_per_user)
        ]
```

For the non-batched Llama path, only the uncached token suffix is sent into the
prefill kernel:

```749:755:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/generator.py
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
                prefill_ids = torch.cat(
                    [
                        tokens[idx : idx + 1, num_cached_tokens:seq_len],
                        torch.zeros(1, prefill_seq_len - (seq_len - num_cached_tokens)).long(),
                    ],
```

For traced APC, tt-metal keeps the full page table for addressing but creates a
chunk page table over the uncached block range:

```421:458:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/generator.py
        num_cached_tokens=0,
        ...
        use_prefix_caching = num_cached_tokens > 0
        chunk_start_idx = num_cached_tokens
        ...
        source_page_table = full_page_table if full_page_table is not None else page_table
        ...
        if batch_size == 1:
            if use_prefix_caching:
                chunk_start_block = num_cached_tokens // block_size
                chunk_end_block = num_blocks_in_seq(num_cached_tokens + prefill_seq_len, block_size)
                chunk_page_table = source_page_table[:, chunk_start_block:chunk_end_block]
```

So for Llama 3 8B, prefix caching does not require a separate tt-metal API. It
requires the existing prefill contract to correctly honor `start_pos`,
`prompt_lens`, and the page table.

### Chunked prefill

vLLM chunked prefill is explicitly disabled for TT.

`TTPlatform.check_and_update_config` flips `enable_chunked_prefill` off and may
bump `max_num_batched_tokens` to `max_model_len`:

```455:475:plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py
        if vllm_config.scheduler_config.enable_chunked_prefill:
            logger.info("Chunked prefill is not yet supported for TT backend")
            vllm_config.scheduler_config.enable_chunked_prefill = False
            ...
            if (
                vllm_config.scheduler_config.max_num_batched_tokens
                < vllm_config.model_config.max_model_len
            ):
                ...
                vllm_config.scheduler_config.max_num_batched_tokens = (
                    vllm_config.model_config.max_model_len
                )
```

`TTModelRunner` asserts that the feature stays disabled:

```147:148:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
        # Currently, TT model runner doesn't support chunked prefill.
        assert self.scheduler_config.enable_chunked_prefill is False
```

The TT scheduler invariant is also explicit:

```30:37:plugins/vllm-tt-plugin/src/vllm_tt_plugin/scheduler.py
class TTScheduler(AsyncScheduler):
    """Scheduler for the TT (Tenstorrent) platform.

    TT constraints:
    - No mixed prefill+decode batches: each batch is either all-prefill
      or all-decode.
    - No chunked prefill: each prefill must be scheduled in full.
```

Important distinction: tt-metal Llama has internal long-prefill chunking, but
that is not vLLM chunked prefill.

vLLM chunked prefill is a scheduler contract: a prompt can be split across
multiple engine iterations, and each iteration's `num_scheduled_tokens` may be
somewhere between full prompt prefill and single-token decode:

```43:49:vllm/v1/core/sched/interface.py
        Essentially, the scheduler produces a dictionary of {req_id: num_tokens}
        that specifies how many tokens to process for each request.
        ...
        Otherwise, it can be somewhere in between in case of chunked prefills,
        prefix caching, speculative decoding, etc.
```

tt-metal's internal chunking happens inside one full prefill call. The runner
still schedules the whole prompt; the generator may break that full prompt into
device-sized chunks for memory/trace reasons:

```636:648:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/generator.py
        # Batched prefill: all prompts share the same padded length so they can
        # be processed in a single forward pass.
        ...
        use_batched_prefill = (
            batch_size > 1
            and len(set(prefill_seq_lens)) == 1
            and self.data_parallel == 1
            ...
            and all(
                n == 0 for n in num_cached_per_user
            )  # batched path feeds full tokens; incompatible with cached prefixes
        )
```

When the prompt is too large for the batched path, Llama falls back to the
sequential per-user path that can internally chunk correctly:

```663:669:/localdev/gwang/vllm_duo/tt-metal-too/models/tt_transformers/tt/generator.py
        if use_batched_prefill and any(s > self.model_args[0].max_prefill_chunk_size for s in prefill_seq_lens):
            logger.info(
                f"Batched prefill disabled: padded prefill len {prefill_seq_lens[0]} exceeds "
                f"max_prefill_chunk_size {self.model_args[0].max_prefill_chunk_size}; chunked "
                f"prefill requires the sequential prefill path (#45234)"
            )
            use_batched_prefill = False
```

To support true vLLM scheduler-level chunked prefill for Llama 3 8B, TT would
need more than a config flip:

- The TT scheduler would need to permit partial-prefill scheduling while still
  respecting TT's no mixed prefill/decode execution constraint.
- `TTModelRunner.build_model_input` would need to build partial-prefill inputs
  from `num_scheduled_tokens`, not always full prompt slices.
- The tt-metal Llama bridge would need a stable repeated-prefill-chunk contract:
  where to write each chunk in paged KV, when to emit logits, how to handle the
  final chunk, and how prefix-cache hits combine with chunk progress.
- Trace capture/warmup would need to cover the partial chunk shapes that can
  appear at runtime.
- DP gather would need to carry partial-prefill state consistently across ranks.

### Other vLLM features that require tt-metal implementation

The current Llama 3 8B picture is:

| Feature | Llama 3 8B status | What tt-metal/model must provide |
| --- | --- | --- |
| Prefix caching | Supported | Correct `prefill_forward` handling for `start_pos`, `prompt_lens`, `page_table`, and uncached suffix execution. |
| vLLM chunked prefill | Disabled | Partial-prefill scheduling, repeated chunk execution, final-logits semantics, trace coverage, and DP-safe state handling. |
| Async decode | Supported | Split decode submission via `decode_forward(..., read_from_device=False)` plus `read_decode_output(..., async_read=True)`. |
| On-device sampling | Supported for compatible batches | Device sampler for temperature/top-k/top-p/penalties/seed, with host fallback for unsupported sampling params. |
| Device logprobs | Partial | Llama can use host fallback; top-K device logprobs are currently treated as model-specific and only enabled for models like GPT-OSS. |
| Hybrid KV / sliding-window groups | Not active for Llama 3 8B | Hybrid models need `get_kv_cache_spec`, `allocate_kv_cache_per_layer`, and per-layer page-table routing. |
| Tensor/pipeline parallelism | Rejected | TT model must shard internally; vLLM TP/PP are not used. |
| Speculative decoding | Rejected | Draft-token scheduling, lookahead KV allocation, verification, and model/runner semantics. |
| LoRA | Rejected | Adapter loading and application in tt-metal model execution. |
| Prompt logprobs | Rejected | Prefill-time logprob extraction and `prompt_logprobs_dict` population. |
| Pooling/embeddings | Unsupported | Pooling runner outputs and tt-metal embedding/pooler model path. |
| Prompt embeds | Unsupported | Non-token prompt input ingestion through TT input batch and model bridge. |
| Encoder-decoder | Unsupported | Encoder input scheduling, cross-attention KV/cache handling, and bridge APIs. |
| Multimodal non-image inputs | Unsupported | The TT plugin currently validates multimodal features as images only. |

#### Async decode

The platform gate is capability-based:

```608:625:plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py
        # Model-gated async scheduling. Async overlap requires generators that
        # support split decode submission via `decode_forward(...,
        # read_from_device=False)` followed by `read_decode_output(...,
        # async_read=True)`.
        supports_async_decode = (
            model_capabilities.get("supports_async_decode", False)
            if model_capabilities
            else False
        )
        if vllm_config.scheduler_config.async_scheduling and not supports_async_decode:
            ...
            vllm_config.scheduler_config.async_scheduling = False
```

For Llama 3 8B, this is enabled by `model_capabilities`. The model-side
obligation is the split decode/readback contract; the plugin keeps the returned
ttnn events opaque and synchronizes them during finalization.

#### On-device sampling and host fallback

Llama 3 8B declares `supports_sample_on_device`, but the runner only uses device
sampling for compatible batches. Host-only sampling parameters force fallback:

```2270:2298:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
    def check_perform_device_sampling(
        self, is_decode: bool, has_structured_outputs: bool
    ) -> bool:
        want_device_sampling = self.sample_on_device_mode == "all" or (
            self.sample_on_device_mode == "decode_only" and is_decode
        )
        if not want_device_sampling:
            return False
        ...
        has_always_host_only_sampling_params = (
            not input_batch.no_allowed_token_ids
            or input_batch.sampling.bad_words_token_ids
            or input_batch.sampling.has_active_logitsprocs()
            or bool(self.model_config.logits_processors)
        )
        if has_always_host_only_sampling_params:
            return False

        # Structured outputs are not supported on device yet
        if has_structured_outputs:
            return False
```

Logprobs also restrict device sampling. Single-device setups fall back to host
for any logprobs, and top-K device logprobs require model-specific support:

```2300:2314:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
        # Logprobs on device require multi-device setups (num_devices in {8,32}).
        # On single device, all logprobs require host sampling.
        ...
        max_lp = input_batch.max_num_logprobs
        if max_lp is not None:
            if num_devices not in (8, 32):
                return False
            if max_lp > 0 and not self.supports_topk_logprobs:
                return False
```

When device sampling is not used, TT returns logits and the plugin runs vLLM's
host sampler:

```2630:2750:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
            if not perform_device_sampling:
                logits = tt_out[rows, -1, :]
                ...
                sampler_output = self.host_sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )
                next_token_ids = sampler_output.sampled_token_ids
```

#### Prompt logprobs

Prompt logprobs are rejected during request validation:

```693:705:plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py
    def validate_request(
        cls,
        prompt: "PromptType | DictPrompt | TokPrompt",
        params: "SamplingParams | PoolingParams",
        processed_inputs: "ProcessorInputs",
    ) -> None:
        ...
        if isinstance(params, SamplingParams) and params.prompt_logprobs is not None:
            raise ValueError(f"Not yet supporting prompt_logprobs on {dev}")
```

The runner output path currently fills prompt logprobs with `None`:

```2843:2855:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = dict.fromkeys(
            (output_req_ids[i] for i in range(num_reqs)), None
        )
        ...
        return ModelRunnerOutput(
            ...
            prompt_logprobs_dict=prompt_logprobs_dict,
```

Supporting prompt logprobs would require tt-metal Llama to return or expose
prefill token logits/logprobs for prompt positions, not just the final sampling
position, and the plugin would need to populate `prompt_logprobs_dict`.

#### Hybrid KV and per-layer page tables

This is not active for Llama 3 8B. Llama stays on the legacy single-group KV
path: one page table is broadcast to all attention layers.

The TT worker uses a model hook only if the resolved TT class exposes
`get_kv_cache_spec`; otherwise it falls back to one homogeneous spec:

```167:202:plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py
    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        ...
        spec_from_hook = self._try_get_spec_from_model_hook()
        if spec_from_hook is not None:
            return spec_from_hook

        return self._build_default_kv_cache_spec()
```

Hybrid models must additionally support per-layer cache allocation:

```429:459:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
    def _allocate_kv_caches(self, kv_cache_config: KVCacheConfig) -> Any:
        ...
        if hasattr(self.model, "allocate_kv_cache_per_layer"):
            return self.model.allocate_kv_cache_per_layer(per_layer_specs)
        ...
        return self.model.allocate_kv_cache(shape, dtype, len(per_layer_specs))
```

For Llama 3 8B, do not chase this path unless the model class changes to expose
`get_kv_cache_spec`.

#### Hard rejects and unsupported request types

The main config-time rejects are:

```477:484:plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py
        assert not vllm_config.speculative_config, (
            "Speculative decoding is not yet supported for TT backend"
        )
        assert (
            vllm_config.parallel_config.tensor_parallel_size == 1
            and vllm_config.parallel_config.pipeline_parallel_size == 1
        ), "TT backend does not support distributed execution"
        assert not vllm_config.lora_config, "LoRA is not supported for TT backend"
```

Pooling and embedding tasks are not exposed:

```269:276:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        # TT backend currently supports text generation only.
        # (No transcription support yet.)
        return ["generate"]

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        # TT backend does not support pooling/embedding tasks yet.
        return []
```

Prompt embeds are rejected when request state is built:

```51:64:plugins/vllm-tt-plugin/src/vllm_tt_plugin/input_batch.py
def build_cached_request_state(new_req_data) -> CachedRequestState:
    ...
    assert new_req_data.sampling_params is not None, (
        "Pooling is not supported for TT yet"
    )
    if new_req_data.prompt_token_ids is None:
        raise NotImplementedError("TT backend does not support prompt_embeds yet")
```

Encoder-decoder models are rejected when the runner initializes:

```138:139:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
        if self.model_config.is_encoder_decoder:
            raise ValueError("Encoder-decoder models aren't yet supported for TT")
```

Multimodal support is image-only at the plugin validation point:

```705:708:plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py
    def _validate_mm_feature(self, mm_feature: MultiModalFeatureSpec) -> None:
        """Validate the multimodal feature is an image."""
        if mm_feature.modality != "image":
            raise NotImplementedError("Only images are supported for now")
```

### Bottom line for Llama 3 8B

Llama 3 8B already has the key TT bridge pieces for standard generation,
prefix caching, async decode, KV paging, and compatible on-device sampling.

The biggest missing vLLM feature for this focal model is true scheduler-level
chunked prefill. The rest of the major gaps are either host-fallback sampling
cases or larger tt-metal runtime/model capabilities: LoRA, speculative decoding,
prompt logprobs, pooling/embeddings, prompt embeds, encoder-decoder, and broader
multimodal input support.

---

## 10. Fresh Llama 3.1 8B measurements on N150, N300, and T3K

Measurements collected on 2026-07-09 using:

- vLLM repo: `/localdev/gwang/vllm_duo/vllm`
- tt-metal repo: `/localdev/gwang/vllm_duo/tt-metal-too`
- `HF_HOME=/proj_sw/user_dev/huggingface`
- `HF_MODEL=meta-llama/Llama-3.1-8B-Instruct`
- `TT_LLAMA_TEXT_VER=tt_transformers`
- `TT_CACHE_PATH=/localdev/gwang/vllm_duo/tt_cache/meta-llama--Llama-3.1-8B-Instruct`

Result artifacts live under:

```text
/localdev/gwang/vllm_duo/perf_results/llama3_8b_20260709T020822Z
```

### Workload

All valid measurements used the OpenAI-compatible completions endpoint:

- Endpoint: `/v1/completions`
- Dataset: random
- Random input length: 2
- Random output length: 256
- Prompts: 320
- Request rate: `inf`
- Max concurrency: 32
- `--ignore-eos`
- Client temperature: `0`
- Metrics: TTFT, TPOT, ITL, E2EL at p50/p90/p95/p99

### Important caveat: on-device sampling is currently broken here

The intended `sample_on_device_mode=decode_only` configuration crashed on first
request, so the valid numbers below are **host-sampling fallback** measurements:

```json
{"tt":{"trace_mode":"decode_only"}}
```

That is, `sample_on_device_mode` was omitted / effectively `none`.

### Valid no-sample results

Use these newly measured no-sample results as the current performance targets
for this checkout. The older PR-reported numbers above are useful historical
context, but they are incomplete across devices and not uniformly comparable
across workload shape, prompt count, and sampling mode.

| Device | Server config | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Peak output tok/s | Mean TTFT ms | P95 TTFT ms | Mean TPOT ms | P95 TPOT ms | P99 TPOT ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N150 | DP1, `max_model_len=32768`, `max_num_seqs=32` | 320 | 0 | 131.93 | 2.43 | 620.92 | 623.34 | 794 | 2115.17 | 7365.37 | 43.44 | 43.55 | 47.95 |
| N300 | DP2, `max_model_len=32768`, `max_num_seqs=16` | 320 | 0 | 191.28 | 1.67 | 428.27 | 429.95 | 592 | 3196.13 | 3438.46 | 57.89 | 66.21 | 68.68 |
| T3K | DP4, `max_model_len=131072`, `max_num_seqs=8`, `fabric_config=FABRIC_1D` | 320 | 0 | 242.03 | 1.32 | 338.47 | 339.79 | 408 | 2364.95 | 3784.99 | 85.63 | 90.07 | 97.10 |

Artifacts:

- N150: `n150_no_sample/result.json`, `n150_server_no_sample.log`
- N300: `n300_no_sample/result.json`, `n300_server_no_sample.log`
- T3K: `t3k_no_sample/result.json`, `t3k_server_no_sample.log`

T3K successfully initialized an 8-device mesh with `FABRIC_1D`. The benchmark
client emitted TTNN nanobind leak warnings at shutdown after writing the T3K
result file; the benchmark itself completed with 0 failed requests.

### Coherence smoke

Prompt:

```text
Explain why rain can make roads slippery in three sentences.
```

Smoke results:

- N150: HTTP 200; coherent and on topic. The output repeated near the end
  because the request used `max_tokens=80` with no stop condition.
- N300: HTTP 200; coherent and on topic. Same token-cap/no-stop repetition.
- T3K: HTTP 200; coherent and on topic. It ended mid-word at the token cap.

Coherence artifacts:

- `n150_no_sample_single_completion.json`
- `n300_no_sample_single_completion.json`
- `t3k_no_sample_single_completion.json`

### On-device sampling crash details

The failing N150 configuration was:

- `MESH_DEVICE=N150`
- `max_model_len=32768`
- `max_num_seqs=32`
- `additional_config={"tt":{"sample_on_device_mode":"decode_only","trace_mode":"decode_only"}}`
- Runner log: `TTModelRunner: trace_mode=decode_only, sample_on_device_mode=decode_only, enable_model_warmup=True`

The crash reproduced in three contexts:

- Benchmark run with async scheduling enabled: `n150_server_bench.log`
- Single request after server reset through `/v1/chat/completions`: `n150_server_reset.log`
- Single `/v1/completions` request without explicitly passing `--async-scheduling`: `n150_server_noasync.log`

The server initialized, warmed up, and accepted the first request. The most
visible Python exception was:

```text
File "/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/worker.py", line 357, in sample_tokens
    return self.model_runner.sample_tokens(grammar_output)
File "/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py", line 2229, in sample_tokens
    finish = self._pending_samples.popleft()
IndexError: pop from an empty deque
```

The API layer then reported:

```text
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
```

Benchmark-visible impact:

- `n150/result.json` recorded 0 completed requests and 320 failed requests.
- Early client errors were `Never received a valid chunk to calculate TTFT. This response will be marked as failed!`
- Later client errors were connection-refused failures after the engine/server exited.
- The `n150/result.json` throughput and latency values are invalid and should not be compared with the no-sample measurements.

Because `n150_server_noasync.log` hit the same `sample_tokens` /
`_pending_samples.popleft()` exception, this is not just an explicit
`--async-scheduling` benchmark artifact.

A later re-check showed that the empty deque is a secondary/masking exception,
not the first failure. In all N150 repro logs (`n150_server_bench.log`,
`n150_server_reset.log`, `n150_server_noasync.log`, and
`sampling_debug_20260709T030000Z/n150_decode_only_patch_server.log`), the line
immediately before the engine dump is the TT runtime failure:

```text
TT_THROW: Statically allocated circular buffers in program 122 clash with L1 buffers on core range [0-0 - 7-7]. L1 buffer allocated at 1301184 and static circular buffer region ends at 1375296
```

The vLLM V1 engine schedules `sample_tokens()` based on
`scheduler_output.total_num_scheduled_tokens`, not on whether TT
`execute_model()` successfully queued a pending sample. When the TT prefill path
throws before appending to `_pending_samples`, the follow-up `sample_tokens()`
call underflows the deque and obscures the original L1 circular-buffer clash.
So the N150 on-device sampling crash has two parts:

- Primary failure: TT runtime L1 circular-buffer clash during the first request's
  prefill/warmup path.
- Secondary failure: TT runner does not preserve/report the `execute_model()`
  failure before `sample_tokens()` is called, so the log ends with
  `IndexError: pop from an empty deque`.

### Nightly sampling-test reference

The tt-metal nightly workflow has a useful comparison point in:

```text
/localdev/gwang/vllm_duo/tt-metal-too/.github/workflows/vllm-nightly-tests-impl.yaml:244-261
```

That matrix entry is:

```text
[WH-T3K] Llama-3.1-8B-Instruct with sampling-tests
model: meta-llama/Llama-3.1-8B-Instruct
mesh-device: T3K
tt-llama-text-ver: tt_transformers
tt-config: {"fabric_config": "FABRIC_1D", "sample_on_device_mode": "all", "trace_region_size": 85000000}
additional-server-args: --async-scheduling
structured-output: true
sampling-tests: true
```

This does not match the failing local repro exactly. The nightly path validates
T3K with `sample_on_device_mode=all`; the local crash above was reproduced with
`sample_on_device_mode=decode_only`, primarily on N150. That makes the nightly
entry a good next isolation baseline:

1. First try the nightly T3K `sample_on_device_mode=all` server shape.
2. Run a small completion smoke and the plugin sampling tests against it.
3. Change only `sample_on_device_mode` from `all` to `decode_only`.

If `all` passes and `decode_only` fails, the bug is likely specific to the
decode-only sampling transition. If both fail in this checkout, the regression is
broader than the mode switch or depends on the local vLLM/tt-metal environment.

### Follow-up sampling debug experiments

Debug logs were written under:

```text
/localdev/gwang/vllm_duo/perf_results/sampling_debug_20260709T030000Z
```

T3K, nightly-style `sample_on_device_mode=all`:

- Config: `MESH_DEVICE=T3K`, `--async-scheduling`,
  `{"tt":{"fabric_config":"FABRIC_1D","sample_on_device_mode":"all","trace_region_size":85000000}}`
- Before the local runner patch, the first completion returned HTTP 200 with an
  internal engine error. The underlying stack was:

```text
RuntimeError: "index_cpu" not implemented for 'UInt32'
plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:2628 in _take
plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py:2757 in _get_output_tokens
```

- Narrow local patch tested:

```python
next_token_ids = (
    _take(tt_out.to(torch.int64)).reshape(sz).to(torch.int32)
)
```

- With that patch, T3K `all` passed a one-request completion smoke. Output was
  coherent:

```text
Rain can make roads slippery because the water on the road surface reduces the friction between the tires of a vehicle and the road...
```

- With that patch, T3K `all` also survived a tiny diagnostic benchmark:
  8 successful requests, 0 failed requests, 128 generated tokens.

T3K, `sample_on_device_mode=decode_only` with the same nightly shape except
`all -> decode_only`:

- Config:
  `{"tt":{"fabric_config":"FABRIC_1D","sample_on_device_mode":"decode_only","trace_region_size":85000000}}`
- Runner reported `TTModelRunner: trace_mode=all, sample_on_device_mode=decode_only`.
- The same one-request completion smoke passed with coherent output and no
  `pop from an empty deque`, `index_cpu`, or engine-dead errors.

N150, original failing repro shape after the local dtype patch:

- Config:
  `{"tt":{"sample_on_device_mode":"decode_only","trace_mode":"decode_only"}}`
- Runner reported `TTModelRunner: trace_mode=decode_only, sample_on_device_mode=decode_only`.
- The first completion still failed. The visible API response was an internal
  engine error, and the Python tail again ended in `_pending_samples.popleft()`.
- The root signal immediately before that was the TT runtime L1
  circular-buffer clash shown above. This confirms that the local dtype patch
  fixes the T3K device-sampled token readback issue but does not fix the N150
  L1 allocation failure.

A final attempted N150 isolation with `sample_on_device_mode=all` could not
produce a useful sampling signal because device startup failed first with:

```text
RuntimeError: ARC startup error at core 0-10 over NOC0: scratch_status=0xdeadc0de, postcode=0xc0defffc
```

Treat that as post-crash hardware/driver state, not evidence about
`sample_on_device_mode=all` on N150.

I later retried the exact T3K working TT config on N150 and N300:

```text
{"tt":{"fabric_config":"FABRIC_1D","sample_on_device_mode":"all","trace_region_size":85000000}}
```

For N150, the server command also included `--max-model-len 32768`, because the
Llama 8B bridge caps N150 context. Both attempts failed before mesh open, again
at `ttnn.get_num_devices()`, with the same ARC startup error:

```text
RuntimeError: ARC startup error at core 0-10 over NOC0: scratch_status=0xdeadc0de, postcode=0xc0defffc
```

Logs:

- `sampling_debug_20260709T030000Z/n150_all_t3k_config_server.log`
- `sampling_debug_20260709T030000Z/n300_all_t3k_config_server.log`

So `sample_on_device_mode=all` remains unvalidated on N150 and N300 in this
session. The failed attempts indicate the hardware/driver was still in a bad
post-crash ARC state, not that the T3K working config is invalid on those
meshes.

### Existing CI coverage for N150/N300

The vLLM nightly workflow does not currently cover Llama 3.1 8B on N150 or N300.
The active Llama 3.1 8B vLLM entries are T3K, Wormhole Galaxy, and Blackhole:

- `vllm-nightly-tests-impl.yaml:245-261`: `[WH-T3K] Llama-3.1-8B-Instruct with sampling-tests`
- `vllm-nightly-tests-impl.yaml:263-276`: `[WH-T3K] Llama-3.1-8B-Instruct (prefix caching)`
- `vllm-nightly-tests-impl.yaml:309-322`: `[WH-GLX] Llama-3.1-8B-Instruct DP=4`
- Blackhole entries are also present at `vllm-nightly-tests-impl.yaml:279-306`.

The only N150 vLLM entry I found in that matrix is Gemma, not Llama 3.1 8B:

```text
vllm-nightly-tests-impl.yaml:138-149
[WH-N150] Gemma4-E2B
model: google/gemma-4-E2B-it
mesh-device: N150
```

There is non-vLLM tt-metal / tt_transformers coverage for Llama 3.1 8B on N150:

- `models-t1-unit-tests.yaml` exposes `llama3.1-8b` and `wh_n150`; the actual
  pipeline command in `tests/pipeline_reorg/models_unit_tests.yaml` runs
  `models/tt_transformers/tests/test_embedding.py`, `test_rms_norm.py`,
  `test_mlp.py`, `test_attention.py`, `test_attention_prefill.py`,
  `test_decoder.py`, and `test_decoder_prefill.py`.
- `models-t1-e2e-tests.yaml` exposes `llama3.1-8b` and `wh_n150`; the actual
  pipeline command in `tests/pipeline_reorg/models_e2e_tests.yaml` runs
  `models/tt_transformers/demo/simple_text_demo.py` token-matching and eval-32
  tests.
- `models-t1-device-perf-tests.yaml` exposes `llama3.1-8b` and `wh_n150`; the
  actual pipeline command in `tests/pipeline_reorg/models_device_perf_tests.yaml`
  runs `models/tt_transformers/tests/test_device_perf.py` prefill/decode cases.

I did not find an active Llama 3.1 8B N300 vLLM entry. I also did not find an
active Llama 3.1 8B N300 tt_transformers pipeline leg in the
`tests/pipeline_reorg` YAML, despite `models/model_targets.yaml` having a
`wh_n300` target entry for `llama3.1-8b`. The workflow UI exposes `wh_n300` as a
generic SKU choice, but the model-specific matrix entries decide whether a job
actually exists.

For regression coverage, the existing nightly sampling suite command is useful
when pointed at a live server:

```text
pytest vllm/plugins/vllm-tt-plugin/tests/tt -v \
  -k "not test_mixed_params_batch and not TestSeedingAndVariety and not test_repetition_penalty_mixed_batch and not test_presence_penalty_mixed_batch and not test_frequency_penalty_mixed_batch" \
  --tt-server-url=http://localhost:8000 \
  --tt-model-name=meta-llama/Llama-3.1-8B-Instruct
```

However, the cheapest targeted regression for this specific failure is a
single live-server completion smoke: `N150`,
`sample_on_device_mode=decode_only`, `trace_mode=decode_only`, `temperature=0`,
`max_tokens=32`, no structured output, and no logprobs. That is enough to catch
the N150 L1 crash and the previous misleading empty-queue tail without pulling
in stochastic sampling-correctness variance.

The local runner patch now also preserves the original exception better: if
`sample_tokens()` is called with an empty `_pending_samples` queue, it logs that
`execute_model` likely failed before enqueue and returns `None`, letting vLLM
core re-raise the original `execute_model()` future exception instead of
masking it as `IndexError: pop from an empty deque`.

---

## TTTv2 Llama 3 8B vLLM bringup status

This section tracks the local TTTv2 Llama 3 8B bringup against the TTTv1 vLLM
target above. The TTTv2 model code lives under:

```text
/localdev/gwang/vllm_duo/tt-metal-too/models/common/models/llama3_8b/
/localdev/gwang/vllm_duo/tt-metal-too/models/common/models/generator.py
```

The known-good non-vLLM TTTv2 reference is the T3K `demo.py` path under:

```text
/localdev/gwang/vllm_duo/tt-metal-too/models/common/tests/demos/llama3_8b/
```

### vLLM adapter work completed

The vLLM plugin now has a selectable registration path for TTTv2 Llama 3 8B:

```text
TT_LLAMA_TEXT_VER=common_llama3_8b
models.common.models.generator:Llama3Generator
```

Local changes made for this path:

- `plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py`
  - Added the `common_llama3_8b` registration path.
  - Added model-capability handling for a required vLLM block size.
  - TTTv2 currently forces `block_size=32`.
- `plugins/vllm-tt-plugin/tests/test_config.py`
  - Added coverage for the registration path and block-size override.
- `/localdev/gwang/vllm_duo/tt-metal-too/models/common/models/generator.py`
  - Added the vLLM-facing adapter methods on `Llama3Generator`:
    `initialize_vllm_model`, `get_max_tokens_all_users`, `prefill_forward`,
    `decode_forward`, `process_decode_output_host`, `allocate_kv_cache`, and
    warmup hooks.
  - Added vLLM model capabilities:
    `supports_prefix_caching=True`, `supports_async_decode=True`,
    `supports_sample_on_device=True`, and `required_block_size=32`.
  - Normalized vLLM tensor inputs (`tokens`, `page_table`, `prompt_lens`,
    `start_pos`) into shapes/dtypes accepted by the TTTv2 generator.
  - Dropped unsupported vLLM kwargs that the TTTv2 path does not consume yet:
    `page_tables_per_layer`, `prompt_tokens`, `output_tokens`, `slot_remap`,
    and `rope_deltas_all_users`.
  - Added the current KV budget policy: `get_max_tokens_all_users()` returns
    `max_model_len`, and `initialize_vllm_model()` uses
    `ceil(max_seq_len / 32) + max_batch_size` blocks.

The adapter currently rejects `tt_data_parallel != 1`. That means it can run on
a T3K mesh, but not in the same DP4 shape used by the TTTv1 T3K baseline.

Verification completed:

```text
python -m py_compile \
  /localdev/gwang/vllm_duo/tt-metal-too/models/common/models/generator.py \
  plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py
```

```text
PYTHONPATH=/localdev/gwang/vllm_duo/vllm:\
/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src:\
/localdev/gwang/vllm_duo/tt-metal-too \
/localdev/gwang/vllm_duo/tt-metal-too/python_env/bin/pytest \
  plugins/vllm-tt-plugin/tests/test_config.py -q
```

Result:

```text
10 passed, 2 warnings
```

### T3K reset and smoke

The first T3K hardware probe failed with an ARC startup error. After an explicit
reset:

```text
tt-smi -r
```

`ttnn.get_num_devices()` reported 8 devices.

The following offline vLLM smoke passed end to end:

```text
PYTHONPATH=/localdev/gwang/vllm_duo/vllm:\
/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src:\
/localdev/gwang/vllm_duo/tt-metal-too \
MESH_DEVICE=T3K \
TT_LLAMA_TEXT_VER=common_llama3_8b \
/localdev/gwang/vllm_duo/tt-metal-too/python_env/bin/python \
  plugins/vllm-tt-plugin/examples/offline_inference_tt.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --max_model_len 1024 \
  --max_num_batched_tokens 1024 \
  --max_seqs_in_batch 1 \
  --max_tokens 4 \
  --num_repeat_prompts 1 \
  --block_size 32 \
  --greedy_sampling \
  --disable_prefix_caching
```

Smoke result:

- 32 prompts completed.
- 4 generated tokens per prompt.
- Approximate output throughput: 18.53 tok/s.
- Outputs were coherent.

An earlier smoke attempt failed because vLLM passed `start_pos` as a numpy
array, while the TTTv2 path expected tensor-like `.dim()`. The adapter now
normalizes those tensor-like kwargs before entering the TTTv2 generator.

### Parity benchmark attempt

The direct TTTv1 parity shape could not be run with TTTv2 yet:

```text
DP4, max_model_len=131072, max_num_seqs=8
```

Reason: the current TTTv2 adapter supports only `tt_data_parallel=1`.

A DP1 attempt with the same `max_model_len=131072` and `max_num_seqs=8` also
failed during model construction with the TTTv2 token-budget guard:

```text
ValueError: Total token budget exceeded: max_batch_size (8) × max_seq_len
(131072) = 1,048,576 tokens, but maximum is 131,072 tokens (128K). Reduce
max_batch_size or max_seq_len to fit in device DRAM.
```

The benchmark below therefore uses the largest validated DP1 shape for this
session:

```text
MESH_DEVICE=T3K
TT_LLAMA_TEXT_VER=common_llama3_8b
max_model_len=16384
max_num_seqs=8
fabric_config=FABRIC_1D
trace_mode=decode_only
trace_region_size=85000000
```

Server command:

```text
PYTHONPATH=/localdev/gwang/vllm_duo/vllm:\
/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src:\
/localdev/gwang/vllm_duo/tt-metal-too \
HF_HOME=/proj_sw/user_dev/huggingface \
TT_CACHE_PATH=/localdev/gwang/vllm_duo/tt_cache/meta-llama--Llama-3.1-8B-Instruct \
VLLM_RPC_TIMEOUT=300000 \
MESH_DEVICE=T3K \
TT_LLAMA_TEXT_VER=common_llama3_8b \
/localdev/gwang/vllm_duo/tt-metal-too/python_env/bin/python \
  plugins/vllm-tt-plugin/examples/server_example_tt.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8002 \
  --host 127.0.0.1 \
  --max-model-len 16384 \
  --max_num_seqs 8 \
  --async-scheduling \
  --additional-config '{"tt":{"fabric_config":"FABRIC_1D","trace_mode":"decode_only","trace_region_size":85000000}}'
```

The server accepted `--async-scheduling` but disabled it because the TTTv2
adapter does not currently declare async decode support:

```text
Async scheduling was requested, but TT model Llama3Generator
(models.common.models.generator) does not declare support
(`model_capabilities['supports_async_decode']`). Disabling async scheduling.
```

Benchmark client workload matched the TTTv1 no-sample target:

```text
vllm bench serve
backend: openai
endpoint: /v1/completions
dataset: random
random input length: 2
random output length: 256
prompts: 320
request rate: inf
max concurrency: 32
ignore eos: true
temperature: 0
```

Result artifacts:

```text
/localdev/gwang/vllm_duo/perf_results/tttv2_llama3_8b_20260709T135631Z/t3k_dp1_no_sample_16k/result.json
/localdev/gwang/vllm_duo/perf_results/tttv2_llama3_8b_20260709T135631Z/t3k_dp1_no_sample_16k/server.log
```

Results:

| Implementation | Server shape | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv1 baseline | T3K DP4, `max_model_len=131072`, `max_num_seqs=8` | 320 | 0 | 242.03 | 1.32 | 338.47 | 339.79 | 2364.95 | 2312.97 | 3792.21 | 85.63 | 84.91 | 97.10 |
| TTTv2 current | T3K DP1, `max_model_len=16384`, `max_num_seqs=8` | 320 | 0 | 616.79 | 0.52 | 132.82 | 133.34 | 44077.40 | 46333.47 | 47270.24 | 59.89 | 59.99 | 62.25 |

The current TTTv2 result is about 39.2% of the TTTv1 baseline output
throughput:

```text
132.82 / 338.47 = 0.392
```

Interpretation:

- The TTTv2 vLLM path is functional enough to serve the full benchmark workload:
  320 successful requests, 0 failed requests.
- This is not a true parity run because TTTv2 is still limited to DP1 and the
  validated server context was 16K, not the TTTv1 DP4/131K shape.
- TTTv2 per-token decode latency is lower in this run (`mean_tpot_ms=59.89`)
  than the TTTv1 baseline (`85.63`), but request-level latency and throughput
  are worse because only 8 requests are actively running on one DP lane while
  the benchmark queues up to 32 concurrent requests.
- The next performance levers are async decode support and DP4 support. Async
  decode can be investigated within the current DP1 shape. DP4 requires a
  separate design pass for TTTv2 mesh/submesh or lane handling.

### Async decode support investigation

The TT vLLM plugin enables async scheduling only when the model advertises:

```python
model_capabilities["supports_async_decode"] = True
```

That capability is valid only if the model implements the split decode-readback
contract:

```text
decode_forward(..., read_from_device=False) -> device decode output
read_decode_output(device_output, async_read=True) -> host_output, read_events
process_decode_output_host(host_output, is_tokens=...) -> torch output
```

Before this change, `Llama3Generator` already supported
`decode_forward(..., read_from_device=False)` and `process_decode_output_host()`,
but it did not expose `read_decode_output()`. Because of that, the server
disabled async scheduling at startup even when `--async-scheduling` was passed.

The TTTv2 adapter now exposes async decode support:

- `model_capabilities["supports_async_decode"] = True`
- `read_decode_output(tt_out, async_read=True)` starts nonblocking host copies
  with `ttnn.Tensor.cpu(blocking=False)` and returns a `ttnn.record_event()`
  read event for vLLM to synchronize later.
- `process_decode_output_host()` now accepts either a device tensor or an
  already-host tensor, so it works for both synchronous decode and deferred
  async readback.

The implementation mirrors the mechanism already used in
`models.common.models.executor.run_perf_benchmark(..., pipeline_readback=True)`,
which also uses nonblocking `.cpu(blocking=False)` plus `ttnn.record_event()`.

Verification:

```text
python -m py_compile \
  /localdev/gwang/vllm_duo/tt-metal-too/models/common/models/generator.py \
  plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py
```

```text
PYTHONPATH=/localdev/gwang/vllm_duo/vllm:\
/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src:\
/localdev/gwang/vllm_duo/tt-metal-too \
/localdev/gwang/vllm_duo/tt-metal-too/python_env/bin/pytest \
  plugins/vllm-tt-plugin/tests/test_config.py -q
```

Result:

```text
10 passed, 2 warnings
```

Live async server smoke:

```text
MESH_DEVICE=T3K
TT_LLAMA_TEXT_VER=common_llama3_8b
max_model_len=1024
max_num_seqs=8
--async-scheduling
additional_config={"tt":{"fabric_config":"FABRIC_1D","trace_mode":"decode_only","trace_region_size":85000000}}
```

Server log confirmed that async remained enabled:

```text
Scheduler class: TTScheduler, async_scheduling=True
Asynchronous scheduling is enabled.
```

The previous warning about `Llama3Generator` not declaring
`supports_async_decode` was not present.

Single completion smoke:

- Endpoint: `/v1/completions`
- Prompt: `Explain why rain can make roads slippery in one sentence.`
- `max_tokens=16`, `temperature=0`, `ignore_eos=true`
- Result: HTTP 200, coherent output.
- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_async_smoke_20260709T1431/completion.json`

Concurrent mini benchmark:

```text
random input length: 2
random output length: 32
prompts: 32
request rate: inf
max concurrency: 32
temperature: 0
ignore eos: true
```

Artifact:

```text
/localdev/gwang/vllm_duo/perf_results/tttv2_async_smoke_20260709T1431/mini_bench_result.json
```

Result:

| Shape | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Mean TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv2 async smoke, T3K DP1, 1K context, 32 x 32 tokens | 32 | 0 | 8.66 | 3.69 | 118.22 | 121.91 | 3631.75 | 62.33 | 72.40 |

Interpretation:

- Async decode is now wired through vLLM for TTTv2 Llama 3 8B.
- The implementation is validated by a live server with async scheduling active,
  a single completion smoke, and a concurrent mini benchmark.
- This is a functional validation, not a final performance comparison against
  the 320-request parity workload. A full async parity rerun should use the same
  16K DP1 shape as the prior TTTv2 benchmark before comparing throughput.

### DP4 support investigation

TTTv2 DP4 should not be implemented by simply removing the current
`tt_data_parallel != 1` guard. The vLLM TT plugin's gathered-DP contract requires
the registered model to behave as one logical model that internally owns several
TT execution lanes.

#### vLLM TT DP contract

For `--data_parallel_size 4`, the TT plugin derives:

```text
tt_data_parallel = 4
max_batch_size = max_num_seqs * data_parallel_size
```

Relevant plugin points:

- `get_tt_data_parallel_size()` returns `parallel_config.data_parallel_size`
  when gathered multi-process DP is active.
- `get_tt_max_batch_size()` returns `max_num_seqs * data_parallel_size` for
  gathered DP.
- `TTModelLoader.load_model()` passes both `max_batch_size` and
  `tt_data_parallel` to `model_class.initialize_vllm_model()`.
- `get_num_available_blocks_tt()` passes `tt_data_parallel` and the per-rank
  `max_num_seqs` to `model_class.get_max_tokens_all_users()`.

`TT_LLAMA_TEXT_VER=common_llama3_8b` is not a Galaxy model version, so the
Galaxy gather-DP-to-lanes conversion does not apply. A T3K DP4 run with
`common_llama3_8b` therefore uses gathered multi-process DP:

```text
TTDPEngineCoreProc -> gather per-rank inputs -> concat_and_execute_dp on rank 0
```

Only local DP rank 0 opens the TT mesh and loads the model. Other ranks
participate in host-side scheduling/gather/scatter. This means TTTv2 DP4 must be
one rank-0 model instance that owns four submesh executors, not four independent
processes trying to own the same hardware.

The gathered input shape differs by phase:

- Decode: fixed-rank layout. The plugin stacks per-rank wire tensors and
  reconstructs tensors with `total_B = world * per_rank_max_num_seqs`. Rows are
  rank-strided by fixed capacity: `rank = row // per_rank_max_num_seqs`.
- Prefill: active per-rank `TTModelInput` objects are concatenated. The merged
  `unpadded_batch_size` is a list with one entry per DP rank, and `empty_slots`
  carries the global slot IDs needed to route prefill rows to the right lane.

The model must return output rows in the same merged order so the plugin can
split sampled tokens/logprobs back to individual DP ranks.

#### TTTv1 pattern to reuse

TTTv1 already implements this model-side contract:

- `initialize_vllm_text_transformer()` calls
  `create_submeshes(mesh_device, tt_data_parallel)`.
- For T3K DP4, `create_submeshes()` creates four `1x2` submeshes.
- TTTv1 builds one `ModelArgs` and one `Transformer` per submesh, using
  `max_batch_size // tt_data_parallel` as the per-lane batch.
- The state dict is loaded once from the first `ModelArgs`, then reused when
  constructing the `Transformer` for each submesh.
- The model wrapper returns `cls(tt_model, model_args, mesh_device)`, where
  `tt_model` and `model_args` are lists.

The base `Generator` treats those lists as the DP runtime contract:

- `self.model = model`
- `self.model_args = model_args`
- `self.data_parallel = len(self.model)`

The model-side batch capacity is per lane:

```text
max_batch_size_per_model = self.model_args[0].max_batch_size
```

That value is used to map global vLLM slots to the right TT submesh.

Prefill contract:

- vLLM gather passes a merged `tokens` tensor and `empty_slots` containing
  global slot IDs.
- TTTv1 disables batched prefill when `self.data_parallel > 1`; DP prefill uses
  the sequential per-user path.
- For each global slot, TTTv1 routes:

```text
model_id = user_id // max_batch_size_per_model
local_user_id = user_id % max_batch_size_per_model
```

- With paged attention, `group_user_id` is pinned to `0` and the per-user page
  table carries the actual KV mapping.
- It slices a one-row `page_table_for_user`, builds the lane-local TT page table
  with `kv_cache[model_id]`, and calls the selected submesh model with that
  lane's KV cache.
- Prefill outputs are written back into the merged output tensor at the
  original global slot index, preserving vLLM's expected row order.

Decode contract:

- vLLM gather reconstructs a fixed-size merged decode batch with rows grouped by
  rank.
- TTTv1 chunks `tokens`, `start_pos`, and `page_table` by
  `self.data_parallel`.
- Each chunk is sent to `self.model[i]` with `kv_cache[i]`.
- Trace state is also per lane: decode trace capture/replay loops over
  `self.data_parallel` and stores one trace id per submesh.
- Host readback/postprocess loops over all lanes and concatenates lane outputs
  in lane order.

KV allocation contract:

```text
kv_cache[dp_rank][layer_idx][k_or_v]
```

TTTv1 allocates one K/V tensor pair per layer per submesh with
`ttnn.ReplicateTensorToMesh(submesh)`. The returned nested list is passed back
to the same model-side DP routing above.

Async decode contract:

- `decode_forward(..., read_from_device=False)` returns a list of per-lane TT
  outputs.
- `read_decode_output(..., async_read=True)` submits a nonblocking host copy per
  lane and records one event per lane mesh.
- `process_decode_output_host()` converts each lane's host output and
  concatenates the result across DP lanes.

This is the shape the vLLM TT runner already expects for gathered DP.

#### Current TTTv2 blockers

The current TTTv2 adapter is single-lane:

- `get_max_tokens_all_users()` rejects `tt_data_parallel != 1`.
- `initialize_vllm_model()` rejects `tt_data_parallel != 1`.
- `Llama3Generator` owns one executor, one model, one `model_args`, and one
  mesh.
- `EagerLLMExecutor` and `TracedLLMExecutor` own one model, one mesh, one KV
  cache identity, and one trace state.
- `process_decode_output_host()` and `read_decode_output()` currently process
  one output tensor and one read event domain.

The lower-level TTTv2 Llama model appears more flexible than the adapter:

- `build_llama3_transformer_1d_config()` derives `num_devices` and
  `cluster_shape` from the provided mesh.
- Llama3-8B requires `n_heads` and `n_kv_heads` to divide `cluster_shape[1]`;
  a `1x2` T3K submesh should satisfy that for 32 attention heads and 8 KV heads.
- On submeshes with fewer than 8 devices, the model uses linear CCL topology and
  disables the 8-device fused all-gather matmul path. That makes DP4 plausible,
  but it may not match the performance of the current 8-device DP1 path per
  lane.

#### Recommended implementation path

Use a generator-level DP wrapper rather than teaching the shared executors about
multiple meshes.

Lowest-risk plan:

1. Keep the existing single-lane `Llama3Generator` behavior for
   `tt_data_parallel == 1`.
2. For `tt_data_parallel > 1`, call the existing TTTv1
   `create_submeshes(mesh_device, tt_data_parallel)`.
3. Build one TTTv2 model/executor per submesh, with
   `per_lane_max_batch_size = max_batch_size // tt_data_parallel`.
4. Store lane objects as lists:

```text
self.executors[lane]
self.models[lane]
self.model_args[lane]
self.mesh_devices[lane]
```

5. Allocate KV per lane and return `kv_cache[lane][layer][k_or_v]`.
6. Decode:
   - split merged `tokens`, `start_pos`, and `page_table` into
     `tt_data_parallel` fixed-size chunks,
   - call each lane executor with `kv_cache[lane]`,
   - concatenate host logits/tokens in lane order.
7. Prefill:
   - use `empty_slots` to map each merged row to `lane = slot //
     per_lane_max_batch_size`,
   - remap the slot to local lane coordinates,
   - call the lane executor with lane-local rows and `kv_cache[lane]`,
   - place outputs back into the merged row order.
8. Async decode:
   - call each lane's `read_decode_output(..., async_read=True)`,
   - return a flattened event list,
   - merge lane host outputs only after all events are synchronized.
9. Start validation with a small DP4 smoke using `max_model_len=1024` and
   `max_num_seqs=1` or `2`, then scale to the TTTv1 parity shape.

Avoid the higher-risk path of modifying `EagerLLMExecutor` /
`TracedLLMExecutor` to own multiple meshes. That would touch shared trace,
page-table, KV identity, and output-processing code.

#### Open questions before implementation

- Whether `from_pretrained()` can cheaply reuse the converted HF state dict
  across four TTTv2 submesh model builds, or whether the first implementation
  should accept repeated load/build cost for lower code risk.
- Whether all TTTv2 Llama3-8B kernels needed by vLLM are compiled and stable on
  `1x2` T3K submeshes.
- Whether on-device sampling should be disabled for the first DP4 bringup, then
  re-enabled after host-sampling DP4 works.
- Whether the first DP4 run should use gathered DP exactly like TTTv1, or add a
  future lane-mode conversion only after gathered DP is functional.

#### TTTv2 implementation entry points

After checking the current TTTv2 adapter and executor code, the generator-level
wrapper remains the right implementation boundary.

Current single-lane assumptions:

- `models/common/models/generator.py::Llama3Generator` stores exactly one
  executor, model, `model_args`, and mesh.
- `Llama3Generator.get_max_tokens_all_users()` and
  `initialize_vllm_model()` explicitly reject `tt_data_parallel != 1`.
- `EagerLLMExecutor.allocate_kv_cache()` allocates one cache on
  `self.mesh_device`, stores it in `self._kv_cache`, and calls
  `self.model.set_kv_cache(kv_cache)`.
- `EagerLLMExecutor._assert_kv_cache_identity()` requires the forward-path
  `kv_cache` object to be the exact object allocated by that executor.
- `prepare_decode_inputs_host()` asserts the decode batch size equals that
  executor's `max_batch_size`; this matches per-lane fixed decode chunks, not a
  merged DP4 batch.
- `TracedLLMExecutor` owns one decode trace state and one previous decode page
  table, so trace capture/replay must stay per lane.
- The output helpers `_process_output_decode()` and
  `_process_output_decode_tokens()` are already reusable per lane once each lane
  produces host output.

The first DP4 patch should therefore keep `EagerLLMExecutor` and
`TracedLLMExecutor` single-mesh. `Llama3Generator.initialize_vllm_model()` can
branch:

```text
tt_data_parallel == 1 -> existing single-lane path
tt_data_parallel > 1  -> build/return a DP wrapper containing N single-lane generators
```

The DP wrapper can live in `models/common/models/generator.py` beside
`Llama3Generator` and expose the same vLLM-facing methods:

```text
get_max_tokens_all_users()
initialize_vllm_model()
allocate_kv_cache()
prefill_forward()
decode_forward()
read_decode_output()
process_decode_output_host()
warmup_model_prefill()
warmup_model_decode()
cache_path
```

Implementation details to preserve the gathered-DP contract:

- Build submeshes with TTTv1's `create_submeshes(mesh_device,
  tt_data_parallel)`.
- Build one existing single-lane TTTv2 generator per submesh with
  `per_lane_max_batch_size = max_batch_size // tt_data_parallel`.
- Return KV as a nested per-lane structure:

```text
kv_cache[lane][layer][k_or_v]
```

- `decode_forward()` should split merged decode tensors by fixed per-lane
  capacity. For DP4 with `max_num_seqs=8`, the merged decode batch has 32 rows,
  and lane `i` receives rows `[i * 8 : (i + 1) * 8)`.
- `prefill_forward()` should route by global `empty_slots`, not by row index:

```text
lane = empty_slot // per_lane_max_batch_size
local_slot = empty_slot % per_lane_max_batch_size
```

- Prefill row order must be restored before returning to vLLM. The model runner
  samples from the returned merged tensor with rank-local contiguous row ranges,
  so the wrapper must scatter lane results back into the original merged row
  order.
- Async decode should return raw per-lane outputs from `decode_forward(...,
  read_from_device=False)`, submit one readback per lane in
  `read_decode_output(..., async_read=True)`, flatten all lane events, and only
  concatenate tensors in `process_decode_output_host()`.
- Warmup should call each lane with that lane's KV cache and
  `max_batch_size=per_lane_max_batch_size`. Passing the merged batch size into
  one executor would trip the per-lane decode batch assertion.
- The first bringup should run without on-device sampling. The vLLM plugin
  already has `slice_tt_sampling_params()`, but the TTTv2 DP wrapper would need
  explicit sampling-param slicing by lane before on-device sampling can be
  considered complete.

Initial implementation scope:

1. Add the DP wrapper and leave the single-lane code path unchanged.
2. Enable `get_max_tokens_all_users(... tt_data_parallel > 1)` to return the
   per-lane `max_model_len`, matching TTTv1's per-submesh KV sizing.
3. Implement host-sampling DP4 decode/prefill routing first.
4. Add defensive errors for unsupported TTTv2 DP4 on-device sampling rather than
   silently using merged sampling parameters on a single lane.
5. Validate with a small T3K DP4 smoke before attempting the TTTv1 parity shape.

Implementation status as of 2026-07-09:

- `models/common/models/generator.py` now has a generator-level
  `_DPLlama3Generator` that is selected when
  `Llama3Generator.initialize_vllm_model(... tt_data_parallel > 1)` is called.
  The single-lane path was factored into `_initialize_single_lane()` and remains
  the path for `tt_data_parallel == 1`.
- The wrapper uses TTTv1 `create_submeshes()` to split the T3K mesh into one
  submesh per DP lane, builds one TTTv2 single-lane generator per submesh, and
  allocates lane-local KV caches as `kv_cache[lane][layer][k_or_v]`.
- Prefill routing groups scheduled rows by global KV slot:
  `lane_idx = empty_slot // per_lane_max_batch_size` and
  `local_slot = empty_slot % per_lane_max_batch_size`. Lane outputs are scattered
  back by original scheduled row index, not by KV slot id.
- Decode routing expects the merged vLLM batch to be exactly
  `per_lane_max_batch_size * tt_data_parallel`, splits contiguous row chunks per
  lane, and concatenates host outputs after lane-local readback.
- Async decode readback is lane-aware: `decode_forward(... read_from_device=False)`
  returns raw per-lane outputs, `read_decode_output(... async_read=True)` submits
  lane-local reads and flattens the event list, and
  `process_decode_output_host()` concatenates the per-lane host tensors.
- Warmup fans out to each lane with lane-local KV cache and
  `max_batch_size=per_lane_max_batch_size`.
- On-device sampling is intentionally rejected for TTTv2 DP for now. Supporting
  it requires slicing device sampling parameters per lane instead of passing
  merged sampling state into each submesh.

Verification:

- Import/compile checks passed with the TT Python environment:
  `python -m py_compile models/common/models/generator.py`.
- A fake-lane routing test passed for non-contiguous prefill slots
  (`empty_slots=[2, 0, 3]`) and decode chunking, verifying row scatter and lane
  split behavior without using TT hardware.
- vLLM TT plugin config tests passed:
  `pytest plugins/vllm-tt-plugin/tests/test_config.py -q` reported
  `10 passed, 2 warnings`.
- T3K DP4 server smoke after `tt-smi -r` reached API startup with
  `--data-parallel-size 4`, `MESH_DEVICE=T3K`, fabric `FABRIC_1D`, trace mode
  `decode_only`, and model warmup disabled. Logs showed DP rank startup, DP0
  opening the 8-device mesh, four submesh lane models being built, lane-local KV
  cache allocation/loading, coordinator subscription completion, and
  `Application startup complete`.
- A longer T3K DP4 smoke after another `tt-smi -r` completed successfully. The
  first `/v1/completions` request with `max_tokens=4` returned HTTP 200 in
  2m24.428s after finishing the remaining lane-specific cache generation and
  decode trace capture. Response text was `" Rain can make roads"`, with
  `prompt_tokens=13`, `completion_tokens=4`, and `total_tokens=17`.
- A second request on the same live server returned HTTP 200 in 0.297s, showing
  that the wrapper path responds quickly once lane caches/traces are populated.
  Response text was `" \n-----------------------------------------------\n\n"`, with
  `prompt_tokens=12`, `completion_tokens=4`, and `total_tokens=16`.
- Server logs for the successful longer smoke showed both requests returning
  HTTP 200 and the final lane reporting `Compiled decode` and
  `Captured decode trace`. No wrapper exception was observed. The server was
  shut down cleanly afterward; only the usual TT nanobind shutdown leak warnings
  were printed.

DP4 parity benchmark attempt:

- The exact TTTv1 parity server shape still cannot be constructed with TTTv2:
  `--data-parallel-size 4`, `--max-model-len 131072`, and `--max_num_seqs 8`
  failed during TTTv2 submesh model construction. The model-side guard rejected
  the per-lane budget:
  `max_batch_size (8) x max_seq_len (131072) = 1,048,576 tokens`, above the
  TTTv2 128K limit. Artifact:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_20260709T152656Z/server.log`.
- The closest valid DP4 benchmark used the same client workload and DP4
  scheduling, but with `--max-model-len 16384` so the per-lane TTTv2 budget is
  `8 x 16384 = 131072` tokens. Server config:
  `MESH_DEVICE=T3K`, `TT_LLAMA_TEXT_VER=common_llama3_8b`,
  `--data-parallel-size 4`, `--max_num_seqs 8`, fabric `FABRIC_1D`, trace mode
  `decode_only`, model warmup disabled, host sampling.
- Client workload matched the TTTv1 no-sample target: `vllm bench serve`,
  OpenAI completions endpoint, random input length `2`, random output length
  `256`, `320` prompts, request rate `inf`, max concurrency `32`,
  `--ignore-eos`, and `temperature=0`.
- Result artifacts:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_20260709T152656Z/result_dp4_16k.json`,
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_20260709T152656Z/client_dp4_16k.log`,
  and
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_20260709T152656Z/server_dp4_16k.log`.

| Implementation | Server shape | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Peak output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv1 baseline | T3K DP4, `max_model_len=131072`, `max_num_seqs=8` | 320 | 0 | 242.03 | 1.32 | 338.47 | 339.79 | 408.00 | 2364.95 | 2312.97 | 3792.21 | 85.63 | 84.91 | 97.10 |
| TTTv2 DP4 valid shape | T3K DP4, `max_model_len=16384`, `max_num_seqs=8` | 320 | 0 | 292.79 | 1.09 | 279.79 | 280.88 | 376.00 | 3472.48 | 1199.00 | 22789.04 | 94.18 | 92.67 | 110.40 |

The valid TTTv2 DP4 run reaches about `82.7%` of the TTTv1 baseline output
throughput (`279.79 / 338.47`). This is a functional DP4 benchmark result, but
not a strict apples-to-apples parity shape because TTTv2 still cannot construct
the TTTv1 `max_model_len=131072` plus `max_num_seqs=8` model-side budget.

131K DP4 token-budget investigation:

- The TTTv2 failure is not caused by DP lane slicing. The DP wrapper receives
  the global gathered batch from the vLLM TT plugin (`max_num_seqs * DP = 32`)
  and correctly builds each lane with `per_lane_max_batch_size=8`.
- The blocking guard is in
  `models/common/modules/attention/attention_1d.py:_resolve_attention1d_config`.
  It unconditionally computes
  `config.max_batch_size * config.max_seq_len` and rejects values above
  `MAX_TOTAL_TOKENS = 128 * 1024`.
- In the vLLM path, `max_seq_len` is the per-request context length
  (`model_config.max_model_len`). It is also used to build RoPE tables for the
  requested context. It is not the physical KV allocation size when
  `use_vllm_paged_kv_cache=True`.
- TTTv2 vLLM paged KV currently builds each lane's paged attention config as
  `ceil(max_seq_len / 32) + max_batch_size` blocks. For the parity shape, that
  is `4096 + 8 = 4104` blocks per lane, matching the vLLM TT worker's
  `max_tokens_all_users=max_model_len` plus one-block-per-lane-slot padding.
- TTTv1 does not reject `max_batch_size * max_seq_len` during model
  construction. Its vLLM DP contract builds four T3K `1x2` submeshes, each with
  `max_batch_size=8`, and sizes the external paged KV cache from the total
  per-submesh token budget, not from eight simultaneous full-length contexts.
- The TTTv2 attention guard is valid for static/non-vLLM KV allocation, where
  the model would allocate `[max_batch_size, max_seq_len]` KV storage. It is too
  strict for externally managed vLLM paged KV, where vLLM owns the block budget
  and scheduling rejects workloads that cannot fit in the available blocks.
- Implemented fix: `attention_1d.py:_resolve_attention1d_config` now delegates
  token-budget validation to a paged-KV-aware helper. Static/non-vLLM KV keeps
  the existing `max_batch_size * max_seq_len <= 128K` guard. For
  `use_vllm_paged_kv_cache=True` with a `paged_attention_config`, TTTv2 now
  validates the physical paged budget (`max_num_blocks * block_size`) against
  the same 128K-family cap plus one block per batch slot, and allows
  `max_seq_len` to remain the per-request context length.
- Main risk to audit before running the full parity benchmark again: any TTTv2
  code that derives a per-user page-table width from
  `max_num_blocks // max_batch_size` would be incompatible with shared vLLM
  paged KV. The vLLM plugin's persistent block tables still allocate
  `ceil(max_model_len / block_size)` entries per request and slice them by the
  actual KV block count at runtime.
- Verification after the fix:
  `python -m py_compile models/common/modules/attention/attention_1d.py models/common/tests/modules/attention/test_attention_1d.py`
  passed, and
  `pytest models/common/tests/modules/attention/test_attention_1d.py -q -k 'token_budget or vllm_paged_kv'`
  reported `4 passed, 1465 deselected`.
- T3K DP4 parity smoke after the fix passed at the exact previously-blocked
  shape: `--data-parallel-size 4`, `--max-model-len 131072`,
  `--max_num_seqs 8`, fabric `FABRIC_1D`, trace mode `decode_only`, and model
  warmup disabled. The server reached `Application startup complete`; all DP
  ranks reported `GPU KV cache size: 131,328 tokens` and
  `Maximum concurrency for 131,072 tokens per request: 1.00x`.
- Smoke request results:
  first `/v1/completions` request returned HTTP 200 in `9.487s`, text
  `" It can cause flooding"`, usage `9` prompt tokens, `4` completion tokens,
  `13` total tokens. A second request on the same server returned HTTP 200 in
  `0.330s`, text `" A compass is used"`, usage `8` prompt tokens, `4`
  completion tokens, `12` total tokens. Server logs showed each completion
  returning HTTP 200 and decode trace compilation/capture before the first
  response. Artifacts:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_smoke_20260709T155435Z/server.log`,
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_smoke_20260709T155435Z/first_request.json`,
  and
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_smoke_20260709T155435Z/second_request.json`.
- Full DP4 parity benchmark after the paged-KV-aware guard passed at the exact
  TTTv1 target shape: `--data-parallel-size 4`, `--max-model-len 131072`,
  `--max_num_seqs 8`, fabric `FABRIC_1D`, trace mode `decode_only`, model
  warmup disabled, host sampling, OpenAI completions backend, random input
  length `2`, random output length `256`, `320` prompts, request rate `inf`,
  max client concurrency `32`, `--ignore-eos`, and `temperature=0`.
- The first full-benchmark server attempt failed before API readiness with a TT
  infrastructure heartbeat timeout on one ETH core:
  `Timed out waiting for ETH heartbeat ... to advance`. This was not a
  model/token-budget failure; other ranks had already reported the expected
  `GPU KV cache size: 131,328 tokens`. After `tt-smi -r`, the retry reached
  `Application startup complete` and completed the benchmark with zero request
  failures.
- Full benchmark artifacts:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_full_20260709T160255Z/server.log`,
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_full_20260709T160255Z/server_retry2.log`,
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_full_20260709T160255Z/client.log`,
  and
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_full_20260709T160255Z/result.json`.

| Implementation | Server shape | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Peak output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv1 baseline | T3K DP4, `max_model_len=131072`, `max_num_seqs=8` | 320 | 0 | 242.03 | 1.32 | 338.47 | 339.79 | 408.00 | 2364.95 | 2312.97 | 3792.21 | 85.63 | 84.91 | 97.10 |
| TTTv2 DP4 parity shape | T3K DP4, `max_model_len=131072`, `max_num_seqs=8` | 320 | 0 | 273.75 | 1.17 | 299.25 | 300.42 | 352.00 | 2466.28 | 1287.15 | 13407.84 | 97.67 | 94.09 | 144.18 |

The exact-shape TTTv2 DP4 run reaches about `88.4%` of the TTTv1 baseline
output throughput (`299.25 / 338.47`) and `88.4%` of total token throughput
(`300.42 / 339.79`). Duration is `13.1%` longer, mean TTFT is `4.3%` slower,
and mean TPOT is `14.1%` slower. Functionally, the DP4 parity shape now works;
performance is still not on par with TTTv1.

Throughput-gap investigation:

- The throughput gap tracks decode TPOT more closely than TTFT. TTTv2 exact
  DP4 mean TPOT is `97.67 ms` versus TTTv1 `85.63 ms`; `85.63 / 97.67 =
  87.7%`, which nearly matches the output-throughput ratio `299.25 / 338.47 =
  88.4%`. Mean TTFT is only `4.3%` slower, though TTTv2 has much worse tail
  TTFT (`13407.84 ms` p99 versus `3792.21 ms`).
- The completed TTTv2 run was not persistently queue-starved. Server logger
  samples mostly show all four engines/lane groups running `8` requests each
  with `Waiting: 0`. The observed per-engine generation samples are commonly
  `75-87 tok/s`, while TTTv1 needs about `84.6 tok/s` per DP rank to reach its
  documented `338.47 tok/s` total.
- The full TTTv2 benchmark was run with `enable_model_warmup=false`; the TTTv1
  baseline used `enable_model_warmup=true`. TTTv2 logs show `Skipping model
  warmup`, then first-benchmark decode trace compile/capture around the early
  measured window. This likely explains part of the TTFT tail and some low
  throughput boundary windows, but not the whole TPOT gap.
- TTTv2 currently preserves the user-facing DP4 shape by converting
  `--data_parallel_size 4 --max_num_seqs 8` into single-process TT lane-DP:
  global `max_num_seqs=32`, four internal lanes, and per-lane capacity `8`.
  TTTv1 baseline used gathered multi-process vLLM DP4. Functional capacity is
  equivalent, but host execution topology is different.
- The current TTTv2 DP wrapper launches lane decode in a Python loop:
  `_DPLlama3Generator.decode_forward()` slices the merged batch and calls each
  lane's `decode_forward` sequentially, then loops again for async readback and
  host output processing. If lane trace submissions or read setup are not fully
  nonblocking, this is a plausible steady TPOT contributor.
- vLLM's gathered-DP async path also finalizes the previous DP step before
  submitting the next one to avoid stale token state. That means
  `async_scheduling=True` does not imply full submit-next/finalize-previous
  overlap for DP decode.
- TTTv2 DP rejects on-device sampling today, and the parity run uses
  `sample_on_device_mode=None`. This matches the no-sample TTTv1 baseline and
  is not the main parity gap, but implementing DP on-device sampling remains a
  separate performance lever.
- Paged-KV page-table refresh is another targeted suspect: the traced executor
  compares and copies the page-table trace input when it changes. If vLLM block
  allocation changes page tables frequently during the benchmark, this could
  contribute to TPOT variance and tail latency.

Next validation target: rerun the exact TTTv2 DP4 parity benchmark with model
warmup enabled, matching the TTTv1 baseline setting, to separate cold
trace/warmup pollution from true steady-state decode cost.

Warmup-enabled validation result:

- Ran the exact DP4 parity workload again with `enable_model_warmup=true`
  (default; the `enable_model_warmup=false` override was removed). Server shape:
  `--data-parallel-size 4`, `--max-model-len 131072`, `--max_num_seqs 8`,
  fabric `FABRIC_1D`, trace mode `decode_only`, host sampling. Client workload:
  `320` random prompts, input length `2`, output length `256`, request rate
  `inf`, max concurrency `32`, `--ignore-eos`, `temperature=0`.
- Artifacts:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_warmup_20260709T162530Z/tt_smi_reset.log`,
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_warmup_20260709T162530Z/server.log`,
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_warmup_20260709T162530Z/client.log`,
  and
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_warmup_20260709T162530Z/result.json`.
- Server warmup reached `Application startup complete`. DP0 reported
  `init engine (profile, create kv cache, warmup model) took 51.02 seconds`
  and captured four decode traces before serving. The benchmark still triggered
  some prefill trace compile/capture for `seq_len=128` batch sizes during the
  measured run, so warmup did not eliminate all cold prefill work.
- Client completed `320` successful requests and `0` failed requests. No
  server/client processes remained after stopping the server.

| Implementation | Server shape | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Peak output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv1 baseline | T3K DP4, `max_model_len=131072`, `max_num_seqs=8`, warmup enabled | 320 | 0 | 242.03 | 1.32 | 338.47 | 339.79 | 408.00 | 2364.95 | 2312.97 | 3792.21 | 85.63 | 84.91 | 97.10 |
| TTTv2 DP4 parity, no warmup | T3K DP4 lane mode, `max_model_len=131072`, `max_num_seqs=8`, warmup disabled | 320 | 0 | 273.75 | 1.17 | 299.25 | 300.42 | 352.00 | 2466.28 | 1287.15 | 13407.84 | 97.67 | 94.09 | 144.18 |
| TTTv2 DP4 parity, warmup enabled | T3K DP4 lane mode, `max_model_len=131072`, `max_num_seqs=8`, warmup enabled | 320 | 0 | 257.04 | 1.24 | 318.71 | 319.95 | 376.00 | 1385.61 | 1304.81 | 2583.76 | 95.36 | 94.46 | 100.15 |

Warmup recovers a material part of the gap: output throughput improves
`6.5%` over the no-warmup TTTv2 run (`318.71 / 299.25`) and reaches `94.2%`
of the TTTv1 baseline (`318.71 / 338.47`). Tail latency also normalizes:
TTTv2 p99 TTFT improves from `13407.84 ms` to `2583.76 ms`, now better than
the TTTv1 baseline's `3792.21 ms`.

The remaining performance gap is still decode cadence. Warmup-enabled TTTv2
mean TPOT is `95.36 ms`, which is `11.4%` slower than TTTv1's `85.63 ms`, and
closely explains the remaining throughput ratio. This keeps the next likely
owners as lane-DP decode hot-path overhead, per-step DP gather/scatter and
finalize-before-submit behavior, or page-table/prefill trace churn rather than
cold decode trace capture alone.

### DP wrapper overhead profile

To start the throughput-gap investigation, I added env-gated host timing probes:

- `TT_DP_PROFILE=1` in
  `plugins/vllm-tt-plugin/src/vllm_tt_plugin/engine.py` emits
  `TT_DP_PROFILE_JSON` for decode `dp_gather_submit` and
  `dp_gather_finalize`.
- The same env var in
  `/localdev/gwang/vllm_duo/tt-metal-too/models/common/models/generator.py`
  emits TTTv2-only wrapper events for `decode_forward`,
  `read_decode_output`, and `process_decode_output_host`.
- Normal runs are unchanged when `TT_DP_PROFILE` is unset.

Short comparison workload:

- Server shape: T3K DP4, `--max-model-len 131072`, `--max_num_seqs 8`,
  fabric `FABRIC_1D`, trace mode `decode_only`, host sampling.
- Client workload: `64` random prompts, input length `2`, output length `128`,
  request rate `inf`, max concurrency `32`, `--ignore-eos`, `temperature=0`.
- TTTv1 env: `TT_LLAMA_TEXT_VER=tt_transformers`,
  `HF_MODEL=meta-llama/Llama-3.1-8B-Instruct`.
- TTTv2 env: `TT_LLAMA_TEXT_VER=common_llama3_8b`.

Artifacts:

- TTTv1:
  `/localdev/gwang/vllm_duo/perf_results/dp_profile_tttv1_20260709T170438Z/`
  (`server_retry.log` is the valid run; the first `server.log` failed before
  load because `HF_MODEL` was missing).
- TTTv2:
  `/localdev/gwang/vllm_duo/perf_results/dp_profile_tttv2_20260709T170149Z/`.

Request-level result:

| Implementation | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv1 short profile | 64 | 0 | 72.68 | 112.71 | 25718.78 | 49255.12 | 83.61 | 89.66 |
| TTTv2 short profile | 64 | 0 | 28.20 | 290.54 | 1744.11 | 2375.82 | 97.24 | 106.62 |

The short TTTv1 run had badly polluted TTFT/output throughput from prefill or
trace work, so the request-level throughput is not useful for parity. The
steady decode metric is useful: TTTv1 mean TPOT was `83.61 ms`, matching the
full-run baseline (`85.63 ms`), while TTTv2 mean TPOT was `97.24 ms`, matching
the warmup-enabled full-run gap (`95.36 ms`).

Decode DP timing summary, all ranks aggregated:

| Event | Impl | Mean total ms | P50 total ms | P95 total ms | Main subfields |
|---|---|---:|---:|---:|---|
| `dp_gather_submit` | TTTv1 | 25.48 | 20.70 | 45.53 | `build_model_input=11.87`, `all_reduce=4.98`, `execute_submit=5.19` |
| `dp_gather_submit` | TTTv2 | 27.57 | 22.79 | 48.03 | `build_model_input=13.52`, `all_reduce=5.56`, `execute_submit=5.19` |
| `dp_gather_finalize` | TTTv1 | 36.87 | 36.64 | 44.52 | rank 0 waits on future; ranks 1-3 wait in scatter |
| `dp_gather_finalize` | TTTv2 | 48.33 | 48.19 | 57.80 | same pattern, about `+11.46 ms` versus TTTv1 |

Rank-level finalize view:

| Impl | Rank 0 finalize mean | Rank 1-3 finalize mean | Interpretation |
|---|---:|---:|---|
| TTTv1 | `36.83 ms` (`future_wait=36.33`) | `36.87-36.90 ms` (`scatter=36.65-36.66`) | rank 0 waits for model result; peers block until scatter |
| TTTv2 | `48.27 ms` (`future_wait=47.79`) | `48.30-48.39 ms` (`scatter=48.07-48.15`) | TTTv2 rank 0 model-side decode/read/process is later by about `11-12 ms` |

TTTv2-only wrapper timing on rank 0:

| Event | Count | Mean total ms | P50 total ms | P95 total ms | Notes |
|---|---:|---:|---:|---:|---|
| `tttv2_dp_wrapper_decode_forward` | 254 | 5.53 | 5.35 | 7.26 | four lane submissions in a Python loop; mean per-lane call `1.37 ms` |
| `tttv2_dp_wrapper_read_decode_output` | 254 | 14.17 | 13.95 | 15.86 | four lane async host reads in a Python loop; mean per-lane read `3.54 ms` |
| `tttv2_dp_wrapper_process_decode_output_host` | 254 | 19.53 | 18.62 | 26.30 | four lane host postprocess calls plus cat; mean per-lane process `4.73 ms`, cat `0.59 ms` |

First bottleneck callout:

- The shared vLLM DP gather path is only modestly slower for TTTv2 submit
  (`+2.09 ms` mean), but finalize is about `+11.46 ms`, which matches the
  request-level TPOT gap (`97.24 - 83.61 = 13.63 ms`).
- The extra finalize time is synchronization delay, not expensive scatter
  payload movement by itself: rank 0 spends it in `future.result()`, and the
  other ranks spend the same time waiting in `dist.scatter`.
- The TTTv2-only wrapper has enough serialized rank-0 work to explain this:
  readback plus host postprocess alone average about `33.70 ms` per decode
  event, and they run after the four lane decode submissions. TTTv1 does not
  use this Python lane wrapper in the same way.

Next optimization target: reduce or overlap the TTTv2 rank-0 wrapper
read/process path before changing the common DP collectives. Concrete ideas to
evaluate next are: return a merged async read result that avoids four
sequential `read_decode_output()` calls, parallelize or fuse per-lane host
postprocess/cat, and compare against the TTTv1 model-side decode output path to
see whether TTTv1 already reads/reshapes a full DP tensor instead of four lane
tensors.

Follow-up implementation:

- First tried the direct "steal TTTv1 model-side shape" fast path in the TTTv2
  DP wrapper: read each lane output directly and call the common
  `_process_output_decode()` helper directly instead of bouncing through four
  `Llama3Generator.read_decode_output()` /
  `Llama3Generator.process_decode_output_host()` adapter calls.
- That direct bypass did not materially help. Artifact:
  `/localdev/gwang/vllm_duo/perf_results/dp_profile_tttv2_fastpath_20260709T173111Z/`.
  Mean TPOT was `97.63 ms` versus the previous TTTv2 profile's `97.24 ms`.
- The useful part of the TTTv1 idea is not avoiding a Python method call; it is
  treating the DP lanes as independent model-side work. The next patch kept the
  direct per-lane read/process helpers and used a persistent
  `ThreadPoolExecutor(max_workers=tt_data_parallel)` in `_DPLlama3Generator` to
  submit/read/process four independent lane outputs concurrently.

Threaded-output profile result:

- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/dp_profile_tttv2_parallel_output_20260709T173452Z/`.
- Server shape and client workload matched the short profile above:
  T3K DP4, `--max-model-len 131072`, `--max_num_seqs 8`, `64` random prompts,
  input length `2`, output length `128`, max concurrency `32`.
- Completed `64`, failed `0`; no server processes remained after shutdown.

| TTTv2 variant | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline wrapper | 64 | 0 | 28.20 | 290.54 | 1744.11 | 2375.82 | 97.24 | 106.62 |
| Direct helper bypass | 64 | 0 | 28.67 | 285.71 | 1932.56 | 2333.31 | 97.63 | 107.27 |
| Parallel lane output | 64 | 0 | 23.46 | 349.16 | 1842.31 | 2329.29 | 77.83 | 87.75 |

Profile deltas:

| Event | Baseline mean ms | Direct bypass mean ms | Parallel lane output mean ms |
|---|---:|---:|---:|
| `dp_gather_submit.total_ms` | 27.57 | 27.75 | 15.66 |
| `dp_gather_finalize.total_ms` | 48.33 | 47.62 | 48.08 |
| `tttv2_dp_wrapper_decode_forward.total_ms` | 5.53 | 5.64 | 5.02 |
| `tttv2_dp_wrapper_read_decode_output.total_ms` | 14.17 | 14.82 | 5.49 |
| `tttv2_dp_wrapper_process_decode_output_host.total_ms` | 19.53 | 19.33 | 13.55 |

Current interpretation:

- The first direct bypass disproved "adapter method-call overhead" as the
  bottleneck.
- Parallel lane output confirmed that serialized rank-0 lane handling was a real
  bottleneck: mean TPOT improved from `97.24 ms` to `77.83 ms`, better than the
  TTTv1 short-profile TPOT of `83.61 ms`.
- `dp_gather_finalize.total_ms` did not drop because the shorter submit/output
  path shifts where the pipeline waits; request-level TPOT and wrapper event
  timing are the better signal for this patch.

Next validation target: run a longer DP4 parity benchmark with the parallel lane
output patch to check whether the short-run gain holds at the full `320 x 256`
workload.

Full DP4 parity validation after parallel lane output:

- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_parallel_output_20260709T174453Z/`.
- Server shape: T3K DP4, `--max-model-len 131072`, `--max_num_seqs 8`,
  fabric `FABRIC_1D`, trace mode `decode_only`, model warmup enabled, host
  sampling.
- Client workload: `320` random prompts, input length `2`, output length `256`,
  request rate `inf`, max concurrency `32`, `--ignore-eos`, `temperature=0`.
- Completed `320`, failed `0`. No server or `EngineCore_DP` processes remained
  after shutdown.

| Implementation | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Peak output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv1 baseline | 320 | 0 | 242.03 | 1.32 | 338.47 | 339.79 | 408.00 | 2364.95 | 2312.97 | 3792.21 | 85.63 | 84.91 | 97.10 |
| TTTv2 before parallel output | 320 | 0 | 257.04 | 1.24 | 318.71 | 319.95 | 376.00 | 1385.61 | 1304.81 | 2583.76 | 95.36 | 94.46 | 100.15 |
| TTTv2 parallel lane output | 320 | 0 | 241.54 | 1.32 | 339.15 | 340.48 | 440.00 | 1689.16 | 1200.45 | 22166.26 | 81.25 | 79.68 | 95.46 |

The full parity run confirms the short-profile improvement:

- Output throughput is now slightly above the TTTv1 baseline:
  `339.15 / 338.47 = 100.2%`.
- Mean TPOT is better than TTTv1: `81.25 ms` versus `85.63 ms`.
- The parallel-output patch improves TTTv2 full-run output throughput by
  `6.4%` over the warmup-enabled pre-patch run (`339.15 / 318.71`) and mean
  TPOT by `14.8%` (`95.36 -> 81.25 ms`).
- P99 TTFT regressed to `22166.26 ms`. The client progress and server log show
  a small set of first-token outliers around `21.7-24.2s`, and server logs show
  measured-run `seq_len=128` prefill trace compile/capture after startup. This
  is a remaining TTFT warmup/trace coverage issue, not a decode throughput
  regression or request failure.

Current status: TTTv2 DP4 decode throughput is on par with TTTv1 for the parity
workload. Remaining work should focus on eliminating measured-run prefill trace
capture/TTFT tail and auditing the threaded lane-output path for TTNN
thread-safety assumptions before treating the optimization as production-ready.

### TTTv2 DP4 TTFT prefill trace tail investigation

Root cause found:

- TTTv2 uses `TracedLlamaExecutor` even when vLLM is configured with
  `trace_mode=decode_only`.
- Before this investigation, vLLM only ran Phase 2 prefill trace warmup when
  `trace_mode=all`, so TTTv2 decode-only mode could still capture prefill
  traces on demand during measured traffic.
- `Llama3Generator.warmup_model_prefill()` also only covered batch sizes
  `[1, max_batch_size]` for `seq_len=128`; the DP4 workload can admit prefill
  batches of `1`, `2`, `4`, and `8`.
- In the prior full DP4 run
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_parallel_output_20260709T174453Z/`,
  the server log showed measured-window prefill trace capture after API
  startup, matching the handful of `~21.7-24.2s` TTFT outliers and the
  `22166.26 ms` p99 TTFT.

Patch summary:

- Added `requires_prefill_trace_warmup = True` to the TTTv2 single-lane and DP
  generator adapters.
- Taught `TTModelRunner` to run Phase 2 prefill trace warmup for
  `trace_mode=decode_only` when the model advertises that requirement.
- Expanded `Llama3Generator.warmup_model_prefill()` coverage so `seq_len=128`
  traces batch sizes `1`, `2`, `4`, and `8` on T3K DP4. Longer supported
  prefill sequence lengths remain single-batch.
- Added a DP wrapper `already_warmed_up_prefill` property/setter so the Phase 1
  eager compile pass and Phase 2 trace pass both run on every lane.

Short validation:

- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_prefill_warmup_tail_20260709T175823Z/`.
- Server shape: T3K DP4, `--max-model-len 131072`, `--max_num_seqs 8`,
  fabric `FABRIC_1D`, trace mode `decode_only`, model warmup enabled, host
  sampling.
- Client workload: `64` random prompts, input length `2`, output length `128`,
  request rate `inf`, max concurrency `32`, `--ignore-eos`, `temperature=0`.
- Server startup took `66.26s`. All prefill trace captures happened before
  `Application startup complete`, including `seq_len=128` batch sizes
  `1/2/4/8`. Post-startup prefill compile/capture count was `0`.

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv2 DP4 broad prefill warmup smoke | 64 | 0 | 22.13 | 370.20 | 1210.81 | 1182.84 | 1322.40 | 77.55 | 84.64 |

This removes the visible TTFT tail for the smoke: the top TTFTs were clustered
around `1.31-1.32s`, and there was no measured-run prefill trace capture.

Full validation attempt:

- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_prefill_warmup_tail_20260709T180211Z/`.
- Same server shape as the parity run; client workload was `320` random
  prompts, input length `2`, output length `256`, request rate `inf`, max
  concurrency `32`, `--ignore-eos`, `temperature=0`.
- Server startup took `62.62s`. There was no post-startup prefill compile or
  capture (`0` after the API startup line).
- The TTFT distribution was fixed statistically: mean `1378.29 ms`, median
  `1202.56 ms`, p99 `1405.57 ms`.
- The run is not a valid parity throughput result. Three requests returned
  short outputs (`92/93` tokens instead of `256`), and the server reported
  `EngineCore_DP0 died unexpectedly` at the end. The client result file still
  reports `completed=320, failed=0`, but the output lengths prove the run was
  truncated.

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms | Output lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TTTv2 DP4 broad prefill warmup full attempt | 320 | 0 | 352.97 | 230.70 | 1378.29 | 1202.56 | 1405.57 | 79.59 | 86.93 | `317x256`, `2x93`, `1x92` |

Dead-end experiment:

- Tried narrowing decode-only retained prefill traces to `seq_len=128` only
  while still compiling the longer eager prefill shapes first.
- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_prefill_trace_tail_narrow_20260709T181332Z/`.
- Startup succeeded and traced only `seq_len=128` batch sizes `1/2/4/8`, but
  the server wedged on first traffic: the 64-prompt smoke had `0` successful
  requests and `64` connection resets, `/health` also reset, and DP0 stayed hot
  without new server log lines.
- That narrowing patch was reverted. The current code is back to the broad
  prefill trace warmup that passed the short TTFT validation.

Current status: the original TTFT tail from measured-run prefill trace capture
is understood and fixed for the short validation. The full parity rerun still
needs a separate late-run DP0 death investigation before the fix can be called
fully validated at `320 x 256`.

### TTTv2 DP4 late DP0 death investigation

The first full broad-prefill-warmup validation artifact
`/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_prefill_warmup_tail_20260709T180211Z/`
initially looked like a final-tail issue because only the last three requests
were short (`93`, `93`, and `92` output tokens). Request timing showed those
three requests were the final partial wave after the clean 32-request waves.
However, follow-up runs ruled out the final partial wave itself as the
sufficient cause.

Cold first-traffic wedge:

- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_late_dp0_67x256_20260709T182358Z/`.
- Server shape: T3K DP4, `--max-model-len 131072`, `--max_num_seqs 8`,
  fabric `FABRIC_1D`, trace mode `decode_only`, broad TTTv2 prefill warmup.
- The server reached `Application startup complete`, but a cold `67 x 256`
  client immediately got connection resets for every request.
- Result: `completed=0`, `failed=67`, duration `1.06s`.
- `/health` also reset after the failure. DP0 stayed hot and the server log did
  not show normal request-progress lines after startup.

Warm-server reduced checks:

- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_late_dp0_resmoke64_20260709T182751Z/`.
- A fresh reset and server start had a healthy `/health` response before
  traffic.
- A cold `64 x 128` smoke passed, but with one TTFT outlier:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv2 DP4 late-death resmoke `64 x 128` | 64 | 0 | 29.98 | 273.27 | 1385.19 | 1258.78 | 4929.61 | 76.90 | 84.68 |

- On that same already-warmed server, `64 x 256` passed:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv2 DP4 late-death resmoke `64 x 256` | 64 | 0 | 41.94 | 390.68 | 1215.04 | 1194.19 | 1409.41 | 77.45 | 81.55 |

- Also on the same already-warmed server, `67 x 256` passed:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv2 DP4 late-death resmoke `67 x 256` | 67 | 0 | 59.46 | 288.45 | 1089.45 | 1094.91 | 1140.30 | 76.50 | 76.92 |

These reduced passes show that output length `256` alone is not sufficient,
and the final partial wave of three requests is not sufficient once the server
has made it through earlier traffic.

Full warm-server repro:

- Still on the same server, a subsequent `320 x 256` run failed after five
  clean 32-request waves.
- Client result: `completed=192`, `failed=128`, duration `289.69s`, output
  throughput `142.22 tok/s`, mean TTFT `1173.26 ms`, p99 TTFT `1329.97 ms`.
  TPOT was polluted by the stall: mean `1141.03 ms`, p99 `27633.79 ms`.
- Output length by request wave:

| Request index range | Output length result |
|---|---|
| `0-31` | `32x256` |
| `32-63` | `32x256` |
| `64-95` | `32x256` |
| `96-127` | `32x256` |
| `128-159` | `32x256` |
| `160-191` | short outputs only: `22x8`, `6x7`, `2x4`, `1x6`, `1x9` |
| `192-319` | no valid chunk / no TTFT |

- Server progress showed all four engines with `8` running requests as
  generation throughput fell toward zero. vLLM then logged
  `Engine core proc EngineCore_DP0 died unexpectedly` at `18:39:20`.
- The vLLM multiprocessing monitor only reports the first process sentinel it
  observes as dead, and this log path does not include the process exit code.
  Given the progress lines, this should be treated as an all-engine stall with
  DP0 observed first, not proof that DP0 is the only failing lane.

Current conclusions:

- The TTFT prefill-trace tail and late DP death are separate issues.
- Broad prefill trace warmup removes measured-window prefill trace capture, but
  the full `320 x 256` DP4 workload is not stable yet.
- The strongest reduced signal so far is cumulative decode failure around the
  sixth 32-request wave, after roughly `160` complete `256`-token responses.
- A cold first-traffic path can also wedge, but a priming smoke can make the
  smaller `64 x 256` and `67 x 256` workloads pass.

Exit-code logging and `192 x 256` reduction:

- Code change: `vllm/v1/engine/core_client.py` now logs the dead EngineCore
  process `pid`, `exitcode`, decoded signal name for negative exit codes, and a
  status snapshot for all EngineCore processes when the MP client monitor sees
  a process sentinel fire.
- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_late_dp0_192x256_20260709T184652Z/`.
- Procedure: `tt-smi -r`, fresh T3K DP4 server on port `8093`, same broad
  prefill warmup config, `/health` check, `64 x 128` primer, `/health` check,
  then `192 x 256`.
- Primer result:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv2 DP4 late-death primer `64 x 128` | 64 | 0 | 21.79 | 375.95 | 1123.13 | 1073.98 | 1356.72 | 76.91 | 80.49 |

- Reduced run result:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv2 DP4 late-death reduction `192 x 256` | 192 | 0 | 143.48 | 342.58 | 1454.45 | 1143.31 | 21151.03 | 77.36 | 81.67 |

- Output length by request wave was clean:

| Request index range | Output length result |
|---|---|
| `0-31` | `32x256` |
| `32-63` | `32x256` |
| `64-95` | `32x256` |
| `96-127` | `32x256` |
| `128-159` | `32x256` |
| `160-191` | `32x256` |

- The server stayed healthy after the run; there was no EngineCore death, so
  the new exit-code logging did not trigger in this artifact.
- The high p99 TTFT came from three full-output requests with TTFT around
  `21-22s` (`request 94`, `95`, and `155`). That is a latency tail, not the
  late-death signature: TPOT remained normal and every request returned all
  `256` output tokens.

Updated conclusion: `64 x 128` primer plus `192 x 256` is not a reproducer for
the late death. The failing `320 x 256` run had more prior same-server history
before the failing workload (`64 x 128`, `64 x 256`, and `67 x 256`) and then
failed after another `160` full `256`-token responses. The next reduction
should target cumulative history directly, for example a second back-to-back
`192 x 256` run on the same server or a `256 x 256` run after the primer, and
should use the new exit-code logging to classify the first actual EngineCore
death.

`256 x 256` cumulative repro:

- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_late_dp0_256x256_20260709T185813Z/`.
- Procedure: `tt-smi -r`, fresh T3K DP4 server on port `8094`, same broad
  prefill warmup config, `/health` check, `64 x 128` primer, `/health` check,
  then `256 x 256`.
- Primer result:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv2 DP4 late-death primer `64 x 128` | 64 | 0 | 30.07 | 272.40 | 1397.27 | 1270.49 | 4937.19 | 76.84 | 84.71 |

- Cumulative run result:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms | Output lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TTTv2 DP4 late-death reduction `256 x 256` | 256 | 0 | 224.12 | 291.54 | 1416.01 | 1182.07 | 10335.11 | 78.52 | 89.15 | `253x256`, `3x191` |

- Output length by request wave:

| Request index range | Output length result |
|---|---|
| `0-31` | `32x256` |
| `32-63` | `32x256` |
| `64-95` | `32x256` |
| `96-127` | `32x256` |
| `128-159` | `32x256` |
| `160-191` | `32x256` |
| `192-223` | `32x256` |
| `224-255` | `29x256`, `3x191` |

- Short requests were `252`, `254`, and `255`, each with output length `191`.
  The client still reported `completed=256, failed=0`, so output-length checks
  are required for this failure mode.
- The server then died after the client had enough chunks to count the run as
  successful. The monitor logged:
  `Engine core proc EngineCore_DP0 died unexpectedly (pid=3652197, exitcode=None, signal=None), shutting down client. All engine process status: EngineCore_DP0(pid=3652197, exitcode=None, alive=True), EngineCore_DP1(pid=3652198, exitcode=None, alive=True), EngineCore_DP2(pid=3652199, exitcode=None, alive=True), EngineCore_DP3(pid=3652200, exitcode=None, alive=True)`.
- That means the first exit-code logging patch confirmed the monitor path and
  all-rank liveness snapshot, but did not classify the exit because
  `multiprocessing.Process.exitcode` had not populated yet when the sentinel
  was handled. The log patch was tightened after this run to do a non-blocking
  `join(timeout=0)` refresh before reading exit codes and signal names.

Updated reduction: `64 x 128` primer plus `256 x 256` is a useful reproducer
for the late failure shape. It is cheaper than the full `320 x 256` run and
gets a closely related symptom: final-wave truncation followed by EngineCore
death. The failure threshold is later than `192 x 256` on a freshly reset
server, and in this cleaner run the truncation moved to the last wave rather
than wave six.

Refreshed exit-code rerun:

- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_late_dp0_256x256_exitcode_20260709T190827Z/`.
- Procedure: `tt-smi -r`, fresh T3K DP4 server on port `8095`, same broad
  prefill warmup config and refreshed exit-code logging, `/health` check,
  `64 x 128` primer, `/health` check, then `256 x 256`.
- Primer result:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv2 DP4 refreshed-exitcode primer `64 x 128` | 64 | 0 | 22.14 | 369.98 | 1266.49 | 1324.15 | 1387.48 | 77.17 | 84.33 |

- Rerun result:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms | Output lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TTTv2 DP4 refreshed-exitcode `256 x 256` | 256 | 0 | 183.95 | 356.28 | 1301.69 | 1327.85 | 1568.38 | 77.24 | 81.65 | `256x256` |

- Output length by request wave was clean:

| Request index range | Output length result |
|---|---|
| `0-31` | `32x256` |
| `32-63` | `32x256` |
| `64-95` | `32x256` |
| `96-127` | `32x256` |
| `128-159` | `32x256` |
| `160-191` | `32x256` |
| `192-223` | `32x256` |
| `224-255` | `32x256` |

- The server stayed healthy after the run. There were no `Engine core proc`
  or `ERROR` log lines, so the refreshed exit-code logging did not trigger.
- The only notable tail was `request 222` with TTFT around `22.3s`; it still
  returned the full `256` output tokens and did not resemble the prior
  final-wave truncation.

Updated conclusion: `64 x 128` primer plus one `256 x 256` run is not
deterministic. The previous `256 x 256` artifact remains a valid failure
candidate, but capturing the refreshed exit code likely needs a repeated
same-server `256 x 256` loop, or a driver that stops on either short outputs or
the first EngineCore death.

Back-to-back `320 x 256` attempt:

- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_late_dp0_320x256_b2b_20260709T192646Z/`.
- Procedure: `tt-smi -r`, fresh T3K DP4 server on port `8096`, same broad
  prefill warmup config and refreshed exit-code logging, `/health` check,
  `64 x 128` primer, `/health` check, then planned back-to-back `320 x 256`.
- The first `320 x 256` run reproduced the failure, so the second run was not
  launched. The server returned `503` on the immediate post-run health check
  and then shut down. T3K was reset after the failure.
- Primer result:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TTTv2 DP4 `320 x 256` primer `64 x 128` | 64 | 0 | 22.40 | 365.78 | 1246.64 | 1217.62 | 1363.01 | 78.32 | 85.69 |

- First `320 x 256` result:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms | Output lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TTTv2 DP4 refreshed-exitcode `320 x 256` run 1 | 320 | 0 | 285.42 | 285.32 | 1285.17 | 1134.48 | 1415.06 | 98.79 | 112.36 | `318x256`, `2x14` |

- Output length by request wave:

| Request index range | Output length result |
|---|---|
| `0-31` | `32x256` |
| `32-63` | `32x256` |
| `64-95` | `32x256` |
| `96-127` | `32x256` |
| `128-159` | `32x256` |
| `160-191` | `32x256` |
| `192-223` | `32x256` |
| `224-255` | `32x256` |
| `256-287` | `32x256` |
| `288-319` | `30x256`, `2x14` |

- Short requests were `318` and `319`, each with output length `14`.
  The client still reported `completed=320, failed=0`, so this again confirms
  that output-length validation is required.
- Refreshed exit-code logging worked. The monitor logged:
  `Engine core proc EngineCore_DP0 died unexpectedly (pid=3672581, exitcode=-9, signal=SIGKILL), shutting down client. All engine process status: EngineCore_DP0(pid=3672581, exitcode=-9, signal=SIGKILL, alive=False), EngineCore_DP1(pid=3672582, exitcode=None, signal=None, alive=True), EngineCore_DP2(pid=3672583, exitcode=None, signal=None, alive=True), EngineCore_DP3(pid=3672584, exitcode=None, signal=None, alive=True)`.
- The final progress window showed Engine 000 and Engine 001 each stuck with
  `1` running request and `0.0` generation throughput before DP0 was killed.
  Engines 002 and 003 were already idle. This lines up with the two final
  short outputs and suggests the backend killed DP0 while the last two
  requests were still draining through the API layer.
- Two full-output requests still had high TTFT tails (`request 190` around
  `21.9s`, `request 254` around `22.3s`), but the actual failure signature was
  final-wave truncation plus DP0 `SIGKILL`, not a TTFT-only tail.

Updated conclusion: `320 x 256` is the stronger repro than `256 x 256`. A
single fresh-server `320 x 256` run was enough to reproduce the final-wave
short-output failure and classify the EngineCore death as DP0 `SIGKILL`.

TTTv1 `320 x 256` profile comparison:

- Artifact:
  `/localdev/gwang/vllm_duo/perf_results/dp_profile_tttv1_320x256_20260709T194414Z/`.
- Procedure: `tt-smi -r`, fresh T3K DP4 TTTv1 server on port `8097`,
  `TT_LLAMA_TEXT_VER=tt_transformers`, `TT_DP_PROFILE=1`, same
  `--max-model-len 131072`, `--data-parallel-size 4`, `--max_num_seqs 8`,
  fabric `FABRIC_1D`, trace mode `decode_only`, then one `320 x 256`
  benchmark. The server stayed healthy after the run and T3K was reset after
  shutdown.
- TTTv1 startup/warmup difference: TTTv1 warmed prefill lengths
  `128`, `1024`, `2048`, `4096`, `8192`, `16384`, `32768`, and `65536`,
  then warmed decode with page table shape `[32, 2048]`. The DP0 init path
  took `55.04s`.
- TTTv1 result:

| Run | Completed | Failed | Duration s | Output tok/s | Peak output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms | Output lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TTTv1 DP4 profiled `320 x 256` | 320 | 0 | 299.01 | 273.97 | 416.00 | 6996.29 | 2283.85 | 49342.92 | 84.01 | 83.85 | 91.22 | `320x256` |

- Output length by request wave was clean:

| Request index range | TTTv1 output length result | TTTv2 failed output length result |
|---|---|---|
| `0-31` | `32x256` | `32x256` |
| `32-63` | `32x256` | `32x256` |
| `64-95` | `32x256` | `32x256` |
| `96-127` | `32x256` | `32x256` |
| `128-159` | `32x256` | `32x256` |
| `160-191` | `32x256` | `32x256` |
| `192-223` | `32x256` | `32x256` |
| `224-255` | `32x256` | `32x256` |
| `256-287` | `32x256` | `32x256` |
| `288-319` | `32x256` | `30x256`, `2x14` |

- TTTv1 request throughput in this profiled run is not comparable to the
  historical baseline because `TT_DP_PROFILE=1` logging and first-traffic
  sampling warmup polluted TTFT and wall time. The steady decode metric is
  comparable: TTTv1 profiled mean TPOT was `84.01 ms`, matching the historical
  TTTv1 baseline (`85.63 ms`) and still much faster than the failed TTTv2
  `320 x 256` run's `98.79 ms`.
- TTTv1 final drain differs from the failed TTTv2 run. In the profiled TTTv1
  run, late progress showed the final active requests continuing to produce
  tokens and all engines eventually reached `Running: 0`. In the failed TTTv2
  run, Engine 000 and Engine 001 stayed at `Running: 1`, `Waiting: 0`, and
  `0.0` generation throughput for roughly `45s` before DP0 was killed.
- TTTv1 DP profile summary:

| Event | Count | Mean total ms | P50 total ms | P95 total ms | P99 total ms | Main subfields |
|---|---:|---:|---:|---:|---:|---|
| `dp_gather_submit` | 11300 | 25.09 | 21.62 | 45.49 | 48.54 | `build_model_input=11.68`, `execute_submit=4.99`, `decode_info_all_reduce=4.80`, `host_sample_params_gather=2.69` |
| `dp_gather_finalize` | 11300 | 36.65 | 36.43 | 43.33 | 49.02 | `ids_scatter=27.40`, `future_wait=9.04`, `apply_result=0.16` |

- Rank-level TTTv1 submit is asymmetric because rank 0 owns the model submit:
  rank 0 mean `dp_gather_submit` was `42.18 ms`, while ranks 1-3 were
  `19.35-19.42 ms`. Finalize was balanced across ranks at
  `36.60-36.70 ms`.
- Host kernel logs provide the strongest explanation for the TTTv2
  `SIGKILL`: near the TTTv2 failure window, `dmesg -T` recorded the Linux OOM
  killer selecting a `VLLM::EngineCor` task:
  `Out of memory: Killed process ... (VLLM::EngineCor) total-vm:587050252kB, anon-rss:488292896kB`.
  The kernel PID is from the host namespace, so it does not numerically match
  vLLM's container PID `3672581`, but the timestamp, process name, and
  observed `exitcode=-9/SIGKILL` line up with the failed TTTv2 run.

Updated conclusion: the DP0 death is host OOM, not an explicit vLLM kill or a
logged TT runtime exception. The next root-cause step should measure per-rank
RSS during the TTTv2 `320 x 256` repro and compare it with TTTv1, with special
attention to DP0/root gather state, TTTv2 per-lane output threading, and any
host-side accumulation that grows until the final-wave stall.

### TTTv2 DP4 host memory allocation audit

Static audit after the OOM finding separates the likely OOM-scale allocations
from small per-step DP plumbing:

- The DP gather tensor path is too small to explain the observed
  `anon-rss:488292896kB` OOM. For decode, `build_dp_decode_gather_input(...)`
  packs tokens, positions, block tables, sampling scalars, and slot remap into
  flat `int_inputs`/`float_inputs`. With `max_num_seqs=8` and
  `max_model_len=131072`, the block table dominates at roughly
  `8 * 4096` int32 entries per rank, about `128 KiB`; root stacking across
  four ranks is still only about `512 KiB` plus small float/scalar tensors.
- Host-only sampling metadata is gathered through `dist.gather_object` and then
  pickled/broadcast from root when sampling is on host. For the parity
  workload this should be modest unless structured-output masks, bad-words
  lists, or logits processors are unexpectedly large. It remains worth logging
  the pickled byte size, but it is not the first OOM suspect.
- DP0 is expected to be the memory-heavy process. In local DP, only local DP
  rank 0 opens the T3K mesh and loads/runs the merged TT model; the other local
  DP ranks return neutral payloads from `concat_and_execute_dp(...)`. For TTTv2
  DP4, that DP0-owned model is itself a wrapper over four single-lane
  generators, so DP0 holds the lane models, per-lane KV references, trace state,
  and output processing pool.
- The most credible OOM-scale path is full-vocab decode output lifetime. TTTv2
  DP wrapper decode returns one lane output per submesh when
  `read_from_device=False`; `read_decode_output(async_read=True)` reads those
  per-lane outputs in parallel and records events; `process_decode_output_host`
  processes every lane and then `torch.cat`s them into one merged logits tensor.
  For batch 8 and vocab 128256, a single merged float32 logits tensor is about
  `8 * 128256 * 4 = 3.9 MiB`; four lane tensors plus the merged tensor are
  about `7.8 MiB` if bf16/fp16 or `15.6 MiB` if float32, before temporary
  copies. Retaining this class of object over hundreds or thousands of decode
  steps can become OOM-scale, unlike the gather metadata.
- The async DP wrapper currently keeps large references until the deferred
  output object is released: `AsyncTTDPGatherOutput` stores the
  `TTDecodeSubmission` (`tt_out`, read events, sampling params) and
  `TTModelInput`, then caches the packed result after finalization. The engine
  currently finalizes the previous DP step before submitting the next, so the
  intended in-flight depth is one, but the deferred-output/cache/pending-list
  lifetime still needs RSS instrumentation to verify that completed full-vocab
  tensors are actually released promptly.
- Current source has only one `register_pending_async_step(...)` call in the
  DP async submit path. A duplicate pending-step registration is therefore not
  the active explanation in this checkout.

Next recommended experiment: add low-overhead RSS and object-lifetime
instrumentation around DP0 decode submit/finalize, including pending async
queue length, pickled host sampling metadata byte size, and per-process
`VmRSS`/`VmSize` sampled every N decode steps. If RSS climbs monotonically,
test an explicit release after `AsyncTTDPGatherOutput` finalizes by clearing
the submission/model-input references once `_cached_output` has been built.

### TTTv2 DP4 memory-profiled 320x256 repro

Artifact:
`/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_mem_profile_320x256_20260709T204934Z/`.

Instrumentation added for this run:

- `TT_DP_MEM_PROFILE=1` and `TT_DP_MEM_PROFILE_INTERVAL=16`.
- Engine `TT_DP_PROFILE_JSON` events include periodic `VmRSS`, `VmHWM`, and
  `VmSize` fields.
- Runner `TT_DP_MEM_PROFILE_JSON` events log pending async queue length,
  completed decode queue length, pending sample count, and materialized torch
  output bytes around async decode registration/finalization.
- Root-side gather serialization logs include pickled sampling metadata sizes
  where root actually serializes those payloads.

Procedure:

- `tt-smi -r`.
- Fresh TTTv2 DP4 server on port `8098` with:
  `TT_LLAMA_TEXT_VER=common_llama3_8b`, `TT_DP_PROFILE=1`,
  `TT_DP_MEM_PROFILE=1`, `TT_DP_MEM_PROFILE_INTERVAL=16`,
  `--max-model-len 131072`, `--data-parallel-size 4`, `--max_num_seqs 8`,
  fabric `FABRIC_1D`, trace mode `decode_only`, trace region size `85000000`.
- `/health` reached `HTTP 200`.
- Ran the same `64 x 128` primer, checked health, then ran `320 x 256`.
- T3K was reset after the failure.

Primer result:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | P99 TPOT ms | Output lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TTTv2 DP4 mem-profile primer `64 x 128` | 64 | 0 | 30.27 | 270.62 | 1402.07 | 1274.11 | 4967.46 | 77.49 | 85.34 | `64x128` |

`320 x 256` result:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms | Output lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TTTv2 DP4 mem-profile `320 x 256` | 318 | 2 | 348.48 | 226.03 | 1302.66 | 1214.60 | 1498.20 | 107.69 | 78.49 | 973.21 | `286x256`, mixed short/fail tail |

Output length by request wave:

| Request index range | Output length result |
|---|---|
| `0-31` | `32x256` |
| `32-63` | `32x256` |
| `64-95` | `32x256` |
| `96-127` | `32x256` |
| `128-159` | `32x256` |
| `160-191` | `32x256` |
| `192-223` | `32x256` |
| `224-255` | `32x256` |
| `256-287` | `30x256`, `1x216`, `1x209` |
| `288-319` | `9x217`, `4x216`, `2x215`, `2x0`, plus many single short lengths |

Key memory-profile findings:

- The primer already showed the leak/growth shape: the first sampled DP0
  memory event had `pending_async_steps=6`, `VmRSS=27,386,096 kB`; by the end
  of the primer DP0 reached roughly `91,523,208 kB` RSS in engine profile
  samples.
- During `320 x 256`, DP0 RSS grew monotonically with pending async steps:
  around `30%` progress, `pending_async_steps=1444` and
  `VmRSS=258,701,140 kB`; around `80%`, `pending_async_steps=2900` and
  `VmRSS=492,351,204 kB`.
- The last runner memory event before death had `pending_async_steps=2911`,
  `VmHWM=492,748,672 kB`, and `VmRSS=489,377,276 kB`.
- Every sampled `decode_finalize_done` event reported
  `tt_out_torch_bytes=16,416,768` bytes, matching a merged batch-32 full-vocab
  float32 logits tensor. This is direct evidence that completed async wrappers
  are retaining full-vocab outputs.
- `completed_decode_steps` stayed `0` in all sampled runner memory events, so
  the growth is not from the completed-decode application queue.
- The DP async pending list was not pruning during the run: the sampled pending
  count grew from `6` to `2911`, and no `async_prune_pending` events appeared
  in the profile. This is the immediate root cause for the OOM-scale host
  retention.
- Server exit-code logging again showed DP0 killed by signal:
  `EngineCore_DP0(pid=3731147, exitcode=-9, signal=SIGKILL, alive=False)`,
  while DP1-DP3 were still alive.
- Host kernel log confirmed OOM:
  `Out of memory: Killed process 418371 (VLLM::EngineCor) total-vm:588501368kB, anon-rss:489056076kB`.

Updated conclusion: the DP0 OOM is caused by unbounded retention of finalized
async DP decode wrappers. Each retained wrapper can cache a packed result while
also holding references to its submission/model input; in this workload the
sampled finalized full-vocab tensor alone is `16.4 MB` per decode step. The
fix direction is now concrete: ensure resolved async DP steps are pruned, and
release large submission/model-input references after finalization. Then rerun
the same `64 x 128` primer first; success criteria are that
`pending_async_steps` stays near the intended in-flight depth and DP0 RSS
stops growing linearly with decode steps before attempting `320 x 256` again.

### TTTv2 DP4 async-prune fix and verification

Implemented fix:

- Added a deferred-output finalization hook.
- `AsyncTTDPGatherOutput` now clears its large `TTDecodeSubmission` and
  `TTModelInput` references after building the cached packed DP result.
- `AsyncTTDPGatherOutput` now prunes resolved pending async steps after
  finalization, so the runner pending list no longer retains every completed
  DP decode wrapper.
- Temporary memory-profile instrumentation was used for the verification runs
  below and later removed from the production diff.

Verification artifact:
`/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_async_prune_fix_64x128_20260709T2110Z/`.

Procedure:

- Fresh TTTv2 DP4 server on port `8099` with the same T3K, model, DP4,
  `max_model_len=131072`, `max_num_seqs=8`, fabric, decode trace, and memory
  profile settings as the failing memory-profiled run.
- `/health` reached `HTTP 200`.
- Ran fixed `64 x 128` primer.
- Ran fixed `320 x 256` on the same server.
- Stopped the server and reset T3K after verification.

Fixed primer result:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms | Output lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TTTv2 DP4 async-prune fixed `64 x 128` | 64 | 0 | 30.10 | 272.14 | 1404.72 | 1293.52 | 4971.79 | 77.13 | 77.08 | 85.07 | `64x128` |

Fixed `320 x 256` result:

| Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms | Output lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TTTv2 DP4 async-prune fixed `320 x 256` | 320 | 0 | 226.39 | 361.86 | 1356.37 | 1175.39 | 1380.07 | 77.03 | 76.99 | 80.64 | `320x256` |

Memory verification:

| Run state | Max pending async steps | Max DP0 VmRSS kB | Max DP0 VmHWM kB | Outcome |
|---|---:|---:|---:|---|
| Before fix, primer + failing `320 x 256` | 2911 | 492,351,204 | 492,748,672 | DP0 OOM/SIGKILL |
| After fix, primer + passing `320 x 256` | 1 | 27,716,324 | 27,716,324 | Pass |

Object lifetime before the fix:

```text
TTAsyncDecodeController
  `- runner._pending_async_steps
       |- AsyncTTDPGatherOutput step 1  [resolved, still retained]
       |    |- _submission
       |    |    `- tt_out / read_events / sampling_params
       |    |- _model_input
       |    `- _cached_output
       |- AsyncTTDPGatherOutput step 2  [resolved, still retained]
       |    |- _submission
       |    |- _model_input
       |    `- _cached_output
       |- ...
       `- AsyncTTDPGatherOutput step 2911
            |- _submission
            |- _model_input
            `- _cached_output
```

Effect before the fix:

```text
_pending_async_steps grows every decode step
resolved wrappers remain strongly referenced
large per-step decode objects cannot be freed
RSS climbs until OOM

pending_async_steps: 6 -> 1444 -> 2911
DP0 RSS:             27 GB -> 259 GB -> ~492 GB
```

Object lifetime after the fix:

```text
TTAsyncDecodeController
  `- runner._pending_async_steps
       `- AsyncTTDPGatherOutput current step  [in flight or just resolving]
            |- _submission     -> cleared after finalization
            |- _model_input    -> cleared after finalization
            `- _cached_output  -> small packed DP token result
```

Finalization flow after the fix:

```text
ensure_finalized()
  |- _cached_output = _get_output_impl()
  |    |- finalize_decode(submission)
  |    |- sample/pack DP token result
  |    `- after successful output build:
  |         |- _submission = None
  |         `- _model_input = None
  |- _completion_event.set()
  `- _on_finalized()
       `- prune_finished_async_events()
            `- popleft resolved wrappers from _pending_async_steps
```

Safety note after review: the heavy references are not cleared from a
`finally` block. If `finalize_decode`, `_get_output_tokens`, or
`pack_dp_results` raises, `_finalized` remains false and `_submission` /
`_model_input` remain attached to the wrapper for retry and debugging. They are
only dropped after the packed output has been built successfully.

Updated conclusion: the fix eliminates the OOM-scale host retention and also
recovers the lost throughput. The fixed TTTv2 DP4 `320 x 256` run completed
all outputs at `361.86 tok/s`, with TPOT `77.03 ms`, versus the failed
pre-fix profiled run's `226.03 tok/s`, TPOT `107.69 ms`, and DP0 OOM. This
also beats the profiled TTTv1 `320 x 256` steady TPOT (`84.01 ms`), though
that TTTv1 run had profile/first-traffic TTFT pollution and should not be used
as a final apples-to-apples throughput gate.

### TTTv2 DP4 clean full parity benchmark after async-prune fix

Artifact:
`/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_async_prune_clean_20260709T212127Z/`.

Procedure:

- `tt-smi -r`.
- Fresh TTTv2 DP4 server on port `8100` with the exact parity shape:
  `TT_LLAMA_TEXT_VER=common_llama3_8b`, `--max-model-len 131072`,
  `--data-parallel-size 4`, `--max_num_seqs 8`, fabric `FABRIC_1D`, trace mode
  `decode_only`, trace region size `85000000`, default warmup enabled, host
  sampling.
- No `TT_DP_PROFILE` or `TT_DP_MEM_PROFILE`, so this is the clean performance
  run after the memory/lifecycle fix.
- `/health` reached `HTTP 200`.
- Client workload: OpenAI completions backend, random input length `2`, random
  output length `256`, `320` prompts, request rate `inf`, max concurrency `32`,
  `--ignore-eos`, `temperature=0`.
- Post-run `/health` was still `HTTP 200`; server logs had no `ERROR`,
  `EngineDead`, `SIGKILL`, or OOM lines. Server stopped cleanly and T3K was
  reset after the run.

Clean full parity result:

| Implementation | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Peak output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms | Output lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TTTv1 baseline | 320 | 0 | 242.03 | 1.32 | 338.47 | 339.79 | 408.00 | 2364.95 | 2312.97 | 3792.21 | 85.63 | 84.91 | 97.10 | `320x256` |
| TTTv2 parallel lane output, before async-prune fix | 320 | 0 | 241.54 | 1.32 | 339.15 | 340.48 | 440.00 | 1689.16 | 1200.45 | 22166.26 | 81.25 | 79.68 | 95.46 | `320x256` |
| TTTv2 async-prune fixed, clean run | 320 | 0 | 225.19 | 1.42 | 363.78 | 365.20 | 448.00 | 1343.98 | 1208.64 | 1367.72 | 76.62 | 76.50 | 80.41 | `320x256` |

Current parity conclusion:

- Functional parity: pass. The clean fixed run completed all `320` requests
  with zero failures and all outputs at `256` tokens.
- Decode throughput parity: pass. TTTv2 now exceeds the documented TTTv1
  baseline output throughput by `7.5%` (`363.78 / 338.47`) and improves mean
  TPOT by `10.5%` (`85.63 -> 76.62 ms`).
- TTFT tail: improved versus the previous parallel-output run. P99 TTFT is now
  `1367.72 ms`, not the earlier `22166.26 ms` trace-tail outlier.
- RSS comparison: this clean performance run intentionally did not enable
  `TT_DP_MEM_PROFILE`, so it has no per-process RSS samples in its server log.
  The fixed verification run immediately before it did enable memory profiling
  and is the comparable RSS datapoint: DP0 max `VmRSS`/`VmHWM` was
  `27,716,324 kB` after the async-prune fix, versus the broken run's
  `492,351,204 kB` max sampled `VmRSS`, `492,748,672 kB` `VmHWM`, and kernel
  OOM `anon-rss:489,056,076 kB`.

### TTTv2 DP4 quick smoke after safer cleanup patch

Artifact:
`/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_quick_smoke_safer_cleanup_20260709T214947Z/`.

Purpose: validate the reviewed cleanup change where `AsyncTTDPGatherOutput`
clears `_submission` and `_model_input` only after successfully building the
packed output, instead of clearing them from a `finally` path.

Procedure:

- `tt-smi -r`.
- Fresh TTTv2 DP4 server on port `8101` with:
  `TT_LLAMA_TEXT_VER=common_llama3_8b`, `TT_DP_PROFILE=1`,
  `TT_DP_MEM_PROFILE=1`, `TT_DP_MEM_PROFILE_INTERVAL=16`,
  `--max-model-len 131072`, `--data-parallel-size 4`, `--max_num_seqs 8`,
  fabric `FABRIC_1D`, trace mode `decode_only`, trace region size `85000000`.
- `/health` reached `HTTP 200`.
- Client workload: OpenAI completions backend, random input length `2`, random
  output length `128`, `64` prompts, request rate `inf`, max concurrency `32`,
  `--ignore-eos`, `temperature=0`.
- Post-run `/health` was still `HTTP 200`; server and client logs had no
  `ERROR`, `Traceback`, `SIGKILL`, `Killed`, `EngineCore.*died`, OOM, or
  `OutOfMemory` matches. Server stopped and T3K was reset after the run.

Smoke result:

| Run | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms | Output lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TTTv2 DP4 safer-cleanup smoke `64 x 128` | 64 | 0 | 21.76 | 2.94 | 376.54 | 379.48 | 1114.53 | 1069.19 | 1340.40 | 76.84 | 76.53 | 81.61 | `64x128` |

Memory/lifecycle checks from `TT_DP_MEM_PROFILE_JSON`:

- `async_prune_pending` events: `64`.
- Max sampled `pending_async_steps`: `0` at prune points.
- Max sampled DP0 `VmRSS`/`VmHWM`: `26,879,476 kB`.

Conclusion: the safer cleanup patch preserves the async-prune behavior and the
quick DP4 smoke passes with no failed requests, no server health regression,
and no renewed pending-wrapper retention.

### Debug instrumentation cleanup

After the safer-cleanup smoke passed, the temporary profiling probes were
removed from the source tree:

- Removed `TT_DP_MEM_PROFILE` / `TT_DP_MEM_PROFILE_INTERVAL` handling from
  `TTModelRunner`.
- Removed `TT_DP_MEM_PROFILE_JSON` pending-wrapper/RSS/tensor-byte logging from
  `async_decode.py`.
- Removed `TT_DP_PROFILE_JSON` timing/RSS payload generation from
  `TTDPEngineCoreProc`.

The functional fix remains: `AsyncTTDPGatherOutput` finalization still prunes
resolved pending async wrappers and drops large `_submission` / `_model_input`
references only after the packed output has been built successfully.

### TTTv2 DP4 full parity benchmark after instrumentation cleanup

Artifact:
`/localdev/gwang/vllm_duo/perf_results/tttv2_dp4_parity_after_cleanup_20260709T220003Z/`.

Purpose: rerun the full parity workload after removing the temporary profiling
instrumentation from the source tree.

Procedure:

- `tt-smi -r`.
- Fresh TTTv2 DP4 server on port `8102` with:
  `TT_LLAMA_TEXT_VER=common_llama3_8b`, `--max-model-len 131072`,
  `--data-parallel-size 4`, `--max_num_seqs 8`, fabric `FABRIC_1D`, trace mode
  `decode_only`, trace region size `85000000`.
- No `TT_DP_PROFILE` or `TT_DP_MEM_PROFILE` env vars.
- `/health` reached `HTTP 200`.
- Client workload: OpenAI completions backend, random input length `2`, random
  output length `256`, `320` prompts, request rate `inf`, max concurrency `32`,
  `--ignore-eos`, `temperature=0`.
- Post-run `/health` stayed `HTTP 200`; server and client logs had no `ERROR`,
  `CRITICAL`, `Traceback`, `SIGKILL`, `Killed`, `EngineCore.*died`, OOM,
  `OutOfMemory`, or `EngineDead` matches. Server stopped and T3K was reset
  after the run.
- Confirmed no `TT_DP_PROFILE_JSON` or `TT_DP_MEM_PROFILE_JSON` lines appeared
  in the server/client logs.

Result:

| Run | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Peak output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms | Output lengths |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TTTv2 DP4 after instrumentation cleanup `320 x 256` | 320 | 0 | 225.29 | 1.42 | 363.62 | 365.04 | 448.00 | 1373.23 | 1210.43 | 1460.32 | 76.62 | 76.47 | 80.54 | `320x256` |

Conclusion: full DP4 parity still passes after removing the temporary debug
instrumentation. Throughput is effectively unchanged versus the prior clean
post-fix run (`363.62` vs `363.78` output tok/s, `76.62` ms mean TPOT in both
runs), and the debug JSON probes are absent from the logs.

### TTTv2 N150/N300 parity plan

Goal: extend the TTTv2 Llama 3.1 8B vLLM path
(`TT_LLAMA_TEXT_VER=common_llama3_8b`) from the validated T3K path to N150 and
N300, then compare against the documented TTTv1 host-sampling fallback
baselines in section 10.

Baseline targets from TTTv1 no-sample runs:

| Device | TTTv1 target shape | Completed | Failed | Output tok/s | Mean TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---|---:|---:|---:|---:|---:|---:|
| N150 | DP1, `max_model_len=32768`, `max_num_seqs=32` | 320 | 0 | 620.92 | 2115.17 | 43.44 | 47.95 |
| N300 | DP2, `max_model_len=32768`, `max_num_seqs=16` | 320 | 0 | 428.27 | 3196.13 | 57.89 | 68.68 |

Initial TTTv2 shapes:

| Device | TTTv2 starting shape | Reasoning |
|---|---|---|
| N150 | DP1, `max_model_len=32768`, `max_num_seqs=32`, `trace_mode=decode_only`, host sampling | N150 has one Wormhole device, so TTTv2 should use the single-lane `Llama3Generator` path. Match the TTTv1 host-sampling fallback target and avoid the previously unstable on-device sampling path. |
| N300 | DP2, `max_model_len=32768`, `max_num_seqs=16`, `trace_mode=decode_only`, host sampling | N300 has two devices, and TTTv1 used two one-device DP replicas. TTTv2 should use the gathered-DP wrapper with two one-device lanes. |

Server commands to validate first:

```bash
# N150 TTTv2, host sampling
PYTHONPATH=/localdev/gwang/vllm_duo/vllm:\
/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src:\
/localdev/gwang/vllm_duo/tt-metal-too \
HF_HOME=/proj_sw/user_dev/huggingface \
TT_CACHE_PATH=/localdev/gwang/vllm_duo/tt_cache/meta-llama--Llama-3.1-8B-Instruct \
VLLM_RPC_TIMEOUT=300000 \
MESH_DEVICE=N150 \
TT_LLAMA_TEXT_VER=common_llama3_8b \
/localdev/gwang/vllm_duo/tt-metal-too/python_env/bin/python \
  plugins/vllm-tt-plugin/examples/server_example_tt.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 \
  --max-model-len 32768 \
  --max_num_seqs 32 \
  --async-scheduling \
  --additional-config '{"tt":{"trace_mode":"decode_only"}}'
```

```bash
# N300 TTTv2, host sampling
PYTHONPATH=/localdev/gwang/vllm_duo/vllm:\
/localdev/gwang/vllm_duo/vllm/plugins/vllm-tt-plugin/src:\
/localdev/gwang/vllm_duo/tt-metal-too \
HF_HOME=/proj_sw/user_dev/huggingface \
TT_CACHE_PATH=/localdev/gwang/vllm_duo/tt_cache/meta-llama--Llama-3.1-8B-Instruct \
VLLM_RPC_TIMEOUT=300000 \
MESH_DEVICE=N300 \
TT_LLAMA_TEXT_VER=common_llama3_8b \
/localdev/gwang/vllm_duo/tt-metal-too/python_env/bin/python \
  plugins/vllm-tt-plugin/examples/server_example_tt.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 \
  --max-model-len 32768 \
  --data-parallel-size 2 \
  --max_num_seqs 16 \
  --async-scheduling \
  --additional-config '{"tt":{"trace_mode":"decode_only"}}'
```

Client workload for parity, matching section 10:

```bash
/localdev/gwang/vllm_duo/tt-metal-too/python_env/bin/python \
  -m vllm.entrypoints.cli.main bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 2 \
  --random-output-len 256 \
  --num-prompts 320 \
  --max-concurrency 32 \
  --request-rate inf \
  --temperature 0 \
  --ignore-eos \
  --save-result \
  --save-detailed
```

Validation ladder:

1. Run `tt-smi -r` before each platform attempt and record the artifact
   directory.
2. Start with a one-request smoke after `/health` reaches `HTTP 200`; expected
   result is HTTP 200, coherent output, and no server `ERROR`, `Traceback`,
   `EngineDead`, OOM, or `SIGKILL`.
3. Run a short `64 x 128` smoke if the one-request smoke passes. This catches
   async lifecycle regressions without spending the full benchmark time.
4. Run the full `320 x 256` parity benchmark.
5. Compare output throughput and mean TPOT to the TTTv1 target for that same
   device. Treat performance as on-par if TTTv2 is within measurement noise of
   TTTv1 or faster; otherwise profile TTTv1 and TTTv2 on that device and chase
   bottlenecks one at a time.
6. Stop the server, reset the hardware, and document commands, logs, results,
   failures, fixes, and final parity status.

Expected risk areas:

- N150 may still expose platform-specific L1 or ARC instability that was
  previously seen when trying on-device sampling. The first pass intentionally
  avoids `sample_on_device_mode` because the TTTv1 target here is host sampling.
- N300 DP2 will exercise the TTTv2 DP wrapper with two one-device lanes instead
  of T3K's four two-device lanes. If it fails, inspect submesh creation,
  per-lane cache sizing, and lane row/slot mapping first.
- The TTTv2 `get_max_tokens_all_users()` policy currently returns
  `max_model_len`, which matches the fixed T3K DP4 contract. If N150/N300 block
  counts differ from TTTv1 expectations, inspect the plugin-side token-budget
  guard and model-side paged-KV block calculation before changing model runtime
  code.

### TTTv2 N150/N300 parity results

Artifacts:

- N150 first failed launch:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_parity_20260709T222917Z/`.
- N150 passing run:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_parity_trace85m_20260709T223347Z/`.
- N300 passing run:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n300_parity_trace85m_20260709T223918Z/`.

N150 first attempt failed before `/health` during prefill trace capture:

```text
Creating trace buffers of size 50323456B on MeshDevice 0, but only
50000000B is allocated for trace region.
```

This was a config/runtime sizing issue, not a model correctness failure. The
passing N150 and N300 runs used `trace_region_size=85000000`, matching the
validated T3K configuration.

Validation procedure for each passing platform:

- `tt-smi -r`.
- Fresh server with `TT_LLAMA_TEXT_VER=common_llama3_8b`,
  `trace_mode=decode_only`, `trace_region_size=85000000`, host sampling, and
  the same `320 x 256` benchmark workload as the TTTv1 baselines.
- One-request completion smoke: HTTP 200 with coherent output.
- `64 x 128` smoke.
- Full `320 x 256` parity benchmark.
- Post-run `/health` stayed `HTTP 200`; server/client logs had no `ERROR`,
  `CRITICAL`, `Traceback`, `SIGKILL`, `Killed`, `EngineCore.*died`, OOM,
  `OutOfMemory`, or `EngineDead` matches.
- Server stopped and hardware reset after each platform.

Smoke results:

| Device | Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Mean TPOT ms | P99 TPOT ms | Output lengths |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| N150 | TTTv2 `64 x 128` | 64 | 0 | 13.46 | 608.83 | 1461.88 | 41.44 | 51.68 | `64x128` |
| N300 | TTTv2 DP2 `64 x 128` | 64 | 0 | 15.35 | 533.62 | 1641.64 | 47.48 | 57.81 | `64x128` |

Full parity results:

| Device | Implementation | Shape | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Peak output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms | Output lengths |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| N150 | TTTv1 baseline | DP1, `32768/32` | 320 | 0 | 131.93 | 2.43 | 620.92 | 623.34 | 794.00 | 2115.17 | n/a | n/a | 43.44 | n/a | 47.95 | `320x256` |
| N150 | TTTv2 | DP1, `32768/32`, `trace_region_size=85000000` | 320 | 0 | 121.89 | 2.63 | 672.10 | 674.73 | 800.00 | 1447.11 | 1489.44 | 1502.10 | 42.12 | 41.95 | 47.29 | `320x256` |
| N300 | TTTv1 baseline | DP2, `32768/16` | 320 | 0 | 191.28 | 1.67 | 428.27 | 429.95 | 592.00 | 3196.13 | n/a | n/a | 57.89 | n/a | 68.68 | `320x256` |
| N300 | TTTv2 | DP2, `32768/16`, `trace_region_size=85000000` | 320 | 0 | 137.46 | 2.33 | 595.96 | 598.29 | 704.00 | 1612.03 | 1648.01 | 2093.00 | 47.58 | 47.33 | 52.74 | `320x256` |

Parity conclusion:

- N150 functional parity: pass. TTTv2 completed all `320` requests with zero
  failures and all outputs at `256` tokens.
- N150 throughput parity: pass. TTTv2 output throughput is `8.2%` higher than
  TTTv1 (`672.10 / 620.92`) and mean TPOT improves from `43.44 ms` to
  `42.12 ms`.
- N300 functional parity: pass. TTTv2 DP2 completed all `320` requests with
  zero failures and all outputs at `256` tokens.
- N300 throughput parity: pass. TTTv2 output throughput is `39.2%` higher than
  TTTv1 (`595.96 / 428.27`) and mean TPOT improves from `57.89 ms` to
  `47.58 ms`.

N150/N300 now have the same practical TTTv2 vLLM status as T3K for the
host-sampling parity workload: functional parity passes and throughput is
on-par or better than the documented TTTv1 baseline.

### TTTv2 N150/N300 `sample_on_device_mode=all` results

Question answered: N150 and N300 can run TTTv2 Llama 3.1 8B vLLM with
`sample_on_device_mode=all` at functional and throughput parity. N300 required
a split sampling path: force the greedy-equivalent top-k path for prefill, where
force-argmax produced invalid sampled token ids, but leave decode on the faster
force-argmax path. The N150 first-token quality oracle now matches host greedy
sampling after fixing the prefill Sampling1D batch contract.

Artifacts:

- N150 initial failing run:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_sample_all_20260709T225535Z/`.
- N150 after decode-token host reshape fix, still failing batch decode:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_sample_all_fix_tokens_20260709T225725Z/`.
- N150 after sampled-token packing and feedback fixes:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_sample_all_no_feedback_20260709T230425Z/`.
- N300 after sampled-token packing and feedback fixes:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n300_sample_all_no_feedback_20260709T230905Z/`.
- N150 root-cause fix preserving device feedback with reshape/copy:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_sample_all_feedback_copy_20260709T233043Z/`.
- N300 root-cause fix preserving device feedback with reshape/copy:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n300_sample_all_feedback_copy_20260709T233634Z/`.
- N150 output-quality host oracle:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_quality_host_20260709T234144Z/`.
- N150 output-quality `sample_on_device_mode=decode_only` phase split:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_quality_decode_only_20260709T234518Z/`.
- N150 output-quality `sample_on_device_mode=all` before prefill row fix:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_quality_sample_all_20260709T234318Z/`.
- N150 output-quality `sample_on_device_mode=all` after logits-row selection
  fix, including the `logprobs=0` forced-host oracle:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_quality_sample_all_prefill_rowfix_20260709T234834Z/`.
- N150 output-quality `sample_on_device_mode=all` after prefill row/logits
  selection work:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_quality_sample_all_prefill_hiddenfix_20260709T235248Z/`.
- N150 output-quality `sample_on_device_mode=all` after padding selected
  single-user prefill logits to Sampling1D's 32-row parameter/buffer contract:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_quality_sample_all_prefill_pad32_20260710T001739Z/`.
- N150 refreshed full `sample_on_device_mode=all` run after the prefill pad32
  fix:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n150_full_parity_prefill_pad32_20260710T002512Z/`.
- N300 full `sample_on_device_mode=all` run after removing the stale DP
  model-side sampling guard:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n300_full_parity_dp_sampling_fix_20260710T003552Z/`.
- N300 invalid-token debug repro after switching non-Galaxy sampling from
  force-argmax to top-k:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n300_invalid_token_topk_batchshape_20260710T010317Z/`.
- N300 reduced clean repro after fixing sampled-token axis handling:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n300_invalid_token_tokenaxis_20260710T010714Z/`.
- N300 clean full `sample_on_device_mode=all` run after the top-k and token-axis
  fixes:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n300_full_parity_sample_all_clean_20260710T011005Z/`.
- N300 top-k decode finalize profile:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n300_topk_finalize_profile_20260710T012222Z/`.
- N300 host-sampling finalize profile:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n300_host_finalize_profile_20260710T012504Z/`.
- N300 TTTv1 host-sampling finalize profile:
  `/localdev/gwang/vllm_duo/perf_results/tttv1_n300_host_profile_hfmodel_20260710T012925Z/`.
- N300 split prefill-top-k/decode-argmax smoke profile:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n300_split_prefill_topk_decode_argmax_20260710T013358Z/`.
- N300 clean full split prefill-top-k/decode-argmax run:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_n300_split_prefill_topk_decode_argmax_full_20260710T013807Z/`.

Fixes made:

- `tt-metal-too/models/common/models/executor.py`:
  `_process_output_decode_tokens()` no longer hardcodes a reshape to
  `[1, 1, 32, 1]`. TTTv2 on-device sampling can return compact token tensors,
  so decode-token host processing now accepts either the legacy 4D layout or a
  compact flattened layout.
- `plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py`:
  `_get_output_tokens()` now normalizes on-device sampled tokens by inspecting
  the tensor shape instead of always row-indexing by the vLLM batch rows. This
  fixed the previous `IndexError` / under-sized sampled-token tensor crash.
- `tt-metal-too/models/common/models/executor.py`:
  traced decode device feedback now keeps separate logical buffers for sampler
  output and decode input. Root cause: `tt_out_tok` is the sampler op's output
  tensor, but the traced-feedback path passed `tt_tokens`, which is the model
  decode-input tensor. Those tensors both carry token ids, but they do not have
  the same layout contract. Reusing `tt_tokens` as `tt_out_tok` caused the
  sampler to return an under-sized/scalar logical tensor on single-device lanes.
  The real fix is to let `Sampling1D` produce its natural sampled-token tensor,
  reshape that tensor to the decode-input shape, and device-to-device copy it
  into the persistent `tt_tokens` trace input for the next replay. This preserves
  device feedback instead of disabling it.
- `tt-metal-too/models/common/models/executor.py`:
  single-user prefill now pads the selected logits row from `[1, 1, 1, V]` to
  `[1, 1, 32, V]` before calling `Sampling1D`. Root cause: the selected prefill
  row was correct, and request params were greedy-equivalent, but
  `_get_decode_sampling_kpt()` caches `k/p/temp` tensors for
  `model.sampling.config.max_batch_size` (`32` on these runs). The
  `ttnn.sampling` op expects its parameter length to match the logits user
  dimension. Decode already uses the fixed 32-row contract, but single-user
  prefill had been passing a 1-row logits tensor into the same 32-row sampler
  contract. Padding the logits batch dimension preserves row 0, aligns the
  sampler parameter/buffer shape, and the runner still extracts only the real
  sampled token.
- `tt-metal-too/models/common/models/llama3_8b/model.py`:
  non-Galaxy decode uses the fast force-argmax path again. Prefill call sites
  in `executor.py` force the greedy-equivalent top-k path instead. Root cause:
  N300 DP2 produced invalid negative sampled token ids from prefill with
  force-argmax enabled, but forcing top-k for decode made every decode step wait
  on the slower device-side Sampling1D top-k path. The split path keeps the
  correctness fix scoped to prefill and preserves decode throughput.
- `tt-metal-too/models/common/modules/sampling/sampling_1d.py`:
  Sampling1D top-k now slices the persistent 4D local-index and global-offset
  buffers to the active user count before `ttnn.topk()` / `ttnn.add()`.
  Root cause: after disabling force-argmax, N300 DP2 warmup hit
  `k must have shape [16]`, then `Invalid subtile broadcast type` because
  logits and `k/p/temp` were lane-local (`16` users) while `_local_indices` and
  `_index_offsets` were materialized for the max sampler batch.
- `tt-metal-too/models/common/models/executor.py`:
  `_process_output_decode_tokens()` now handles both `[1, 1, B, 1]` and
  `[1, 1, 1, B]` sampled-token layouts before falling back to compact flattening.
  Root cause: the N300 DP wrapper processed one token per lane from the top-k
  sampled-token output, producing a gathered tensor of shape `(2,)` and crashing
  with `TT sampled-token tensor is too small for rank 0: shape=(2,), start=0,
  sz=16`.
- `tt-metal-too/models/common/models/executor.py`:
  `_sampling_decode_forward()` now accepts `force_topk`. Prefill sampling passes
  `force_topk=True`; decode leaves it false. `_get_decode_sampling_kpt()` caches
  top-k parameter tensors by logical batch size and by the forced-top-k flag, so
  prefill can use top-k while decode still returns `None` for the force-argmax
  fast path when greedy params allow it.
- `tt-metal-too/models/common/models/generator.py`:
  the gathered-DP wrapper no longer rejects `can_sample_on_device` during
  warmup and no longer rejects runtime `sampling_params`; it slices per-row
  sampling parameters for each lane. This lets the N300 DP2 `all` server start,
  and the split prefill-top-k/decode-argmax full run is correctness-clean.

Failure progression:

- Initial N150 startup failed during warmup prefill sampling:

```text
TT_FATAL ... physical_data.size() == physical_shape.height() * physical_shape.width()
Physical data size 2 should be same as volume indicated by physical shape (32, 1)
```

- After the host reshape fix, one request completed but `64 x 128` failed:
  `32` successful, `32` failed, total generated tokens `32`, and the engine
  died with:

```text
RuntimeError: TT sampled-token tensor is too small for rank 0:
shape=(1,), start=0, sz=32
```

- Temporary workaround: disabling traced device-feedback reuse for single-device
  lanes made N150 and N300 complete smoke and full benchmark runs with zero
  failures and healthy servers.
- Root-cause fix: preserving device feedback with sampler-output-to-decode-input
  reshape/copy also passed N150 `64 x 128`: `64/64` completed, `0` failed,
  `669.93 tok/s`, mean TTFT `1425.56 ms`, mean TPOT `36.89 ms`, P99 TPOT
  `46.86 ms`, post-run `/health=200`.
- The same root-cause fix passed N300 DP2 `64 x 128`: `64/64` completed,
  `0` failed, `597.22 tok/s`, mean TTFT `3083.35 ms`, mean TPOT `23.04 ms`,
  P99 TPOT `24.39 ms`, post-run `/health=200`.
- Output-quality investigation localized the semantic mismatch to prefill
  on-device sampling:
  - Host sampling and `sample_on_device_mode=decode_only` matched
    token-for-token for the three prompt oracle (`rain_raw`, `math_raw`,
    `chatish`).
  - `sample_on_device_mode=all` diverged at completion token `0` for all three
    prompts.
  - The initial `all` output sampled the wrong row from the 32-row prefill
    logits tile (`Question ...` style completions). Single-user prefill now
    selects the real last-token row before device sampling; traced prefill uses
    the same pre-LM-head row-selection contract as batched prefill.
  - After that row-selection fix, outputs became coherent, but still do not
    match host greedy sampling. In the same `sample_on_device_mode=all` server,
    a `logprobs=0` request forced host sampling on N150 and returned
    ` When rain falls ...`, while device sampling returned
    `assistant\n\nRain can make roads ...`.
  - Root cause of the remaining mismatch: `Sampling1D` was receiving a
    `[1, 1, 1, V]` prefill logits tensor while the cached greedy-equivalent
    `k/p/temp` tensors were length 32. After padding the selected logits row to
    `[1, 1, 32, V]`, same-server device sampling and `logprobs=0` forced-host
    sampling matched for `rain_raw`, `math_raw`, and `chatish`.
  - Decode-side device sampling was not implicated by this oracle.

Validation procedure:

- `tt-smi -r` between platforms.
- Fresh server with `TT_LLAMA_TEXT_VER=common_llama3_8b`,
  `trace_mode=decode_only`, `trace_region_size=85000000`, and
  `sample_on_device_mode=all`.
- N150 used `max_model_len=32768`, `max_num_seqs=32`.
- N300 used DP2 with `max_model_len=32768`, `max_num_seqs=16`.
- One-request completion smoke.
- `64 x 128` smoke.
- Full `320 x 256` benchmark.
- Post-run `/health` stayed `HTTP 200` after the passing smoke/full runs.

Smoke results:

| Device | Run | Completed | Failed | Duration s | Output tok/s | Mean TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| N150 | TTTv2 `sample_on_device_mode=all`, `64 x 128` | 64 | 0 | 12.21 | 670.99 | 1421.19 | 36.85 | 46.81 |
| N150 | TTTv2 `sample_on_device_mode=all`, prefill pad32 fix, `64 x 128` | 64 | 0 | 12.23 | 669.73 | 1429.89 | 36.87 | 46.83 |
| N300 | TTTv2 DP2 `sample_on_device_mode=all`, `64 x 128` | 64 | 0 | 13.73 | 596.67 | 3086.70 | 23.05 | 24.41 |

Full results:

| Device | Implementation | Shape | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Peak output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N150 | TTTv2 host sampling | DP1, `32768/32`, `trace_region_size=85000000` | 320 | 0 | 121.89 | 2.63 | 672.10 | 674.73 | 800.00 | 1447.11 | 1489.44 | 1502.10 | 42.12 | 41.95 | 47.29 |
| N150 | TTTv2 `sample_on_device_mode=all` | DP1, `32768/32`, `trace_region_size=85000000` | 320 | 0 | 109.71 | 2.92 | 746.68 | 749.59 | 896.00 | 1403.61 | 1441.00 | 1494.69 | 37.51 | 37.30 | 42.53 |
| N150 | TTTv2 `sample_on_device_mode=all`, prefill pad32 refresh | DP1, `32768/32`, `trace_region_size=85000000` | 320 | 0 | 109.68 | 2.92 | 746.90 | 749.81 | n/a | 1408.17 | 1443.10 | 1485.68 | 37.48 | 37.32 | 42.45 |
| N300 | TTTv2 host sampling | DP2, `32768/16`, `trace_region_size=85000000` | 320 | 0 | 137.46 | 2.33 | 595.96 | 598.29 | 704.00 | 1612.03 | 1648.01 | 2093.00 | 47.58 | 47.33 | 52.74 |
| N300 | TTTv2 `sample_on_device_mode=all` | DP2, `32768/16`, `trace_region_size=85000000` | 320 | 0 | 128.46 | 2.49 | 637.71 | 640.20 | 704.00 | 6579.44 | 6875.86 | 6986.09 | 23.31 | 23.24 | 23.61 |
| N300 | TTTv2 `sample_on_device_mode=all`, DP sampling guard removed | DP2, `32768/16`, `trace_region_size=85000000` | 320 | 0 | 134.61 | 2.38 | 608.57 | 610.95 | 832.00 | 1688.10 | 1694.12 | 2062.45 | 42.43 | 42.00 | 47.54 |
| N300 | TTTv2 `sample_on_device_mode=all`, top-k/token-axis correctness fix | DP2, `32768/16`, `trace_region_size=85000000` | 320 | 0 | 303.42 | 1.05 | 269.98 | 271.04 | 320.00 | 138531.03 | 138609.56 | 274594.15 | 112.86 | 112.79 | 118.78 |
| N300 | TTTv2 `sample_on_device_mode=all`, prefill top-k / decode argmax split | DP2, `32768/16`, `trace_region_size=85000000` | 320 | 0 | 123.27 | 2.60 | 664.56 | 667.16 | 800.00 | 57198.48 | 57266.34 | 112397.37 | 42.30 | 42.24 | 42.71 |

Performance notes:

- N150 `sample_on_device_mode=all` improves output throughput over host
  sampling by `11.1%` (`746.68 / 672.10`) and improves mean TPOT from
  `42.12 ms` to `37.51 ms`.
- The earlier N300 `sample_on_device_mode=all` diagnostic runs improved output
  throughput over host sampling, but some were invalid-token runs and the first
  correctness-clean top-k/token-axis run forced top-k for decode too.
- Profile comparison isolated the N300 throughput gap to decode-side top-k
  sampling, not DP wrapper overhead:
  - TTTv2 forced top-k decode, `64 x 128`: `251.48 tok/s`, mean TPOT
    `114.71 ms`; `tt_async_finalize_decode.event_sync_ms` mean `104.252 ms`.
  - TTTv2 host sampling, `64 x 128`: `544.26 tok/s`, mean TPOT `46.95 ms`;
    `event_sync_ms` mean `33.085 ms`.
  - TTTv1 host sampling, `64 x 128`: `320.74 tok/s`, mean TPOT `62.99 ms`;
    `event_sync_ms` mean `33.472 ms`.
  - TTTv2 split prefill-top-k/decode-argmax smoke, `64 x 128`: `567.84 tok/s`,
    mean TPOT `43.19 ms`; `event_sync_ms` mean `34.142 ms`.
- The split N300 run is now throughput-parity: output throughput is `664.56
  tok/s`, `11.5%` higher than TTTv2 host sampling (`595.96 tok/s`) and `55.2%`
  higher than the documented TTTv1 N300 baseline (`428.27 tok/s`). Mean TPOT is
  `42.30 ms`, better than TTTv2 host sampling (`47.58 ms`) and TTTv1
  (`57.89 ms`). TTFT is high because `320` requests queue behind DP2
  `max_num_seqs=16`, but steady-state TPOT and throughput are parity-positive.

Correctness caveat:

- Runtime stability and throughput now pass for `sample_on_device_mode=all` on
  N150 and N300.
- Decode-only device sampling matches host greedy output in the N150 oracle.
- Full `all` mode now passes the N150 three-prompt first-token quality oracle:
  device sampling and same-server `logprobs=0` forced-host sampling produced
  identical 32-token completions for `rain_raw`, `math_raw`, and `chatish`.
- N300 DP2 `all` mode is correctness-clean after the split-path fix:
  the clean full run completed `320/320`, generated all `81920` requested output
  tokens, and `server.log` had no invalid-token, detokenizer, engine-death, or
  sampled-token-size errors.
- With force-argmax disabled, N300 startup first exposed the lane-local top-k
  shape contract:

```text
RuntimeError: TT_FATAL ... sampling_device_operation.cpp:157:
k.logical_shape() == Shape({num_users})
k must have shape [16] (one per user)!
```

- After `k/p/temp` were built for the lane-local user count, the next failure
  was the stale max-batch index-offset tensor:

```text
RuntimeError: TT_THROW ... binary_ng_device_operation.cpp:225
info: Invalid subtile broadcast type
...
sampling_1d.py ... topk_global_indices = ttnn.add(self._index_offsets, topk_indices_int32, ...)
```

- After slicing the top-k index buffers, the reduced N300 repro completed but
  decode host processing only returned one token per lane:

```text
RuntimeError: TT sampled-token tensor is too small for rank 0:
shape=(2,), start=0, sz=16
```

- After sampled-token axis handling was fixed, the reduced N300 `32 x 64` repro
  completed `32/32`, generated the full `2048` tokens, and had no invalid-token
  or engine errors in `server.log`.

### TTTv2 full N150/N300/T3K `sample_on_device_mode=all` matrix

Full matrix run on 2026-07-10 after the N300 split prefill-top-k/decode-argmax
fix.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/tttv2_full_matrix_20260710T014925Z/`.

Procedure:

- Ran the platforms sequentially because TT hardware is not shareable.
- Reset with `tt-smi -r` before each platform and again after the matrix.
- Used `TT_LLAMA_TEXT_VER=common_llama3_8b`, `trace_mode=decode_only`,
  `trace_region_size=85000000`, and `sample_on_device_mode=all`.
- Client workload was the full parity workload: random input length `2`, random
  output length `256`, `320` prompts, `max-concurrency=320`, temperature `0`,
  and `--ignore-eos`.
- Searched server and client logs for `ERROR`, `Exception`, `Traceback`,
  `out of range`, `invalid`, `EngineDead`, sampled-token-size errors,
  `SIGKILL`, `OOM`, `Killed`, and `OutOfMemory`; there were no matches.
- Each server stayed healthy after its benchmark before shutdown.
- The throughput benchmark used random one-token prompts; those saved
  `generated_texts` are not a coherent-output quality oracle.

Artifacts:

- N150:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_full_matrix_20260710T014925Z/n150_sample_all/`.
- N300:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_full_matrix_20260710T014925Z/n300_sample_all/`.
- T3K:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_full_matrix_20260710T014925Z/t3k_sample_all/`.
- N150 deterministic quality smoke:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_full_matrix_20260710T014925Z/n150_quality_smoke/`.
- N300 deterministic quality smoke:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_full_matrix_20260710T014925Z/n300_quality_smoke/`.
- T3K deterministic quality smoke:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_full_matrix_20260710T014925Z/t3k_quality_smoke/`.

Results:

| Device | Shape | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Peak output tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N150 | DP1, `max_model_len=32768`, `max_num_seqs=32` | 320 | 0 | 109.74 | 2.92 | 746.48 | 749.40 | 896.00 | 51147.23 | 51174.66 | 100186.31 | 37.48 | 37.33 | 38.98 |
| N300 | DP2, `max_model_len=32768`, `max_num_seqs=16` | 320 | 0 | 123.72 | 2.59 | 662.13 | 664.72 | 800.00 | 57602.88 | 57571.82 | 112953.06 | 42.37 | 42.16 | 47.93 |
| T3K | DP4, `max_model_len=131072`, `max_num_seqs=8`, `fabric_config=FABRIC_1D` | 320 | 0 | 192.00 | 1.67 | 426.66 | 428.32 | 544.00 | 77847.48 | 76785.66 | 169112.59 | 61.34 | 61.73 | 65.51 |

Quality smoke:

- Prompt: `Explain why rain can make roads slippery in three sentences.`
- Request: `/v1/completions`, `max_tokens=80`, `temperature=0`,
  `ignore_eos=true`, same server configs as the matrix.
- N150 and N300 produced identical coherent completions beginning:
  `When rain falls on a road, it creates a layer of water on the surface...`
- T3K produced a coherent on-topic completion beginning:
  `When rain falls on a road, it creates a layer of water that can make the surface slippery...`
- All three quality responses reached the `max_tokens=80` length cap; no text
  corruption or invalid-token symptoms were observed in this deterministic
  smoke.

Conclusion:

- The full N150/N300/T3K `sample_on_device_mode=all` matrix passes
  functionally: all three platforms completed `320/320`, generated the full
  `81920` output tokens, and stayed healthy after the run.
- The matrix is throughput-parity-positive against the documented TTTv1
  baselines: N150 `746.48 tok/s` vs TTTv1 `620.92 tok/s`; N300 `662.13 tok/s`
  vs TTTv1 `428.27 tok/s`; T3K `426.66 tok/s` vs TTTv1 `338.47 tok/s`.
- The N300 split-path change did not regress N150 or T3K in this full matrix.

TTFT caveat:

- The full matrix used `max-concurrency=320`, while the TTTv1 baseline in
  section 10 used `max-concurrency=32`.
- `max-concurrency` is the benchmark client's in-flight HTTP request cap. It is
  not the same as the server's active sequence capacity, `max_num_seqs`.
- With request rate `inf` and `max-concurrency=320`, the client is allowed to
  keep all `320` requests in flight at once. The server does not wait for all
  requests to arrive before serving: it starts the first schedulable wave as
  soon as it can.
- The server only actively admits up to its active sequence capacity (`32` on
  N150, `16` on N300, `8` on T3K in these configs). Later requests sit behind
  earlier waves until capacity opens.
- Benchmark TTFT is measured per request from that request's submission time to
  its first generated token, so later-wave requests include backlog wait in
  TTFT. This explains the very large mean/P99 TTFT in the full matrix despite
  good decode throughput and TPOT.

### T3K DP8 throughput experiment

Follow-up run on 2026-07-10 to test whether T3K total throughput was limited by
the DP4 server shape rather than available hardware.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/tttv2_t3k_dp8_experiment_20260710T152632Z/`.

Server shape:

- T3K, `--data-parallel-size 8`, `--max-model-len 32768`,
  `--max_num_seqs 16`, async scheduling.
- TT config: `trace_mode=decode_only`, `trace_region_size=85000000`,
  `sample_on_device_mode=all`, `fabric_config=FABRIC_1D`.
- This gives a nominal global active sequence capacity of `8 * 16 = 128`, vs
  `4 * 8 = 32` for the T3K DP4 full-matrix run.

Procedure:

- Started from `tt-smi -r`.
- First DP8 startup was cold for additional single-device tensor-cache entries;
  DP0 generated cache entries through `device_8` before prefill trace warmup and
  API health.
- Ran a quick `128 x 128` smoke at `max-concurrency=128`.
- Ran a deterministic quality smoke with the rain/roads prompt.
- Ran the full comparison workload: random input length `2`, random output
  length `256`, `320` prompts, `max-concurrency=320`, temperature `0`, and
  `--ignore-eos`.
- Searched server and client logs for error, exception, invalid-token,
  `EngineDead`, `SIGKILL`, `OOM`, `Killed`, and `OutOfMemory` signatures; there
  were no matches.
- Stopped the server and reset T3K after the run.

Results:

| Run | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DP8 quick, `128 x 128`, `max-concurrency=128` | 128 | 0 | 16.42 | 7.79 | 997.51 | 1005.30 | 6318.28 | 6318.52 | 6330.80 | 79.41 | 79.40 | 79.45 |
| DP8 full, `320 x 256`, `max-concurrency=320` | 320 | 0 | 80.71 | 3.96 | 1015.02 | 1018.99 | 29203.18 | 35476.63 | 60239.01 | 83.00 | 80.39 | 93.54 |

Quality smoke:

- Prompt: `Explain why rain can make roads slippery in three sentences.`
- Output was coherent and on-topic, beginning:
  `When rain falls on a road, it creates a layer of water on the surface...`
- The request used `ignore_eos=true`, so the completion continued into an extra
  `Answer:` fragment after the three-sentence answer. That matches the request
  shape and was not token corruption.

Interpretation:

- DP8 removes the surprising total-throughput inversion from the full matrix:
  T3K DP8 `1015.02` output tok/s is higher than N300 DP2 `662.13` output tok/s
  and T3K DP4 `426.66` output tok/s for the same `320 x 256`
  `max-concurrency=320` client workload.
- The DP8 comparison is not identical to the T3K DP4 full-matrix shape because
  DP8 used `max_model_len=32768`, while that DP4 row used `131072`.
- Per-token latency is still not better: DP8 mean TPOT was `83.00 ms`, slower
  than the T3K DP4 full-matrix `61.34 ms` and N300 DP2 `42.37 ms`. The DP8
  throughput gain comes from admitting more active sequences globally, not from
  faster per-rank decode.
- DP4's lower total tok/s was therefore primarily a server-shape/concurrency
  issue: it used only `32` global active slots (`4 * 8`), the same global active
  slot count as N300 DP2 (`2 * 16`), while also using a larger context budget and
  slower per-token execution.

### T3K DP4 vs DP8 apples-to-apples throughput

Follow-up run on 2026-07-10 to isolate DP rank count by keeping the T3K server
shape and client workload fixed.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/tttv2_t3k_dp4_dp8_apples_20260710T155924Z/`.

Matched server/client shape:

- T3K, `--max-model-len 32768`, `--max_num_seqs 16`, async scheduling.
- TT config: `trace_mode=decode_only`, `trace_region_size=85000000`,
  `sample_on_device_mode=all`, `fabric_config=FABRIC_1D`.
- Client workload: random input length `2`, random output length `256`, `320`
  prompts, `max-concurrency=320`, temperature `0`, and `--ignore-eos`.
- The only intentional server difference was `--data-parallel-size`: DP4 vs DP8.
- Reset with `tt-smi -r` before DP4, between DP4 and DP8, and after DP8.
- Searched server and client logs for error, exception, invalid-token,
  `EngineDead`, `SIGKILL`, `OOM`, `Killed`, and `OutOfMemory` signatures; there
  were no matches.

Results:

| Run | Global active slots | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DP4, `max_model_len=32768`, `max_num_seqs=16` | 64 | 320 | 0 | 100.21 | 3.19 | 817.51 | 820.71 | 38228.62 | 38320.62 | 73493.85 | 61.49 | 61.24 | 68.87 |
| DP8, `max_model_len=32768`, `max_num_seqs=16` | 128 | 320 | 0 | 79.28 | 4.04 | 1033.27 | 1037.31 | 28349.88 | 34239.02 | 58705.77 | 81.33 | 80.53 | 103.97 |

Interpretation:

- DP8 improved matched-workload total output throughput by `26.4%`
  (`1033.27 / 817.51 - 1`) by doubling global active slots from `64` to `128`.
- DP8 mean TPOT was `32.3%` slower than DP4 (`81.33 / 61.49 - 1`), so the
  throughput gain is from more concurrent active sequences, not faster
  per-sequence decode.
- DP4's faster TPOT means the T3K per-rank/per-step path still needs profiling.
  The next bottleneck question is why DP8's one-device-rank topology pays more
  per token than DP4 at the same per-rank `max_num_seqs=16`.

### N300 DP1 throughput experiment

Follow-up run on 2026-07-10 to compare the existing N300 DP2 matrix result with
a single N300 DP lane.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/tttv2_n300_dp1_20260710T162208Z/`.

Server/client shape:

- N300, `--data-parallel-size 1`, `--max-model-len 32768`,
  `--max_num_seqs 16`, async scheduling.
- TT config: `trace_mode=decode_only`, `trace_region_size=85000000`,
  `sample_on_device_mode=all`.
- Client workload: random input length `2`, random output length `256`, `320`
  prompts, `max-concurrency=320`, temperature `0`, and `--ignore-eos`.
- Reset with `tt-smi -r` before and after the run.
- Searched server and client logs for error, exception, invalid-token,
  `EngineDead`, `SIGKILL`, `OOM`, `Killed`, and `OutOfMemory` signatures; there
  were no matches.

Results:

| Run | Global active slots | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N300 DP1, `max_model_len=32768`, `max_num_seqs=16` | 16 | 320 | 0 | 130.96 | 2.44 | 625.53 | 627.97 | 62936.49 | 62679.41 | 124991.44 | 23.69 | 23.61 | 25.17 |
| N300 DP2 full-matrix row, `max_model_len=32768`, `max_num_seqs=16` | 32 | 320 | 0 | 123.72 | 2.59 | 662.13 | 664.72 | 57602.88 | 57571.82 | 112953.06 | 42.37 | 42.16 | 47.93 |

Quality smoke:

- Prompt: `Explain why rain can make roads slippery in three sentences.`
- Output was coherent and on-topic, beginning:
  `When rain falls on a road, it creates a layer of water that can make the surface slippery...`

Interpretation:

- N300 DP1 works functionally with `sample_on_device_mode=all` and the same
  `max_model_len=32768`, `max_num_seqs=16` shape.
- DP1 has much faster per-token latency than DP2 (`23.69 ms` vs `42.37 ms`
  mean TPOT), but lower total throughput (`625.53` vs `662.13` output tok/s)
  because DP1 has half the global active slots.
- The small total-throughput gap despite halving active slots suggests N300 DP2
  is dominated by per-rank/per-token overhead at this workload, not just by
  available concurrency.

### Decode-only without prefill trace warmup experiment

Follow-up run on 2026-07-10 to test whether
`trace_mode=decode_only` can skip TTTv2 Llama prefill trace warmup even when the
model advertises `requires_prefill_trace_warmup=True`.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/decode_only_no_prefill_trace_smoke_20260710T172155Z/`.

Temporary code change tested:

- Changed `TTModelRunner.warmup_model()` so `trace_prefill_mode` was true only
  for `trace_mode=all`.
- This meant `trace_mode=decode_only` captured decode traces but did not capture
  prefill traces.
- The temporary change was reverted after the smoke tests because deterministic
  output quality failed on both N300 and T3K.

N300 DP2 smoke:

- Server: N300, `--data-parallel-size 2`, `--max-model-len 32768`,
  `--max_num_seqs 16`, `trace_mode=decode_only`,
  `sample_on_device_mode=all`.
- Startup reached health. Logs showed decode trace capture and no
  `Captured prefill trace` entries.
- Random smoke `64 x 128`, `max-concurrency=64`: `64/64` completed, `0`
  failed, `568.15` output tok/s, mean TPOT `41.52 ms`.
- No error, exception, invalid-token, `EngineDead`, `SIGKILL`, `OOM`, `Killed`,
  or `OutOfMemory` signatures were found.
- Deterministic quality smoke failed: the rain/roads prompt produced repeated
  `O` and parenthesis tokens instead of coherent text.

T3K DP4 smoke:

- Server: T3K, `--data-parallel-size 4`, `--max-model-len 32768`,
  `--max_num_seqs 16`, `trace_mode=decode_only`,
  `sample_on_device_mode=all`, `fabric_config=FABRIC_1D`.
- Startup reached health. Logs showed decode trace capture and no
  `Captured prefill trace` entries.
- Random smoke `64 x 128`, `max-concurrency=64`: `64/64` completed, `0`
  failed, `720.18` output tok/s, mean TPOT `60.84 ms`.
- No error, exception, invalid-token, `EngineDead`, `SIGKILL`, `OOM`, `Killed`,
  or `OutOfMemory` signatures were found.
- Deterministic quality smoke failed: the rain/roads prompt produced unrelated
  procedural text beginning `1. The first step is to make sure...`.

Interpretation:

- Removing prefill trace warmup does not break startup or random benchmark
  completion, but it breaks deterministic output quality.
- The `requires_prefill_trace_warmup` path is therefore not just stale
  defensive trace-memory setup; for TTTv2 Llama with on-device sampling it is
  required for correct text under the current decode-only runtime.
- The bad behavior is silent: no invalid-token or runtime error signatures were
  emitted. Deterministic quality smoke is required for this class of change.
- Root cause found: vLLM passed `enable_trace=False` for runtime prefill when
  `trace_mode=decode_only`, but the TTTv2 generator dropped that argument and
  always called the traced executor's `prefill_forward()`. That meant
  decode-only serving could still lazily capture prefill traces during live
  inference, after decode traces were resident, instead of using the intended
  eager prefill path.

### Trace-all control validation for TTTv2 Llama

Follow-up run on 2026-07-10 to validate the known-good control path while
investigating why `trace_mode=decode_only` corrupts deterministic text quality.
The control path is `trace_mode=all` with `sample_on_device_mode=all`.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/trace_all_failfast_smoke_20260710T174208Z/`.

Code status:

- The attempted fail-fast workaround was removed so `trace_mode=decode_only`
  remains runnable as the failing oracle.
- `TTModelRunner` remains model-agnostic: `trace_prefill_mode` is true only for
  `trace_mode=all`, and `trace_decode_mode` is true for `trace_mode=all` and
  `trace_mode=decode_only`.

N300 DP2 smoke:

- Server: N300, `--data-parallel-size 2`, `--max-model-len 32768`,
  `--max_num_seqs 16`, `trace_mode=all`, `sample_on_device_mode=all`.
- Startup reached health. Logs showed both `Captured prefill trace` and
  `Captured decode trace` entries.
- Random smoke `64 x 128`, `max-concurrency=64`: `64/64` completed, `0`
  failed, `565.09` output tok/s, `569.50` total tok/s, mean TTFT
  `5407.92 ms`, mean TPOT `43.47 ms`, p99 TPOT `47.84 ms`.
- Deterministic quality smoke was coherent and on-topic for the rain/roads
  prompt.
- No error, exception, invalid-token, `EngineDead`, `SIGKILL`, `OOM`, `Killed`,
  or `OutOfMemory` signatures were found.

T3K DP4 smoke:

- Server: T3K, `--data-parallel-size 4`, `--max-model-len 32768`,
  `--max_num_seqs 16`, `trace_mode=all`, `sample_on_device_mode=all`,
  `fabric_config=FABRIC_1D`.
- Startup reached health. Logs showed both `Captured prefill trace` and
  `Captured decode trace` entries.
- Random smoke `64 x 128`, `max-concurrency=64`: `64/64` completed, `0`
  failed, `827.08` output tok/s, `833.55` total tok/s, mean TTFT
  `2290.17 ms`, mean TPOT `59.87 ms`, p99 TPOT `65.52 ms`.
- Deterministic quality smoke was coherent and on-topic for the rain/roads
  prompt.
- No error, exception, invalid-token, `EngineDead`, `SIGKILL`, `OOM`, `Killed`,
  or `OutOfMemory` signatures were found.

Interpretation:

- The prior decode-only experiment proved that skipping prefill trace warmup can
  silently corrupt text quality even when benchmark completion succeeds.
- The later root-cause fix below makes `enable_trace=False` effective for TTTv2
  runtime prefill, so decode-only no longer relies on traced prefill warmup for
  quality.
- N300 and T3K both pass random throughput smoke plus deterministic text-quality
  smoke under the `trace_mode=all` control configuration.

### Decode-only root-cause fix

Follow-up run on 2026-07-10 after removing the fail-fast workaround and fixing
the TTTv2 executor contract so `enable_trace=False` actually disables traced
prefill at runtime.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/decode_only_enable_trace_fix_20260710T222419Z/`.

Patch:

- `Llama3Generator.prefill_forward()` and `_DPLlama3Generator.prefill_forward()`
  no longer drop `enable_trace`; they pass it through to the executor.
- `TracedLLMExecutor.prefill_forward(..., enable_trace=False)` delegates to the
  eager executor instead of capturing/replaying prefill traces.
- `TracedLLMExecutor.decode_forward(..., enable_trace=False)` has the same eager
  fallback for contract symmetry.
- The Llama3 traced executor wrapper forwards `enable_trace` into the shared
  traced engine.

N300 DP2 decode-only validation:

- Server: N300, `--data-parallel-size 2`, `--max-model-len 32768`,
  `--max_num_seqs 16`, `trace_mode=decode_only`,
  `sample_on_device_mode=all`.
- Startup reached health. Logs showed decode trace capture and no
  `Captured prefill trace` entries.
- Deterministic rain/roads quality smoke was coherent and on-topic.
- Random smoke `64 x 128`, `max-concurrency=64`: `64/64` completed, `0`
  failed, `588.39` output tok/s, `592.98` total tok/s, mean TTFT
  `5289.42 ms`, mean TPOT `41.32 ms`, p99 TPOT `45.56 ms`.

T3K DP4 decode-only validation:

- Server: T3K, `--data-parallel-size 4`, `--max-model-len 32768`,
  `--max_num_seqs 16`, `trace_mode=decode_only`,
  `sample_on_device_mode=all`, `fabric_config=FABRIC_1D`.
- Startup reached health. Logs showed decode trace capture and no
  `Captured prefill trace` entries.
- Deterministic rain/roads quality smoke was coherent and on-topic.
- Random smoke `64 x 128`, `max-concurrency=64`: `64/64` completed, `0`
  failed, `825.19` output tok/s, `831.64` total tok/s, mean TTFT
  `2208.74 ms`, mean TPOT `60.68 ms`, p99 TPOT `60.76 ms`.

Interpretation:

- The issue was not that TTTv2 fundamentally requires prefill trace warmup for
  decode-only quality. The bug was that runtime prefill ignored the trace-mode
  contract and still used the traced prefill path.
- With the flag respected, `trace_mode=decode_only` keeps prefill eager, captures
  decode traces only, and matches the `trace_mode=all` quality control on both
  N300 and T3K.

### Larger decode-only benchmark after root-cause fix

Follow-up run on 2026-07-11 to exercise the fixed `trace_mode=decode_only`
path with the larger apples-to-apples workload shape: random input length `2`,
random output length `256`, `320` prompts, request rate `inf`,
`max-concurrency=32`, temperature `0`, and `--ignore-eos`.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/decode_only_larger_20260711T011830Z/`.

Procedure:

- Ran N300 first, then T3K, so TT hardware was not shared between servers.
- Reset with `tt-smi -r` before each platform and after the final T3K run. The
  first pre-N300 reset hit an ARC telemetry timeout once, then recovered on the
  first retry.
- Used `TT_LLAMA_TEXT_VER=common_llama3_8b`, `trace_mode=decode_only`,
  `trace_region_size=85000000`, and `sample_on_device_mode=all`.
- N300 used DP2 with `max_model_len=32768`, `max_num_seqs=16`.
- T3K used DP4 with `max_model_len=32768`, `max_num_seqs=16`, and
  `fabric_config=FABRIC_1D`.
- Searched logs for `ERROR`, `Exception`, `Traceback`, `EngineDead`, `SIGKILL`,
  `OOM`, `Killed`, and `OutOfMemory`; there were no matches beyond normal
  decode-trace capture lines.
- Logs showed decode trace capture only: N300 captured `4` decode traces and
  T3K captured `8` decode traces. Neither run logged `Captured prefill trace`.

Results:

| Device | DP | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N300 | 2 | 320 | 0 | 127.44 | 2.51 | 642.79 | 645.30 | 1901.90 | 2006.16 | 2344.78 | 42.51 | 42.12 | 48.54 |
| T3K | 4 | 320 | 0 | 167.95 | 1.91 | 487.75 | 489.66 | 1304.05 | 1387.39 | 1548.55 | 60.74 | 60.38 | 64.52 |

Interpretation:

- The larger decode-only workload completed cleanly on both N300 and T3K with
  no prefill trace captures, so the `enable_trace=False` fix still holds beyond
  the small quality/throughput smokes.
- T3K's lower output tok/s in this exact run should not be read as hardware
  scaling: the client concurrency was fixed globally at `32`, while the T3K
  server used DP4 and N300 used DP2. This shape is useful for TTFT comparison
  and decode-only stability, but not for saturated T3K throughput.

### TTTv2 apples-to-apples TTFT matrix at `max-concurrency=32`

Follow-up matrix run on 2026-07-10 to compare TTTv2 TTFT against the TTTv1
section 10 baseline with the same client workload shape: random input length
`2`, random output length `256`, `320` prompts, request rate `inf`,
`max-concurrency=32`, temperature `0`, and `--ignore-eos`.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/tttv2_ttft_apples_20260710T042559Z/`.

Procedure:

- Ran N150, N300, and T3K sequentially.
- Reset with `tt-smi -r` before each platform and after the final T3K run.
- Used the same TTTv2 server configs as the full matrix:
  `TT_LLAMA_TEXT_VER=common_llama3_8b`, `trace_mode=decode_only`,
  `trace_region_size=85000000`, and `sample_on_device_mode=all`.
- N150 used DP1 with `max_model_len=32768`, `max_num_seqs=32`.
- N300 used DP2 with `max_model_len=32768`, `max_num_seqs=16`.
- T3K used DP4 with `max_model_len=131072`, `max_num_seqs=8`, and
  `fabric_config=FABRIC_1D`.
- Searched logs for `ERROR`, `Exception`, `Traceback`, `out of range`,
  `invalid`, `EngineDead`, sampled-token-size errors, `SIGKILL`, `OOM`,
  `Killed`, `OutOfMemory`, and leftover profile markers `TT_DP_PROFILE` /
  `PROFILE_JSON`; there were no matches.
- Each server stayed healthy after its benchmark before shutdown.

Artifacts:

- N150:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_ttft_apples_20260710T042559Z/n150/`.
- N300:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_ttft_apples_20260710T042559Z/n300/`.
- T3K:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_ttft_apples_20260710T042559Z/t3k/`.

Results:

| Device | TTTv1 mean TTFT ms | TTTv2 mean TTFT ms | TTFT delta | TTTv1 output tok/s | TTTv2 output tok/s | TTTv2 mean TPOT ms | TTTv2 P99 TPOT ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| N150 | 2115.17 | 1540.16 | `27.2%` faster | 620.92 | 737.30 | 37.52 | 42.75 |
| N300 | 3196.13 | 1911.24 | `40.2%` faster | 428.27 | 598.39 | 42.49 | 48.07 |
| T3K | 2364.95 | 1328.19 | `43.8%` faster | 338.47 | 451.14 | 61.62 | 65.17 |

Conclusion:

- Under the same client concurrency as the TTTv1 baseline, TTTv2 is
  TTFT-parity-positive on all three platforms.
- The earlier full-matrix TTFT numbers were dominated by burst queueing from
  `max-concurrency=320`, not by first-token compute regression.
- TTTv2 also remains output-throughput-positive in this apples-to-apples
  concurrency-32 comparison.

### TTTv2 T3K post-instrumentation-cleanup smoke

Quick T3K smoke run on 2026-07-10 after removing the temporary DP/profile
instrumentation from the vLLM async decode wrapper and the TTTv2 DP wrapper.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/tttv2_t3k_post_instrument_cleanup_smoke_20260710T040454Z/`.

Procedure:

- Reset T3K with `tt-smi -r` before the server run and again after shutdown.
- Used DP4 with `max_model_len=131072`, `max_num_seqs=8`,
  `fabric_config=FABRIC_1D`, `trace_mode=decode_only`,
  `trace_region_size=85000000`, and `sample_on_device_mode=all`.
- Ran a short throughput smoke with random input length `2`, random output
  length `128`, `64` prompts, `max-concurrency=64`, temperature `0`, and
  `--ignore-eos`.
- Ran a deterministic text smoke with prompt
  `Explain why rain can make roads slippery in three sentences.`, `max_tokens=80`,
  `temperature=0`, and `ignore_eos=true`.
- Searched logs for the usual failure signatures and for leftover profile
  markers `TT_DP_PROFILE` and `PROFILE_JSON`; there were no matches.
- Server health was OK before shutdown.

Results:

| Device | Shape | Completed | Failed | Output tok/s | Mean TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---|---:|---:|---:|---:|---:|---:|
| T3K | DP4, `max_model_len=131072`, `max_num_seqs=8`, `fabric_config=FABRIC_1D` | 64 | 0 | 456.49 | 5781.70 | 61.03 | 64.07 |

Quality smoke:

- T3K produced a coherent on-topic completion beginning:
  `When rain falls on a road, it creates a layer of water that can make the surface slippery...`
- No text corruption or invalid-token symptoms were observed.

Conclusion: the cleanup did not regress the quick T3K DP4 functional path.

### TTTv2 N300 post-instrumentation-cleanup smoke

Quick N300 smoke run on 2026-07-10 after the same instrumentation cleanup.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/tttv2_n300_post_cleanup_smoke_20260710T041844Z/`.

Procedure:

- Reset N300 with `tt-smi -r` before the server run and again after shutdown.
- Used DP2 with `max_model_len=32768`, `max_num_seqs=16`,
  `trace_mode=decode_only`, `trace_region_size=85000000`, and
  `sample_on_device_mode=all`.
- Ran a short throughput smoke with random input length `2`, random output
  length `128`, `64` prompts, `max-concurrency=64`, temperature `0`, and
  `--ignore-eos`.
- Ran a deterministic text smoke with prompt
  `Explain why rain can make roads slippery in three sentences.`, `max_tokens=80`,
  `temperature=0`, and `ignore_eos=true`.
- Searched logs for the usual failure signatures and for leftover profile
  markers `TT_DP_PROFILE` and `PROFILE_JSON`; there were no matches.
- Server health was OK before shutdown.

Results:

| Device | Shape | Completed | Failed | Output tok/s | Mean TTFT ms | Mean TPOT ms | P99 TPOT ms |
|---|---|---:|---:|---:|---:|---:|---:|
| N300 | DP2, `max_model_len=32768`, `max_num_seqs=16` | 64 | 0 | 583.47 | 5357.93 | 41.78 | 46.14 |

Quality smoke:

- N300 produced a coherent on-topic completion beginning:
  `When rain falls on a road, it creates a layer of water on the surface...`
- The response reached the `max_tokens=80` length cap; no text corruption or
  invalid-token symptoms were observed.

Conclusion: the cleanup did not regress the quick N300 DP2 functional path.

### TTTv2 full trace-mode matrix with 128-token input

Full matrix run on 2026-07-11 after the `trace_mode=decode_only` root-cause fix.
This reruns the parity-style benchmark across all requested platform/DP
combinations and compares `trace_mode=decode_only` against `trace_mode=all`.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/tttv2_full_trace_matrix_128in_20260711T013904/`.

Client workload:

- `vllm bench serve`, OpenAI completions endpoint.
- Model: `meta-llama/Llama-3.1-8B-Instruct`.
- Random input length `128`, random output length `256`.
- `320` prompts, request rate `inf`, `max-concurrency=320`.
- `temperature=0`, `--ignore-eos`, saved detailed results.

Server/config shape:

- All runs used `TT_LLAMA_TEXT_VER=common_llama3_8b`,
  `sample_on_device_mode=all`, `trace_region_size=85000000`, and async
  scheduling.
- N150 DP1: `max_model_len=32768`, `max_num_seqs=32`.
- N300 DP1/DP2: `max_model_len=32768`, `max_num_seqs=16`.
- T3K DP1/DP4/DP8: `max_model_len=32768`, `max_num_seqs=16`,
  `fabric_config=FABRIC_1D`.
- The T3K DP4 row intentionally uses the matched DP4/DP8 comparison shape
  (`max_num_seqs=16`) rather than the older long-context parity shape
  (`max_model_len=131072`, `max_num_seqs=8`).

Procedure notes:

- Hardware was used sequentially. Each case reset TT hardware before startup
  and after shutdown.
- The first N150 decode-only row completed before the runner cleanup was
  hardened. A stale server child was detected before the N150 trace-all client
  could produce a valid result; the runner was stopped, all server/client
  processes were cleaned up, TT hardware was reset, and the matrix was resumed.
  The completed N150 decode-only result was kept; N150 trace-all and all later
  rows were run with process-group cleanup.
- T3K DP8 decode-only had a long cold startup because DP0 generated additional
  single-device tensor-cache entries through `device_8`. This was startup-only
  cache generation; the benchmark result below is the client run after API
  health.
- A narrow failure scan over server/client logs found no `ERROR`, `CRITICAL`,
  `Traceback`, `RuntimeError`, `EngineDead`, `SIGKILL`, `OOM`, `OutOfMemory`,
  `Killed`, or nonzero failed-request signatures.
- All client exits were `0`. The final process check found no lingering
  server/benchmark process, and the final T3K reset completed.

Results:

| Device | DP | Trace mode | Client max concurrency | Per-rank `max_num_seqs` | Global active slots | Completed | Failed | Duration s | Req/s | Output tok/s | Total tok/s | Mean TTFT ms | Median TTFT ms | P99 TTFT ms | Mean TPOT ms | Median TPOT ms | P99 TPOT ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N150 | 1 | `decode_only` | 320 | 32 | 32 | 320 | 0 | 115.77 | 2.76 | 707.64 | 1058.69 | 53873.60 | 53171.01 | 105814.44 | 39.65 | 39.55 | 44.38 |
| N150 | 1 | `all` | 320 | 32 | 32 | 320 | 0 | 115.79 | 2.76 | 707.49 | 1058.47 | 53878.52 | 53144.89 | 105823.92 | 39.71 | 39.62 | 44.37 |
| N300 | 1 | `decode_only` | 320 | 16 | 16 | 320 | 0 | 133.66 | 2.39 | 612.88 | 916.92 | 64354.26 | 64087.26 | 127588.39 | 24.13 | 24.04 | 25.65 |
| N300 | 1 | `all` | 320 | 16 | 16 | 320 | 0 | 133.52 | 2.40 | 613.53 | 917.89 | 64218.68 | 63941.38 | 127442.16 | 24.14 | 24.05 | 25.65 |
| N300 | 2 | `decode_only` | 320 | 16 | 32 | 320 | 0 | 124.66 | 2.57 | 657.17 | 983.19 | 57935.35 | 57976.74 | 113674.11 | 42.72 | 42.70 | 47.19 |
| N300 | 2 | `all` | 320 | 16 | 32 | 320 | 0 | 143.25 | 2.23 | 571.85 | 855.55 | 58135.90 | 57541.59 | 123308.12 | 42.28 | 42.59 | 43.06 |
| T3K | 1 | `decode_only` | 320 | 16 | 16 | 320 | 0 | 89.74 | 3.57 | 912.90 | 1365.78 | 43187.82 | 43188.02 | 85514.46 | 16.40 | 16.39 | 16.45 |
| T3K | 1 | `all` | 320 | 16 | 16 | 320 | 0 | 91.11 | 3.51 | 899.15 | 1345.22 | 43716.72 | 43510.77 | 86876.78 | 16.90 | 16.77 | 17.58 |
| T3K | 4 | `decode_only` | 320 | 16 | 64 | 320 | 0 | 100.54 | 3.18 | 814.80 | 1219.02 | 38299.82 | 38361.78 | 73665.41 | 61.44 | 61.22 | 69.01 |
| T3K | 4 | `all` | 320 | 16 | 64 | 320 | 0 | 100.69 | 3.18 | 813.62 | 1217.25 | 38346.00 | 38316.30 | 73687.62 | 61.63 | 61.45 | 68.74 |
| T3K | 8 | `decode_only` | 320 | 16 | 128 | 320 | 0 | 80.21 | 3.99 | 1021.38 | 1528.08 | 29208.23 | 35669.10 | 59683.08 | 83.24 | 81.15 | 104.35 |
| T3K | 8 | `all` | 320 | 16 | 128 | 320 | 0 | 78.17 | 4.09 | 1048.00 | 1567.90 | 27640.79 | 33590.86 | 57568.93 | 81.24 | 80.96 | 100.21 |

Trace capture counts:

| Device | DP | Trace mode | Prefill traces | Decode traces |
|---|---:|---|---:|---:|
| N150 | 1 | `decode_only` | 0 | 2 |
| N150 | 1 | `all` | 7 | 2 |
| N300 | 1 | `decode_only` | 0 | 2 |
| N300 | 1 | `all` | 9 | 2 |
| N300 | 2 | `decode_only` | 0 | 4 |
| N300 | 2 | `all` | 12 | 4 |
| T3K | 1 | `decode_only` | 0 | 2 |
| T3K | 1 | `all` | 9 | 2 |
| T3K | 4 | `decode_only` | 0 | 8 |
| T3K | 4 | `all` | 36 | 8 |
| T3K | 8 | `decode_only` | 0 | 16 |
| T3K | 8 | `all` | 48 | 16 |

Interpretation:

- Functional parity passes for every requested platform/DP/trace-mode
  combination: all 12 rows completed `320/320` requests with `0` failures.
- `decode_only` behaved as intended in this matrix: no prefill traces were
  captured, while decode traces scaled with DP size. `trace_mode=all` captured
  prefill traces plus the same decode-trace count.
- `trace_mode=all` is not generally slower. N150 DP1, N300 DP1, and T3K DP4 are
  effectively tied across trace modes. N300 DP2 trace-all lost throughput
  (`571.85` vs `657.17` output tok/s) without a TPOT regression, suggesting
  extra non-steady-state overhead. T3K DP8 trace-all was slightly faster in this
  run (`1048.00` vs `1021.38` output tok/s).
- T3K DP8 is the best total-throughput and TTFT row in the matrix, reaching
  `1048.00` output tok/s in trace-all and `1021.38` output tok/s in
  decode-only. Its mean TPOT is still slower than T3K DP4 and DP1, so the
  scaling comes from more global active slots, not faster per-sequence decode.
- T3K DP1 has the best per-token decode latency (`16.40` ms mean TPOT in
  decode-only), but fewer global active slots produce worse burst TTFT than DP4
  or DP8 under `max-concurrency=320`.
- With input length `128`, TTFT is substantially higher than the earlier
  input-length-`2` apples-to-apples runs because the burst workload queues many
  128-token prefills behind limited active slots. These TTFT numbers should be
  read as full burst queueing plus prefill behavior, not isolated first-token
  compute latency.

### TTTv2 global sampled-token contract matrix rerun

Follow-up run on 2026-07-11 after moving sampled-token shape normalization out
of vLLM and into the TTTv2 generator boundary.

Code boundary change:

- TTTv2 DP `process_decode_output_host(..., is_tokens=True)` now returns a
  global-slot sampled-token tensor with shape
  `tt_data_parallel * per_lane_max_batch_size`.
- TTTv2 host-processed sampled tokens are converted to signed `torch.int64`.
  The first full-matrix attempt after the shape-boundary change exposed that
  the model boundary returned `torch.uint32`; vLLM then failed when indexing
  sampled tokens with `_rows`:
  `RuntimeError: "index_cpu" not implemented for 'UInt32'`.
- vLLM no longer guesses between global and compact sampled-token shapes; it
  uses the strict global-slot contract:
  `_take(tt_out).reshape(sz).to(torch.int32)`.

Validation artifacts:

- Failed first full-matrix attempt before the dtype fix:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_full_trace_matrix_128in_global_tokens_20260711T145735Z/`.
- Focused N150 smoke after the dtype fix:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_global_token_dtype_n150_smoke_20260711T150131Z/`.
- Full successful matrix:
  `/localdev/gwang/vllm_duo/perf_results/tttv2_full_trace_matrix_128in_global_tokens_20260711T150310Z/`.

The successful full matrix reused the same workload as the prior 128-input
trace-mode matrix:

- Random input length `128`, random output length `256`.
- `320` prompts, request rate `inf`, `max-concurrency=320`.
- `temperature=0`, `--ignore-eos`.
- `sample_on_device_mode=all`, `trace_region_size=85000000`.
- N150 DP1, N300 DP1/DP2, and T3K DP1/DP4/DP8.
- Both `trace_mode=decode_only` and `trace_mode=all`.

Results compared with the previous matrix at
`/localdev/gwang/vllm_duo/perf_results/tttv2_full_trace_matrix_128in_20260711T013904/`:

| Device | DP | Trace | Completed | Failed | New output tok/s | Old output tok/s | Delta | New TTFT ms | Old TTFT ms | New TPOT ms | Old TPOT ms |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| N150 | 1 | `decode_only` | 320 | 0 | 704.21 | 707.64 | -0.5% | 54424.04 | 53873.60 | 39.66 | 39.65 |
| N150 | 1 | `all` | 320 | 0 | 720.41 | 707.49 | +1.8% | 52815.52 | 53878.52 | 38.90 | 39.71 |
| N300 | 1 | `decode_only` | 320 | 0 | 613.68 | 612.88 | +0.1% | 64186.96 | 64354.26 | 24.13 | 24.13 |
| N300 | 1 | `all` | 320 | 0 | 613.85 | 613.53 | +0.1% | 64140.79 | 64218.68 | 24.14 | 24.14 |
| N300 | 2 | `decode_only` | 320 | 0 | 658.71 | 657.17 | +0.2% | 57749.45 | 57935.35 | 42.58 | 42.72 |
| N300 | 2 | `all` | 320 | 0 | 611.45 | 571.85 | +6.9% | 57661.22 | 58135.90 | 42.61 | 42.28 |
| T3K | 1 | `decode_only` | 320 | 0 | 892.98 | 912.90 | -2.2% | 44156.65 | 43187.82 | 16.84 | 16.40 |
| T3K | 1 | `all` | 320 | 0 | 907.25 | 899.15 | +0.9% | 43401.87 | 43716.72 | 16.63 | 16.90 |
| T3K | 4 | `decode_only` | 320 | 0 | 822.75 | 814.80 | +1.0% | 37584.38 | 38299.82 | 60.58 | 61.44 |
| T3K | 4 | `all` | 320 | 0 | 824.30 | 813.62 | +1.3% | 37601.70 | 38346.00 | 60.74 | 61.63 |
| T3K | 8 | `decode_only` | 320 | 0 | 1039.44 | 1021.38 | +1.8% | 28262.52 | 29208.23 | 81.29 | 83.24 |
| T3K | 8 | `all` | 320 | 0 | 1039.20 | 1048.00 | -0.8% | 27864.69 | 27640.79 | 81.64 | 81.24 |

Conclusion:

- The global-slot sampled-token contract is functionally validated across all
  requested platforms, DP sizes, and trace modes: all 12 rows completed
  `320/320` requests with `0` failures.
- The rerun matches the prior 128-input matrix within expected run-to-run
  variance. Output throughput ranges from `-2.2%` to `+6.9%` versus the prior
  run, with most rows within about `2%`.
- The previous vLLM compact-shape fallback is not needed once TTTv2 returns the
  global-slot shape and signed token dtype at the model boundary.

### TTTv2 global sampled-token quality smoke

Follow-up deterministic text-quality smoke on 2026-07-11 after the successful
global sampled-token full matrix.

Artifact root:
`/localdev/gwang/vllm_duo/perf_results/tttv2_global_token_quality_smoke_20260711T165839Z/`.

Procedure:

- Reused the same 12 platform/DP/trace-mode combinations as the full matrix:
  N150 DP1, N300 DP1/DP2, T3K DP1/DP4/DP8, each with `trace_mode=decode_only`
  and `trace_mode=all`.
- Used `TT_LLAMA_TEXT_VER=tt_transformers_v2`,
  `sample_on_device_mode=all`, `trace_region_size=85000000`, async scheduling,
  and T3K `fabric_config=FABRIC_1D`.
- Request: `/v1/completions`, prompt
  `Explain why rain can make roads slippery in three sentences.`,
  `max_tokens=80`, `temperature=0`, `ignore_eos=true`.
- Hardware was reset before and after each row.
- Failure-signature scan found no `Traceback`, `RuntimeError`, `EngineDead`,
  `index_cpu`, sampled-token, invalid-token, OOM, or SIGKILL signatures. The
  final process check found no lingering server/benchmark processes.

Results:

| Device | DP | Trace | Status | Finish | Tokens | Text prefix |
|---|---:|---|---|---|---:|---|
| N150 | 1 | `decode_only` | ok | length | 80 | `When rain falls on a road, it creates a layer of water on the surface...` |
| N150 | 1 | `all` | ok | length | 80 | `When rain falls on a road, it creates a layer of water on the surface...` |
| N300 | 1 | `decode_only` | ok | length | 80 | `When rain falls on a road, it creates a layer of water that can make the surface slippery...` |
| N300 | 1 | `all` | ok | length | 80 | `When rain falls on a road, it creates a layer of water that can make the surface slippery...` |
| N300 | 2 | `decode_only` | ok | length | 80 | `When rain falls on a road, it creates a layer of water on the surface...` |
| N300 | 2 | `all` | ok | length | 80 | `When rain falls on a road, it creates a layer of water on the surface...` |
| T3K | 1 | `decode_only` | ok | length | 80 | `When rain falls on a road, it can create a layer of water on the surface...` |
| T3K | 1 | `all` | ok | length | 80 | `When rain falls on a road, it can create a layer of water on the surface...` |
| T3K | 4 | `decode_only` | ok | length | 80 | `When rain falls on a road, it creates a layer of water that can make the surface slippery...` |
| T3K | 4 | `all` | ok | length | 80 | `When rain falls on a road, it creates a layer of water that can make the surface slippery...` |
| T3K | 8 | `decode_only` | ok | length | 80 | `When rain falls on a road, it creates a layer of water on the surface...` |
| T3K | 8 | `all` | ok | length | 80 | `When rain falls on a road, it creates a layer of water on the surface...` |

Conclusion:

- Text quality is coherent and on-topic across all 12 platform/DP/trace-mode
  rows.
- For each fixed platform/DP pair, `decode_only` and `all` produced matching
  deterministic prefixes, which is the expected behavior for this greedy
  smoke.
- All rows reached the requested `max_tokens=80` cap because `ignore_eos=true`;
  this is expected and not a quality failure.
