# vLLM TT Plugin

Tenstorrent backend plugin package for the TT vLLM fork.

This package owns TT-specific model registration, platform configuration,
scheduling, worker execution, engine-core behavior, and `tt-run` launcher
integration. The shared vLLM fork provides generic extension hooks; TT
behavior should live here instead of in shared core.

## Install

Install from the repository root:

```bash
pip install -e plugins/vllm-tt-plugin
```

## Entry Points

The package registers two vLLM plugin entry points:

- `vllm.general_plugins`: `vllm_tt_plugin.entrypoints:register`
- `vllm.platform_plugins`: `vllm_tt_plugin.entrypoints:platform_plugin`

`platform_plugin()` returns `vllm_tt_plugin.platform.TTPlatform` only when the
TT runtime is importable. Model registration is idempotent and runs through
`vllm_tt_plugin.model_registry`.

## Runtime Ownership

`TTPlatform.check_and_update_config()` selects the TT-owned runtime classes:

- `parallel_config.worker_cls = "vllm_tt_plugin.worker.TTWorker"`
- `parallel_config.engine_core_cls = "vllm_tt_plugin.engine.TTEngineCore"`
- `parallel_config.engine_core_proc_cls = "vllm_tt_plugin.engine.TTEngineCoreProc"`
- `parallel_config.dp_engine_core_proc_cls = "vllm_tt_plugin.engine.TTDPEngineCoreProc"`
- `parallel_config.engine_core_launcher_cls = "vllm_tt_plugin.launcher.TTCoreEngineLauncher"`
- `scheduler_config.scheduler_cls = "vllm_tt_plugin.scheduler.TTScheduler"`

The plugin also owns the TT model runner, model loader, async decode helper, DP
gather execution path, and mixed local/MPI launch path.

## Configuration

Prefer the generic plugin namespace:

```bash
--plugin-config '{"tt": {"sample_on_device_mode": "all"}}'
```

TT code reads this through `vllm_tt_plugin.config.get_tt_config()`, which returns
`vllm_config.plugin_config["tt"]`.

The fork still accepts the compatibility flag:

```bash
--override-tt-config '{"trace_mode": "decode_only"}'
```

During the current transition, shared argparse maps `--override-tt-config` into
`plugin_config["tt"]`. New TT options should be handled through the `tt` plugin
config namespace.

Common TT config keys include:

- `register_test_models`
- `sample_on_device_mode`
- `trace_mode`
- `enable_model_warmup`
- `optimizations`
- `rank_binding`
- `mpi_args`
- `extra_ttrun_args`
- `config_pkl_dir`
- `env_passthrough`

Launcher keys such as `rank_binding`, `mpi_args`, `extra_ttrun_args`,
`config_pkl_dir`, and `env_passthrough` are used by
`TTCoreEngineLauncher` for mixed local and `tt-run`/MPI engine launches.

## Scope

This plugin targets the TT vLLM fork with generic hooks. It is not a
standalone stock-upstream vLLM plugin until those hooks are available upstream.
