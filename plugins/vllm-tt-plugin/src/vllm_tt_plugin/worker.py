# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import math
import os
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Optional

import torch
import ttnn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_architecture
from vllm.tasks import SupportedTask
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MLAAttentionSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase
from vllm_tt_plugin.config import get_tt_config
from vllm_tt_plugin.model_runner import TTModelInput, TTModelRunner
from vllm_tt_plugin.platform import (
    TTPlatform,
    _should_pre_register_tt_test_models_from_cli,
    register_tt_models,
)

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.outputs import LogprobsLists

logger = init_logger(__name__)

# Ensure TT model architectures are registered in this process as early as
# possible. `WorkerWrapperBase.init_worker` imports the worker class module
# before initializing multimodal caches; without this, early architecture
# inspection may fail for TT-prefixed architectures.
register_tt_models(register_test_models=_should_pre_register_tt_test_models_from_cli())


class TTWorker(WorkerBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = True,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )

        # Initialized by init_device
        self.mesh_device = None

        # Whether to use ttnn tracing for model execution
        override_tt_config = get_tt_config(self.vllm_config)
        trace_key = "trace_mode"
        self.trace_mode = "all"
        if override_tt_config and trace_key in override_tt_config:
            assert override_tt_config[trace_key] in ["decode_only", "all", "none"], (
                f"Invalid {trace_key}: {override_tt_config[trace_key]}"
            )
            self.trace_mode = override_tt_config[trace_key]

        enable_model_warmup_key = "enable_model_warmup"
        self.enable_model_warmup = True
        if override_tt_config and enable_model_warmup_key in override_tt_config:
            assert override_tt_config[enable_model_warmup_key] in [True, False], (
                f"Invalid {enable_model_warmup_key}: \
                {override_tt_config[enable_model_warmup_key]}"
            )

            self.enable_model_warmup = override_tt_config[enable_model_warmup_key]

    def init_device(self) -> None:
        # Validate/apply TT config in this worker process (multiprocessing
        # means platform class attrs + config mutations must be applied per
        # subprocess) before runner init.
        TTPlatform.check_and_update_config(self.vllm_config)

        local_dp_rank = self.parallel_config.data_parallel_rank_local
        # Open mesh only on local DP rank 0 (device ranks).
        if local_dp_rank == 0:
            self.mesh_device = open_mesh_device(
                get_tt_config(self.vllm_config), self.trace_mode, local_dp_rank
            )
            self.device_config.device = self.mesh_device
            assert self.mesh_device is not None
            self.device_config.num_devices = self.mesh_device.get_num_devices()
        else:
            mesh_grid = get_mesh_grid(local_dp_rank)
            self.mesh_device = None
            # Num devices is required for determining num blocks in KV cache.
            self.device_config.num_devices = mesh_grid[0] * mesh_grid[1]
        # Init ModelRunner here, so that we have access to self.mesh_device.
        self.model_runner: TTModelRunner = TTModelRunner(
            vllm_config=self.vllm_config,
            mesh_device=self.mesh_device,
            trace_mode=self.trace_mode,
            enable_model_warmup=self.enable_model_warmup,
        )

    def load_model(self):
        # Only local DP rank 0 (device rank) loads the model
        if self.parallel_config.data_parallel_rank_local == 0:
            self.model_runner.load_model()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        For the GPU/TPU backends, this method generates the KVCacheSpec by
        parsing the kv cache format from each Attention module in the static
        forward context (compilation_config.static_forward_context).
        core/kv_cache_utils.py uses the KVCacheSpec along with available
        memory info from a profiling run to determine num blocks.

        For the TT backend, the static forward context is not populated since
        the modelling code is independent. Two paths are supported:

        1. Hybrid models (mixed sliding-window + full-attention layers, e.g.
           Gemma3/4 / GPT-OSS) opt in by defining a ``get_kv_cache_spec``
           classmethod on the registered TT model class:

               @classmethod
               def get_kv_cache_spec(
                   cls, vllm_config
               ) -> dict[str, KVCacheSpec] | None:
                   ...

           The returned dict maps a layer name to its per-layer spec. This
           lets upstream's hybrid kv cache manager pack each attention type
           into its own group with its own per-request block budget.

        2. Models without the hook (and models that return ``None``) fall
           back to a single homogeneous spec under the dummy ``"foo"`` layer
           name, the same behaviour the TT backend has always had. As before
           we don't run profiling for available memory and instead override
           num blocks via ``self.cache_config.num_gpu_blocks_override``.
        """
        spec_from_hook = self._try_get_spec_from_model_hook()
        if spec_from_hook is not None:
            return spec_from_hook

        return self._build_default_kv_cache_spec()

    def _try_get_spec_from_model_hook(self) -> dict[str, KVCacheSpec] | None:
        """If the resolved TT model class implements ``get_kv_cache_spec``,
        invoke it and return the result. Returns ``None`` when the hook is
        absent or explicitly returns ``None`` (signalling fallback to the
        single-spec default).
        """
        from vllm.model_executor.models.registry import ModelRegistry

        # ``ModelConfig.architecture`` (singular) is computed in
        # ``ModelConfig.__post_init__`` from ``hf_config.architectures``
        # *before* :meth:`TTPlatform.check_and_update_config` prepends ``"TT"``
        # to the architectures list. As a result the cached property still
        # holds the upstream (e.g. CUDA) name, and resolving it would find
        # upstream's vLLM model class — which doesn't have our
        # ``get_kv_cache_spec`` hook. Prefer the prefixed entry from the
        # ``architectures`` list (which the platform modifies in-place) and
        # fall back to prepending ``"TT"`` when neither is available.
        arch = next(
            (a for a in self.model_config.architectures if a.startswith("TT")),
            None,
        )
        if arch is None:
            arch = self.model_config.architecture
            if not arch.startswith("TT"):
                arch = "TT" + arch
        model_cls, _ = ModelRegistry.resolve_model_cls(
            arch, model_config=self.model_config
        )
        hook = getattr(model_cls, "get_kv_cache_spec", None)
        if hook is None:
            return None
        spec = hook(self.vllm_config)
        if spec is None:
            return None
        if not isinstance(spec, dict) or not all(
            isinstance(k, str) and isinstance(v, KVCacheSpec) for k, v in spec.items()
        ):
            raise TypeError(
                f"{model_cls.__name__}.get_kv_cache_spec() must return "
                f"dict[str, KVCacheSpec] or None, got {type(spec).__name__}"
            )
        return spec

    def _build_default_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Single-layer spec used by the legacy non-hybrid path. Downstream
        sizing is overridden via ``cache_config.num_gpu_blocks_override``.
        """
        model_config = self.model_config
        parallel_config = self.parallel_config
        cache_config = self.cache_config

        # Excludes TP factor since that is handled on the model side for TT.
        total_num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )

        use_mla = model_config.use_mla
        sliding_window = model_config.get_sliding_window()
        attn_spec: KVCacheSpec
        if use_mla:
            assert not sliding_window, "MLA not supported for sliding window"
            attn_spec = MLAAttentionSpec(
                block_size=cache_config.block_size,
                num_kv_heads=total_num_kv_heads,
                head_size=head_size,
                dtype=dtype,
            )
        else:
            attn_spec = FullAttentionSpec(
                block_size=cache_config.block_size,
                num_kv_heads=total_num_kv_heads,
                head_size=head_size,
                dtype=dtype,
                sliding_window=sliding_window,
            )
        return {"foo": attn_spec}

    def determine_available_memory(self) -> int:
        """
        For the GPU/TPU backends, this method runs profiling to determine
        available memory for the KV cache. The available memory is then used
        in conjunction with the output of get_kv_cache_spec to determine
        the number of kv cache blocks (total memory / page_size / num layers).

        Currenly we just return a large dummy number of bytes similar to the
        Spyre/Neuron backends and override the number of kv cache blocks.
        """

        # TODO: Once we can run profiling, return real available memory
        # instead of overriding the number of blocks.
        num_tt_blocks = get_num_available_blocks_tt(self.vllm_config)
        self.cache_config.num_gpu_blocks_override = num_tt_blocks
        return 1 << 64

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate TT KV cache (only DP rank 0) and initialize persistent
        input batch (all DP ranks) with the specified kv_cache_config.
        """
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        # Cache is already initialized in initialize_from_config.
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    def compile_or_warm_up_model(self) -> None:
        if not self.enable_model_warmup:
            logger.warning("Skipping model warmup")
            return
        local_rank = self.parallel_config.data_parallel_rank_local
        if local_rank == 0:
            self.model_runner.warmup_model()

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        """Expose the non-DP TT execution service to the executor layer.

        Returns the runner's non-DP execution result for the provided
        scheduler output.
        """
        return self.execute_model_with_grammar(scheduler_output, None)

    def execute_model_with_grammar(
        self,
        scheduler_output: "SchedulerOutput",
        grammar_output: "GrammarOutput | None",
    ) -> ModelRunnerOutput | None:
        """Execute a non-DP TT step with plugin-owned structured-output data."""
        assert self.is_driver_worker, "There should only be one Worker for TT"
        output = self.model_runner.execute_model(scheduler_output, grammar_output)
        return output

    def check_health(self) -> None:
        # Worker will always be healthy as long as it's running.
        return

    # ---- DP gather hooks called by DPEngineCoreProc in core.py ----

    def build_dp_model_input(
        self,
        scheduler_output: Optional["SchedulerOutput"],
        grammar_output: Optional["GrammarOutput"],
    ) -> tuple[
        TTModelInput | None,
        int,
        int,
        int,
        int,
        int,
        int,
        list[str],
        dict[str, int],
    ]:
        """Build the local DP payload consumed by gathered-DP orchestration.

        Returns `(local_input, max_blocks, has_structured_input,
        has_penalties, reset_batch, can_sample_device, needs_logprobs,
        req_ids, req_id_to_index)`, where `local_input` is this rank's
        TT model input (or `None`) and the remaining fields are the
        per-rank metadata consumed by gathered-DP orchestration.
        """
        return self.model_runner.prepare_dp_model_input(
            scheduler_output, grammar_output
        )

    def can_attempt_steady_dp_decode_from_scheduler(
        self,
        scheduler_output: Optional["SchedulerOutput"],
        grammar_output: Optional["GrammarOutput"],
    ) -> bool:
        """Return whether this rank can submit decode one step ahead.

        This checks only local runner invariants. The engine combines all ranks'
        answers into a single global decision before using the DP steady path.
        """
        return self.model_runner.can_attempt_steady_dp_decode_from_scheduler(
            scheduler_output, grammar_output
        )

    def can_attempt_steady_decode_from_scheduler(
        self,
        scheduler_output: "SchedulerOutput",
        grammar_output: Optional["GrammarOutput"],
    ) -> bool:
        """Return whether a scheduled non-DP step can overlap steady decode."""
        return self.model_runner.can_attempt_steady_decode_from_scheduler(
            scheduler_output, grammar_output
        )

    def build_dp_decode_gather_input(
        self,
        model_input: TTModelInput | None,
        max_blocks_decode_batch: int,
        any_structured_inputs: bool,
        any_penalties_inputs: bool,
    ) -> dict[str, Any]:
        """Prepare the fixed-shape decode gather payload for DP orchestration.

        Returns the fixed-shape decode gather payload used by gathered-DP
        orchestration.
        """
        return self.model_runner.build_dp_decode_gather_input(
            model_input,
            max_blocks_decode_batch,
            any_structured_inputs,
            any_penalties_inputs,
        )

    def concat_and_execute_dp(
        self,
        inputs: list[TTModelInput | None] | dict[str, Any],
        is_decode: bool,
        max_blocks_decode_batch: int | None,
        any_structured_inputs: bool,
        non_block: bool = False,
    ) -> Any:
        """Execute one merged DP batch through the worker facade.

        Returns either the packed DP execution result or an async DP decode
        wrapper for the merged batch. The worker also enforces the "device rank
        0 only" rule for merged TT execution.
        """
        assert self.is_driver_worker, "concat_and_execute_dp must run on driver"

        local_dp_rank = self.parallel_config.data_parallel_rank_local
        if local_dp_rank != 0:
            return self._empty_dp_execute_result()

        return self.model_runner.submit_dp_execution(
            inputs,
            is_decode,
            max_blocks_decode_batch,
            any_structured_inputs,
            non_block=non_block,
        )

    def _empty_dp_execute_result(self) -> tuple[torch.Tensor, list]:
        """Return the neutral DP payload for non-device local ranks.

        Produces the correctly shaped no-op DP payload for colocated ranks that
        do not execute the merged TT batch.
        """
        world = self.parallel_config.data_parallel_size
        batch_size = int(self.model_runner.scheduler_config.max_num_seqs)
        return torch.zeros((world, batch_size, 1), dtype=torch.int32), [None] * world

    def apply_dp_execution_result(
        self,
        sampled_token_ids: torch.Tensor,
        logprobs_lists: Optional["LogprobsLists"] = None,
        req_ids: list[str] | None = None,
        req_id_to_index: dict[str, int] | None = None,
    ) -> ModelRunnerOutput:
        """Apply the local DP rank result through the worker facade.

        Applies the local DP rank result and returns the corresponding
        `ModelRunnerOutput`.
        """
        return self.model_runner.apply_dp_execution_result(
            sampled_token_ids,
            logprobs_lists,
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
        )

    # ---- Destructor (used to close devices) ----

    def __del__(self):
        # Delete model runner first in case there are model artifacts
        with suppress(AttributeError):
            # attributes may be already torn down when destructor is called
            del self.model_runner

            if self.mesh_device:
                close_mesh_device(self.mesh_device, get_tt_config(self.vllm_config))
                del self.mesh_device

        if hasattr(super(), "__del__"):
            super().__del__()  # type: ignore


def get_num_available_blocks_tt(vllm_config: VllmConfig) -> int:
    """
    Used to set the number of available blocks for the TT KV cache as we
    currently do not run profiling to determine available memory.
    """

    model_config = vllm_config.model_config
    device_config = vllm_config.device_config
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config

    # region Get default or model- and device-specific `max_tokens_all_users`
    try:
        data_parallel = vllm_config.parallel_config.data_parallel_size
        model_class, _ = get_model_architecture(model_config)
        max_tokens_all_users = model_class.get_max_tokens_all_users(
            model_name=model_config.model,
            num_devices=device_config.num_devices,
            tt_data_parallel=data_parallel,
        )

        logger.info(
            "Getting max_tokens_all_users=%d for number of blocks in KV cache "
            "from generator '%s'.",
            max_tokens_all_users,
            model_class,
        )
    except AttributeError:
        max_tokens_all_users = 131_072

        logger.warning(
            "Setting max_tokens_all_users=%d for number of blocks in KV cache "
            "using rules in `get_num_available_blocks_tt`.",
            max_tokens_all_users,
        )
    # endregion

    # To fit a max batch with (max_tokens_all_users / max batch) per user,
    # allocate an extra block_size per user since vLLM uses a worst-case
    # heuristic and assumes each touched block will require a new
    # allocation. E.g. batch 32, block 64 needs an extra 2048 tokens.
    max_batch = scheduler_config.max_num_seqs
    max_tokens_all_users += cache_config.block_size * max_batch

    # Hybrid attention models (Gemma3/4, GPT-OSS, ...) split layers into
    # multiple kv_cache_groups: a full-attention group plus several
    # sliding-window groups. Upstream's hybrid manager packs these into
    # ``group_size = min(layer_counts_per_type)`` buffers and indexes them
    # via per-group block tables, so each request consumes
    # ``full_blocks_per_request + Σ sliding_blocks_per_request`` block IDs
    # from the pool. Our heuristic above sizes ``num_tt_blocks`` for the
    # full-attention demand only; add headroom for the sliding overhead so
    # hybrid models don't run out of blocks at scheduled batch.
    sliding_window = model_config.get_sliding_window()
    if sliding_window is not None:
        # Conservative cap: assume up to a few sliding groups per buffer
        # (typical for Gemma3 5:1 / GPT-OSS 1:1 hybrid patterns) and add
        # ``sliding_window * max_batch`` worth of tokens per group as
        # padding. The exact number of sliding groups isn't known here
        # (the spec hook hasn't run yet); bound it with a small constant
        # rather than walking the model layer types from raw HF config.
        _MAX_SLIDING_GROUPS_HEURISTIC = 8
        max_tokens_all_users += (
            sliding_window * max_batch * _MAX_SLIDING_GROUPS_HEURISTIC
        )

    num_tt_blocks = math.ceil(max_tokens_all_users / cache_config.block_size)

    return num_tt_blocks


# TT-NN utilities


def get_dispatch_core_config(override_tt_config):
    dispatch_core_axis: ttnn.DispatchCoreAxis = None
    if override_tt_config is not None and "dispatch_core_axis" in override_tt_config:
        assert override_tt_config["dispatch_core_axis"] in ["row", "col"], (
            "Invalid dispatch_core_axis:"
            f"{override_tt_config['dispatch_core_axis']}. "
            "Expected: row, col."
        )
        dispatch_core_axis = (
            ttnn.DispatchCoreAxis.COL
            if override_tt_config["dispatch_core_axis"] == "col"
            else ttnn.DispatchCoreAxis.ROW
        )

    return ttnn.DispatchCoreConfig(axis=dispatch_core_axis)


def get_fabric_config(override_tt_config, num_devices):
    if num_devices == 1:
        # No fabric config for single device
        fabric_config = None
    else:
        # Set the most common value as default
        is_6u = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
        fabric_config = (
            ttnn.FabricConfig.FABRIC_1D_RING if is_6u else ttnn.FabricConfig.FABRIC_1D
        )

    # Override fabric_config if specified in override_tt_config
    if override_tt_config is not None and "fabric_config" in override_tt_config:
        fabric_config_str = override_tt_config["fabric_config"]
        fabric_config_map = {
            "DISABLED": ttnn.FabricConfig.DISABLED,
            "FABRIC_1D": ttnn.FabricConfig.FABRIC_1D,
            "FABRIC_1D_RING": ttnn.FabricConfig.FABRIC_1D_RING,
            "FABRIC_2D": ttnn.FabricConfig.FABRIC_2D,
            "CUSTOM": ttnn.FabricConfig.CUSTOM,
        }
        fabric_config = fabric_config_map.get(fabric_config_str)
        assert fabric_config is not None, (
            f"Invalid fabric_config: {fabric_config_str}. "
            f"Expected one of {list(fabric_config_map.keys())}."
        )
    return fabric_config


def get_reliability_mode(override_tt_config):
    # Default to strict init and override if specified in override_tt_config.
    reliability_mode = ttnn.FabricReliabilityMode.STRICT_INIT
    if (
        override_tt_config is not None
        and "fabric_reliability_mode" in override_tt_config
    ):
        reliability_mode_str = override_tt_config["fabric_reliability_mode"]
        reliability_mode_map = {
            "STRICT_INIT": ttnn.FabricReliabilityMode.STRICT_INIT,
            "RELAXED_INIT": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
        reliability_mode = reliability_mode_map.get(reliability_mode_str)
        assert reliability_mode is not None, (
            f"Invalid fabric_reliability_mode: {reliability_mode_str}. "
            f"Expected one of {list(reliability_mode_map.keys())}."
        )
    return reliability_mode


# From tt-metal/conftest.py:
# Set fabric config to passed in value
# Do nothing if not set
# Must be called before creating the mesh device
def set_fabric(override_tt_config, num_devices):
    fabric_config = get_fabric_config(override_tt_config, num_devices)
    if fabric_config:
        reliability_mode = get_reliability_mode(override_tt_config)
        logger.info(
            "Setting fabric config: %s, reliability mode: %s",
            fabric_config,
            reliability_mode,
        )
        ttnn.set_fabric_config(fabric_config, reliability_mode)


# From tt-metal/conftest.py:
# Reset fabric config to DISABLED if not None, and do nothing otherwise
# Temporarily require previous state to be passed
# in as even setting it to DISABLED might be unstable
# This is to ensure that we don't propagate
# the instability to the rest of CI
def reset_fabric(override_tt_config, num_devices):
    fabric_config = get_fabric_config(override_tt_config, num_devices)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def device_params_from_override_tt_config(override_tt_config, trace_mode):
    device_params = {}

    if trace_mode in ["all", "decode_only"]:
        # Set the most common value as default, override later
        device_params["trace_region_size"] = 50000000
        if override_tt_config and "trace_region_size" in override_tt_config:
            device_params["trace_region_size"] = override_tt_config["trace_region_size"]

    if override_tt_config and "worker_l1_size" in override_tt_config:
        device_params["worker_l1_size"] = override_tt_config["worker_l1_size"]

    if override_tt_config and "l1_small_size" in override_tt_config:
        device_params["l1_small_size"] = override_tt_config["l1_small_size"]

    return device_params


def get_mesh_grid(local_dp_rank=0):
    if local_dp_rank == 0:
        # Only DP rank 0 should query devices.
        num_devices_available = ttnn.get_num_devices()
    mesh_grid_dict = {
        "N150": (1, 1),
        "P100": (1, 1),
        "P150": (1, 1),
        "P150x2": (1, 2),
        "N300": (1, 2),
        "P300": (1, 2),
        "N150x4": (1, 4),
        "P150x4": (1, 4),
        "T3K": (1, 8),
        "P150x8": (1, 8),
        "P300x2": (1, 4),
        "TG": (8, 4),
    }
    mesh_device_env = os.environ.get("MESH_DEVICE")
    if mesh_device_env is not None:
        try:
            # Try to parse as a literal tuple first
            parsed_value = ast.literal_eval(mesh_device_env)
            if isinstance(parsed_value, tuple) and len(parsed_value) == 2:
                mesh_grid = parsed_value
            else:
                raise ValueError("Not a valid tuple")
        except (ValueError, SyntaxError):
            # If parsing fails, treat as a string key for mesh_grid_dict
            assert mesh_device_env in mesh_grid_dict, (
                f"Invalid MESH_DEVICE: {mesh_device_env}"
            )
            mesh_grid = mesh_grid_dict[mesh_device_env]
    else:
        assert local_dp_rank == 0, (
            "MESH_DEVICE must be set when running with data_parallel_size > 1"
        )
        mesh_grid = (1, num_devices_available)

    assert (
        local_dp_rank != 0
        or ttnn.using_distributed_env()
        or (mesh_grid[0] * mesh_grid[1] <= num_devices_available)
    ), (
        f"Requested mesh grid shape {mesh_grid} is larger than "
        f"number of available devices {num_devices_available}"
    )

    return mesh_grid


def open_mesh_device(override_tt_config, trace_mode, local_dp_rank=0):
    assert local_dp_rank == 0, "open_mesh_device must run on local DP rank 0"
    mesh_grid = get_mesh_grid(local_dp_rank)
    logger.info("Attempting to open mesh device with grid shape %s", mesh_grid)

    device_params = device_params_from_override_tt_config(
        override_tt_config, trace_mode
    )

    # Set fabric before opening the device
    num_devices_requested = mesh_grid[0] * mesh_grid[1]
    set_fabric(override_tt_config, num_devices_requested)

    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(*mesh_grid),
        dispatch_core_config=get_dispatch_core_config(override_tt_config),
        **device_params,
    )
    logger.info(
        "multidevice with %d devices and grid %s is created",
        mesh_device.get_num_devices(),
        mesh_grid,
    )
    return mesh_device


def close_mesh_device(mesh_device, override_tt_config):
    # Read device profiler (no-op if not profiling with tracy)
    ttnn.ReadDeviceProfiler(mesh_device)

    # Close devices
    num_devices = mesh_device.get_num_devices()
    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)

    # Reset fabric
    reset_fabric(override_tt_config, num_devices)
