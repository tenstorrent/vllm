# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
import fcntl
import math
import os
import tempfile
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
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase
from vllm_tt_plugin.config import (
    get_tt_config,
    get_tt_data_parallel_size,
    get_tt_per_lane_max_num_seqs,
    should_open_mesh_for_rank,
)
from vllm_tt_plugin.model_input import TTModelInput
from vllm_tt_plugin.model_runner import TTModelRunner
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

        # region Stashed outputs
        # The upstream ``EngineCore`` batch-queue path always calls
        # ``sample_tokens`` after ``execute_model``.  Since TT performs sampling
        # on-device during the forward pass, we simply return the stashed
        # output produced by execute_model.
        # TODO: Might need refactoring after `tt-metal` introduces forward-sample
        #       decoupling. Attributes affected:
        #       `self.execute_model`,
        #       `self.sample_tokens`,
        #       `self._pending_scheduler_output_for_sample`,
        #       and `self._last_output`.
        # Stash for `execute_model`-`sample_tokens` handoff.
        self._last_output: ModelRunnerOutput | None = None
        # Standard-DP handoff: keep scheduler output until `self.sample_tokens`
        # provides `grammar_output` from upstream engine logic.
        self._pending_scheduler_output_for_sample: SchedulerOutput | None = None
        # endregion

        # Whether to use ttnn tracing for model execution
        tt_config = get_tt_config(self.vllm_config)
        trace_key = "trace_mode"
        self.trace_mode = "all"
        if tt_config and trace_key in tt_config:
            assert tt_config[trace_key] in ["decode_only", "all", "none"], (
                f"Invalid {trace_key}: {tt_config[trace_key]}"
            )
            self.trace_mode = tt_config[trace_key]

        enable_model_warmup_key = "enable_model_warmup"
        self.enable_model_warmup = True
        if tt_config and enable_model_warmup_key in tt_config:
            assert tt_config[enable_model_warmup_key] in [True, False], (
                f"Invalid {enable_model_warmup_key}: \
                {tt_config[enable_model_warmup_key]}"
            )

            self.enable_model_warmup = tt_config[enable_model_warmup_key]

    def init_device(self) -> None:
        # Validate/apply TT config in this worker process (multiprocessing
        # means platform class attrs + config mutations must be applied per
        # subprocess) before runner init.
        TTPlatform.check_and_update_config(self.vllm_config)

        local_dp_rank = self.parallel_config.data_parallel_rank_local
        gathered_dp_mode = TTPlatform.gathered_dp_mode
        if should_open_mesh_for_rank(local_dp_rank, gathered_dp_mode):
            mesh_rank = local_dp_rank or 0
            dp_size = (
                1 if gathered_dp_mode
                else self.parallel_config.data_parallel_size
            )

            # region Open mesh device
            # Serialize device opening across DP ranks. The UMD Cluster
            # constructor programs firmware on ALL physical devices and
            # concurrent opens corrupt kernel binaries (see tt-metal
            # risc_firmware_initializer.cpp: "Launch FW on each device
            # sequentially, since a multithreaded launch leads to
            # initialization hangs").
            # Use a POSIX file lock so ranks open devices one at a time.
            if not gathered_dp_mode and dp_size > 1:
                lock_fd = tempfile.TemporaryFile(suffix=".tt_vllm_mesh_init.lock")

                logger.info("DP rank %d: waiting for mesh device lock...", mesh_rank)

                fcntl.flock(lock_fd, fcntl.LOCK_EX)

                logger.info(
                    "DP rank %d: acquired mesh device lock, opening mesh",
                    mesh_rank,
                )
            else:
                lock_fd = None

            try:
                self.mesh_device = open_mesh_device(
                    get_tt_config(self.vllm_config),
                    self.trace_mode,
                    mesh_rank,
                    dp_size,
                )
            finally:
                if lock_fd is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    lock_fd.close()
                    logger.info(
                        "DP rank %d: released mesh device lock", mesh_rank
                    )
            # endregion

            self.device_config.device = self.mesh_device
            assert self.mesh_device is not None
            self.device_config.num_devices = self.mesh_device.get_num_devices()
        else:
            mesh_grid = get_mesh_grid(0 if local_dp_rank is None else local_dp_rank)
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
        # In standard DP mode each rank owns a device and loads the model.
        local_dp_rank = self.parallel_config.data_parallel_rank_local
        gathered_dp_mode = TTPlatform.gathered_dp_mode
        if should_open_mesh_for_rank(local_dp_rank, gathered_dp_mode):
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
        local_dp_rank = self.parallel_config.data_parallel_rank_local
        gathered_dp_mode = TTPlatform.gathered_dp_mode
        if should_open_mesh_for_rank(local_dp_rank, gathered_dp_mode):
            self.model_runner.warmup_model()

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        """Executes one step according to DP mode semantics.

        In gathered-DP mode, keep the plugin's legacy execute->sample stash
        contract. In standard DP mode, defer execution until ``sample_tokens``
        so upstream grammar ownership remains intact.

        See also notes for `self._last_output`.
        """
        if TTPlatform.gathered_dp_mode:
            output = self.execute_model_with_grammar(scheduler_output, None)
            if output is not EMPTY_MODEL_RUNNER_OUTPUT and output is not None:
                # Real work was done => stash for `sample_tokens`.
                self._last_output = output
                return None
            # If empty/no-op batch, return directly.
            return output

        if scheduler_output.total_num_scheduled_tokens > 0:
            # Standard DP: wait for upstream grammar_output delivered via
            # `sample_tokens` before executing the step.
            self._pending_scheduler_output_for_sample = scheduler_output
            return None

        # Empty/no-op batch: execute directly and pass through.
        return self.execute_model_with_grammar(scheduler_output, None)

    def sample_tokens(
        self,
        grammar_output: "GrammarOutput | None",
    ) -> ModelRunnerOutput | None:
        """Completes sampling according to DP mode semantics.

        See also notes for `self._pending_scheduler_output_for_sample`.
        """
        if TTPlatform.gathered_dp_mode:
            output, self._last_output = self._last_output, None
            return output

        pending = self._pending_scheduler_output_for_sample
        if pending is None:
            raise RuntimeError(
                "`sample_tokens` called in standard DP mode without a pending "
                "`execute_model` step"
            )
        self._pending_scheduler_output_for_sample = None
        return self.execute_model_with_grammar(pending, grammar_output)

    def execute_dummy_batch(self) -> None:
        """No-op for standard DP: each rank has an independent submesh with no
        cross-rank collectives during model execution."""

    def execute_model_with_grammar(
        self,
        scheduler_output: "SchedulerOutput",
        grammar_output: "GrammarOutput | None",
    ) -> ModelRunnerOutput | None:
        """Execute a single-process TT step with plugin-owned structured-output
        data.

        ``execute_model`` handles both plain single-process and lane-DP steps:
        it dispatches on the lane scheduler's per-step plan, so the worker does
        not need to know whether lane-DP is active.
        """
        assert self.is_driver_worker, "There should only be one Worker for TT"
        return self.model_runner.execute_model(scheduler_output, grammar_output)

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
        batch_size = self.model_runner.tt_per_lane_max_num_seqs
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
    cache_config = vllm_config.cache_config

    # region Get default or model- and device-specific `max_tokens_all_users`
    model_class = None
    try:
        tt_data_parallel = get_tt_data_parallel_size(vllm_config)
        model_class, _ = get_model_architecture(model_config)
        # Pass the per-submesh batch (the requests one submesh actually serves),
        # not the global engine capacity, so a model that derives a per-user
        # token budget from ``max_num_seqs`` computes the same value whether
        # parallelism is expressed as gathered DP (each rank its own engine) or
        # single-process lane mode. This matches the padding term below, which
        # also uses ``get_tt_per_lane_max_num_seqs``, and keeps the KV shape
        # identical across both modes.
        max_tokens_all_users = model_class.get_max_tokens_all_users(
            model_name=model_config.model,
            num_devices=device_config.num_devices,
            tt_data_parallel=tt_data_parallel,
            max_model_len=model_config.max_model_len,
            max_num_seqs=get_tt_per_lane_max_num_seqs(vllm_config),
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

    # Cap max_tokens_all_users by max_model_len when explicitly configured,
    # since the KV cache cannot hold more tokens than the model supports.
    # ??? Is it only for Llama-3.2 or for all models?
    max_model_len = getattr(model_config, "max_model_len", None)
    if isinstance(max_model_len, int) and max_model_len < max_tokens_all_users:
        max_tokens_all_users = max_model_len
        logger.info(
            "Capping max_tokens_all_users to max_model_len=%d",
            max_tokens_all_users,
        )

    # To fit a max batch with (max_tokens_all_users / max batch) per user,
    # allocate an extra block_size per user since vLLM uses a worst-case
    # heuristic and assumes each touched block will require a new
    # allocation. E.g. batch 32, block 64 needs an extra 2048 tokens.
    #
    # ``num_blocks`` is applied to each submesh KV cache un-divided, so the
    # padding must use the *per-lane/per-rank* batch -- the number of requests
    # a single submesh actually serves -- not the global engine capacity. In
    # gathered DP this is ``max_num_seqs`` (each rank is its own engine); in
    # single-process lane mode it is ``max_num_seqs // lane count``.
    # Both reduce to the same per-submesh value, keeping the KV shape identical
    # regardless of how parallelism is expressed.
    max_batch = get_tt_per_lane_max_num_seqs(vllm_config)
    max_tokens_all_users += cache_config.block_size * max_batch

    # Hybrid attention models (Gemma3/4, GPT-OSS, ...) normally split layers
    # into multiple kv_cache_groups: a full-attention group plus several
    # sliding-window groups. Upstream's hybrid manager packs these into
    # ``group_size = min(layer_counts_per_type)`` buffers and indexes them via
    # per-group block tables, so each request consumes
    # ``full_blocks_per_request + Σ sliding_blocks_per_request`` block IDs.
    #
    # Whether a given model actually emits SlidingWindowSpec (and therefore
    # needs this sliding-window headroom) is decided per model class via
    # ``_HYBRID_KV_CACHE_GROUPS_ENABLED``. Gemma4 re-enables it (it ships the
    # bounded sliding-window decode fix); Gemma3 / GPT-OSS keep it ``False`` and
    # emit FullAttentionSpec for every layer, so adding headroom for them would
    # over-allocate full-size KV blocks and can OOM Gemma3-27B on T3K. Read the
    # resolved model class's flag rather than a single global so re-enabling for
    # one model doesn't regress the others; default to ``False`` when the class
    # can't be resolved.
    hybrid_kv_cache_groups_enabled = getattr(
        model_class, "_HYBRID_KV_CACHE_GROUPS_ENABLED", False
    )
    sliding_window = model_config.get_sliding_window()
    if hybrid_kv_cache_groups_enabled and sliding_window is not None:
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


def get_dispatch_core_config(tt_config, num_devices=1):
    dispatch_core_axis: ttnn.DispatchCoreAxis = None
    if tt_config is not None and "dispatch_core_axis" in tt_config:
        assert tt_config["dispatch_core_axis"] in ["row", "col"], (
            "Invalid dispatch_core_axis:"
            f"{tt_config['dispatch_core_axis']}. "
            "Expected: row, col."
        )
        dispatch_core_axis = (
            ttnn.DispatchCoreAxis.COL
            if tt_config["dispatch_core_axis"] == "col"
            else ttnn.DispatchCoreAxis.ROW
        )

    # For multi-chip clusters, explicitly set ETH dispatch core type to avoid
    # triggering cluster discovery (and ETH heartbeat checks) in each process.
    # This prevents race conditions when multiple DP rank processes initialize
    # simultaneously.
    if num_devices > 1:
        return ttnn.DispatchCoreConfig(
            type=ttnn.DispatchCoreType.ETH, axis=dispatch_core_axis
        )
    return ttnn.DispatchCoreConfig(axis=dispatch_core_axis)


def get_fabric_config(tt_config, num_devices):
    if num_devices == 1:
        # No fabric config for single device
        fabric_config = None
    else:
        # Set the most common value as default
        is_6u = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
        fabric_config = (
            ttnn.FabricConfig.FABRIC_1D_RING if is_6u else ttnn.FabricConfig.FABRIC_1D
        )

    # Override fabric_config if specified in TT plugin config.
    if tt_config is not None and "fabric_config" in tt_config:
        fabric_config_str = tt_config["fabric_config"]
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


def get_reliability_mode(tt_config):
    # Default to strict init and override if specified in TT plugin config.
    reliability_mode = ttnn.FabricReliabilityMode.STRICT_INIT
    if tt_config is not None and "fabric_reliability_mode" in tt_config:
        reliability_mode_str = tt_config["fabric_reliability_mode"]
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
def set_fabric(tt_config, num_devices):
    fabric_config = get_fabric_config(tt_config, num_devices)
    if fabric_config:
        reliability_mode = get_reliability_mode(tt_config)
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
def reset_fabric(tt_config, num_devices):
    fabric_config = get_fabric_config(tt_config, num_devices)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def device_params_from_tt_config(tt_config, trace_mode):
    device_params = {}

    if trace_mode in ["all", "decode_only"]:
        # Set the most common value as default, override later
        device_params["trace_region_size"] = 50000000
        if tt_config and "trace_region_size" in tt_config:
            device_params["trace_region_size"] = tt_config["trace_region_size"]

    if tt_config and "worker_l1_size" in tt_config:
        device_params["worker_l1_size"] = tt_config["worker_l1_size"]

    if tt_config and "l1_small_size" in tt_config:
        device_params["l1_small_size"] = tt_config["l1_small_size"]

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


def compute_per_rank_mesh(mesh_grid, local_dp_rank, data_parallel_size):
    """Partitions a full mesh grid into per-rank submesh shape and offset.

    Divides rows first to preserve columns for tensor parallelism, then
    splits the remaining DP factor from columns. This produces submeshes
    with maximum column width (e.g. (1,2) from (2,4) with DP=4).
    """
    if data_parallel_size <= 1:
        return mesh_grid, None

    rows, cols = mesh_grid

    # Find largest factor of dp_size that divides rows
    row_factor = 1
    for f in range(min(rows, data_parallel_size), 0, -1):
        if data_parallel_size % f == 0 and rows % f == 0:
            row_factor = f
            break
    col_factor = data_parallel_size // row_factor

    if cols % col_factor != 0:
        raise ValueError(
            f"Cannot partition mesh grid {mesh_grid} into "
            f"{data_parallel_size} equal parts for data parallelism"
        )

    per_rank_rows = rows // row_factor
    per_rank_cols = cols // col_factor
    per_rank_grid = (per_rank_rows, per_rank_cols)

    # 2D tiling: ranks fill columns first within each row strip
    row_idx = local_dp_rank // col_factor
    col_idx = local_dp_rank % col_factor
    offset = ttnn.MeshCoordinate([row_idx * per_rank_rows, col_idx * per_rank_cols])

    return per_rank_grid, offset


def open_mesh_device(tt_config, trace_mode, local_dp_rank=0, data_parallel_size=1):
    """Opens TT mesh device for the given DP config, applying TT config and trace mode.

    Returns the opened mesh device.
    """
    mesh_grid = get_mesh_grid(local_dp_rank)
    per_rank_grid, offset = compute_per_rank_mesh(
        mesh_grid, local_dp_rank, data_parallel_size
    )

    logger.info(
        "DP rank %d/%d: opening submesh %s at offset %s (full mesh %s)",
        local_dp_rank,
        data_parallel_size,
        per_rank_grid,
        offset,
        mesh_grid,
    )

    device_params = device_params_from_tt_config(tt_config, trace_mode)

    # Only enable fabric when opening the full mesh (no DP partitioning).
    # Fabric requires ALL system devices to be active, which is incompatible
    # with per-rank subsets opened via offset.
    if offset is None:
        num_devices_requested = per_rank_grid[0] * per_rank_grid[1]
        set_fabric(tt_config, num_devices_requested)

    open_kwargs = dict(
        dispatch_core_config=get_dispatch_core_config(
            tt_config, per_rank_grid[0] * per_rank_grid[1]
        ),
        **device_params,
    )
    if offset is not None:
        open_kwargs["offset"] = offset

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(*per_rank_grid), **open_kwargs)

    logger.info(
        "multidevice with %d devices and grid %s is created",
        mesh_device.get_num_devices(),
        per_rank_grid,
    )

    return mesh_device


def close_mesh_device(mesh_device, tt_config):
    # Read device profiler (no-op if not profiling with tracy)
    ttnn.ReadDeviceProfiler(mesh_device)

    # Close devices
    num_devices = mesh_device.get_num_devices()
    for submesh in mesh_device.get_submeshes():
        ttnn.close_mesh_device(submesh)
    ttnn.close_mesh_device(mesh_device)

    # Reset fabric
    reset_fabric(tt_config, num_devices)
