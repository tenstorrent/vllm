# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, cast

import torch
import ttnn

from vllm.v1.outputs import AsyncModelRunnerOutput, LogprobsLists, ModelRunnerOutput
from vllm_tt_plugin.input_batch import SEED_NONE_SENTINEL

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm_tt_plugin.input_batch import CachedRequestState
    from vllm_tt_plugin.model_runner import TTModelInput, TTModelRunner


@dataclass(frozen=True)
class TTDecodeSubmission:
    """Carries the raw result of decode submission until finalization."""

    tt_out: Any | None
    read_events: list[Any] | None
    batch_size_per_dp: list[int]
    sampling_params: Any
    perform_device_sampling: bool


@dataclass(frozen=True)
class TTFinalizedDecode:
    """Normalized decode result after TT event waits and host processing."""

    tt_out: torch.Tensor
    tt_log_probs: torch.Tensor | None


@dataclass(frozen=True)
class SubmittedStepContext:
    """Immutable snapshot of the host state associated with one decode submit."""

    req_ids: list[str]
    req_id_to_index: dict[str, int]
    request_states: tuple[CachedRequestState, ...]
    row_indices: tuple[int, ...]
    submit_time_ns: int


@dataclass(frozen=True)
class CompletedDecodeStep:
    """Decode output that has completed readback but is not yet applied."""

    sampled_token_ids: torch.Tensor
    logprobs: LogprobsLists | None
    context: SubmittedStepContext
    completion_time_ns: int


class AsyncTTModelRunnerOutput(AsyncModelRunnerOutput):
    """Wrap a non-blocking TT decode submission plus async read submit."""

    def __init__(
        self,
        controller: TTAsyncDecodeController,
        submission: TTDecodeSubmission,
        model_input: TTModelInput,
        completion_event: threading.Event,
        context: SubmittedStepContext,
    ):
        self._controller = controller
        self._submission = submission
        self._model_input = model_input
        self._completion_event = completion_event
        self._context = context

    def get_output(self) -> ModelRunnerOutput:
        try:
            return self._get_output_impl()
        finally:
            self._completion_event.set()

    def _get_output_impl(self) -> ModelRunnerOutput:
        completed = self._controller.complete_non_dp_decode_step(
            submission=self._submission,
            model_input=self._model_input,
            context=self._context,
        )
        self._controller.enqueue_completed_decode_step(completed)
        return self._controller.build_runner_output_from_completed_step(completed)


class AsyncTTDPGatherOutput(AsyncModelRunnerOutput):
    """Wrap a non-blocking DP decode submission plus async read submit."""

    def __init__(
        self,
        controller: TTAsyncDecodeController,
        submission: TTDecodeSubmission,
        model_input: TTModelInput,
    ):
        self._controller = controller
        self._submission = submission
        self._model_input = model_input

    def get_output(self) -> tuple[torch.Tensor, list]:  # type: ignore[override]
        finalized = self._controller.finalize_decode(self._submission)
        runner = self._controller.runner
        if finalized is None:
            return runner.pack_dp_results(
                [torch.tensor([], dtype=torch.int32)]
                * len(self._submission.batch_size_per_dp),
                [None] * len(self._submission.batch_size_per_dp),
            )

        sampled_token_ids_per_dp, logprobs_per_dp = runner._get_output_tokens(
            tt_out=finalized.tt_out,
            tt_log_probs=finalized.tt_log_probs,
            sampling_params=self._submission.sampling_params,
            model_input=self._model_input,
            batch_size_per_dp=self._submission.batch_size_per_dp,
            perform_device_sampling=self._submission.perform_device_sampling,
            is_decode=True,
            grammar_outputs=None,
        )
        return runner.pack_dp_results(sampled_token_ids_per_dp, logprobs_per_dp)


class TTAsyncDecodeController:
    """Own the TT async decode lifecycle for a `TTModelRunner`."""

    def __init__(self, runner: TTModelRunner):
        self.runner = runner

    def capture_submitted_step_context(self) -> SubmittedStepContext:
        runner = self.runner
        num_reqs = runner.input_batch.num_reqs
        req_ids = list(runner.input_batch.req_ids[:num_reqs])
        return SubmittedStepContext(
            req_ids=req_ids,
            req_id_to_index=dict(runner.input_batch.req_id_to_index),
            request_states=tuple(runner.requests[req_id] for req_id in req_ids),
            row_indices=tuple(
                runner.input_batch.req_id_to_index[req_id] for req_id in req_ids
            ),
            submit_time_ns=time.perf_counter_ns(),
        )

    def steady_decode_base_enabled(self, *, dp_gather: bool) -> bool:
        runner = self.runner
        if dp_gather:
            if not runner.scheduler_config.async_scheduling:
                return False
        else:
            if not runner.non_dp_async_scheduling:
                return False
            if runner.parallel_config.data_parallel_size != 1:
                return False
        if runner.trace_mode == "none":  # noqa: SIM103
            return False
        return True

    def steady_decode_scheduler_invariants_met(
        self,
        scheduler_output: SchedulerOutput,
    ) -> bool:
        runner = self.runner
        cached_reqs = scheduler_output.scheduled_cached_reqs
        is_prompt = (len(scheduler_output.scheduled_new_reqs) > 0) or bool(
            cached_reqs.resumed_req_ids
        )
        if is_prompt or runner._decode_layout_changed_since_last_decode:
            return False
        if scheduler_output.pending_structured_output_tokens:
            return False
        if runner._scheduler_output_has_structured_outputs(scheduler_output):
            return False
        input_batch = runner.input_batch
        if not input_batch.no_penalties:
            return False
        if not input_batch.no_allowed_token_ids:
            return False
        if input_batch.sampling.bad_words_token_ids:
            return False
        max_num_logprobs = input_batch.max_num_logprobs
        # Treat logprobs=0 as a real logprobs request so decode does not
        # bypass the slower path that preserves per-token logprob metadata.
        if max_num_logprobs is not None:
            return False
        if input_batch.sampling.has_active_logitsprocs():
            return False
        if runner.model_config.logits_processors:
            return False
        return runner.check_perform_device_sampling(
            is_decode=True,
            has_structured_outputs=False,
        )

    def can_attempt_steady_decode_from_scheduler(
        self,
        scheduler_output: SchedulerOutput,
    ) -> bool:
        if not self.steady_decode_base_enabled(dp_gather=False):
            return False
        return self.steady_decode_scheduler_invariants_met(scheduler_output)

    def can_attempt_steady_dp_decode_from_scheduler(
        self,
        scheduler_output: SchedulerOutput | None,
    ) -> bool:
        if not self.steady_decode_base_enabled(dp_gather=True):
            return False
        if scheduler_output is None or scheduler_output.total_num_scheduled_tokens == 0:
            return True
        return self.steady_decode_scheduler_invariants_met(scheduler_output)

    def can_use_steady_decode_fast_path(self, model_input: TTModelInput) -> bool:
        if not self.steady_decode_base_enabled(dp_gather=False):
            return False
        if model_input.prompt_lens is not None:
            return False
        if not model_input.perform_device_sampling:
            return False
        if model_input.reset_batch:
            return False
        if model_input.has_structured_outputs:
            return False
        if (
            model_input.prompt_tokens is not None
            or model_input.output_tokens is not None
        ):
            return False
        if model_input.allowed_token_ids_mask_list[0] is not None:
            return False
        if model_input.bad_words_token_ids_list[0]:
            return False
        max_num_logprobs = model_input.max_num_logprobs[0]
        if max_num_logprobs is not None:  # noqa: SIM103
            return False
        return True

    def enqueue_completed_decode_step(self, completed: CompletedDecodeStep) -> None:
        with self.runner._steady_decode_lock:
            self.runner._completed_decode_steps.append(completed)

    def register_pending_async_event(
        self,
        event: threading.Event,
        *,
        overlap_ok: bool,
    ) -> None:
        with self.runner._steady_decode_lock:
            self.runner._pending_async_events.append(event)
            self.runner._pending_async_overlap_ok.append(overlap_ok)

    def prune_finished_async_events(self) -> None:
        with self.runner._steady_decode_lock:
            while (
                self.runner._pending_async_events
                and self.runner._pending_async_events[0].is_set()
            ):
                self.runner._pending_async_events.popleft()
                self.runner._pending_async_overlap_ok.popleft()

    def drain_completed_decode_steps(self) -> list[CompletedDecodeStep]:
        completed: list[CompletedDecodeStep] = []
        with self.runner._steady_decode_lock:
            while self.runner._completed_decode_steps:
                completed.append(self.runner._completed_decode_steps.popleft())
        return completed

    def apply_ready_completed_decode_steps(self) -> None:
        for completed in self.drain_completed_decode_steps():
            self.apply_completed_decode_step(completed)
        self.prune_finished_async_events()

    def wait_for_all_pending_async_steps(self) -> None:
        with self.runner._steady_decode_lock:
            events = list(self.runner._pending_async_events)
        for event in events:
            event.wait()
        self.apply_ready_completed_decode_steps()

    def must_drain_pending_async_steps(
        self,
        steady_decode_candidate: bool,
    ) -> bool:
        with self.runner._steady_decode_lock:
            if not self.runner._pending_async_events:
                return False
            if not steady_decode_candidate:
                return True
            return any(
                not overlap_ok for overlap_ok in self.runner._pending_async_overlap_ok
            )

    def complete_non_dp_decode_step(
        self,
        submission: TTDecodeSubmission,
        model_input: TTModelInput,
        context: SubmittedStepContext,
    ) -> CompletedDecodeStep:
        finalized = self.finalize_decode(submission)
        if finalized is None:
            sampled_token_ids = torch.empty((0, 1), dtype=torch.int32)
            logprobs = None
        else:
            sampled_token_ids_per_dp, logprobs_per_dp = self.runner._get_output_tokens(
                tt_out=finalized.tt_out,
                tt_log_probs=finalized.tt_log_probs,
                sampling_params=submission.sampling_params,
                model_input=model_input,
                batch_size_per_dp=submission.batch_size_per_dp,
                perform_device_sampling=submission.perform_device_sampling,
                is_decode=True,
                grammar_outputs=None,
            )
            sampled_token_ids = sampled_token_ids_per_dp[0]
            logprobs_tensors = logprobs_per_dp[0] if logprobs_per_dp else None
            logprobs = logprobs_tensors.tolists() if logprobs_tensors else None
        return CompletedDecodeStep(
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs,
            context=context,
            completion_time_ns=time.perf_counter_ns(),
        )

    def build_runner_output_from_completed_step(
        self,
        completed: CompletedDecodeStep,
    ) -> ModelRunnerOutput:
        return self.runner._build_runner_output(
            sampled_token_ids=completed.sampled_token_ids,
            logprobs=completed.logprobs,
            req_ids=completed.context.req_ids,
            req_id_to_index=completed.context.req_id_to_index,
        )

    def apply_completed_decode_step(self, completed: CompletedDecodeStep) -> None:
        self.runner._apply_sampled_tokens_to_state(
            sampled_token_ids=completed.sampled_token_ids,
            req_ids=completed.context.req_ids,
            request_states=completed.context.request_states,
            row_indices=completed.context.row_indices,
        )

    def submit_async_non_dp_decode(
        self,
        model_input: TTModelInput,
        *,
        steady_decode_fast_path: bool,
    ) -> AsyncTTModelRunnerOutput:
        event = threading.Event()
        context = self.capture_submitted_step_context()
        submission = self.submit_decode(
            model_input,
            read_from_device=False,
            async_read=True,
        )
        self.register_pending_async_event(
            event,
            overlap_ok=steady_decode_fast_path,
        )
        if submission.tt_out is None:
            event.set()
        return AsyncTTModelRunnerOutput(
            controller=self,
            submission=submission,
            model_input=model_input,
            completion_event=event,
            context=context,
        )

    def submit_async_dp_decode(
        self,
        model_input: TTModelInput,
    ) -> AsyncTTDPGatherOutput:
        submission = self.submit_decode(
            model_input,
            read_from_device=False,
            async_read=True,
        )
        return AsyncTTDPGatherOutput(
            controller=self,
            submission=submission,
            model_input=model_input,
        )

    def submit_decode(
        self,
        model_input: TTModelInput,
        *,
        read_from_device: bool,
        async_read: bool = False,
    ) -> TTDecodeSubmission:
        runner = self.runner
        batch_size_per_dp = model_input.unpadded_batch_size
        if not isinstance(batch_size_per_dp, list):
            batch_size_per_dp = [batch_size_per_dp]

        sampling_params = model_input.tt_sampling_params
        perform_device_sampling = model_input.perform_device_sampling
        if not any(bs > 0 for bs in batch_size_per_dp):
            return TTDecodeSubmission(
                tt_out=None,
                read_events=None,
                batch_size_per_dp=batch_size_per_dp,
                sampling_params=sampling_params,
                perform_device_sampling=perform_device_sampling,
            )

        kwargs: dict[str, Any] = {
            "tokens": model_input.input_tokens,
            "page_table": model_input.block_tables,
            "kv_cache": runner.kv_caches,
            "start_pos": model_input.input_positions,
        }
        # Hybrid attention models route per-layer block tables; the
        # runner already populated ``block_tables_per_layer`` at
        # submission time when the kv_cache_config has multiple groups.
        # Legacy/uniform models leave it as ``None`` and never see the
        # kwarg.
        if model_input.block_tables_per_layer is not None:
            kwargs["page_tables_per_layer"] = model_input.block_tables_per_layer
        if perform_device_sampling:
            sampling_param_dict = {
                field.name: (
                    getattr(sampling_params, field.name).tolist()
                    if getattr(sampling_params, field.name) is not None
                    else None
                )
                for field in fields(sampling_params)
            }
            sampling_param_dict["seed"] = [
                None if s == SEED_NONE_SENTINEL else s
                for s in sampling_param_dict["seed"]
            ]
            kwargs["sampling_params"] = type(sampling_params)(**sampling_param_dict)
            if model_input.prompt_tokens is not None:
                assert model_input.output_tokens is not None
                kwargs["prompt_tokens"] = model_input.prompt_tokens
                kwargs["output_tokens"] = model_input.output_tokens
            kwargs["reset_batch"] = model_input.reset_batch
            if model_input.slot_remap is not None:
                kwargs["slot_remap"] = model_input.slot_remap

        enc_dec_kwargs: dict[str, Any] = {}
        if runner.request_specific_rope:
            if any(
                req_id not in runner.previous_req_ids
                for req_id in runner.input_batch.req_ids
            ):
                enc_dec_kwargs = {
                    "rope_deltas_all_users": [
                        runner.requests[req_id].mrope_position_delta
                        for req_id in runner.input_batch.req_ids
                    ]
                }
            else:
                enc_dec_kwargs = {"rope_deltas_all_users": None}
            runner.previous_req_ids = set(runner.input_batch.req_ids)

        enable_trace = runner.trace_mode in ["all", "decode_only"]
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
            else:
                is_host_tensor = isinstance(tt_out, torch.Tensor)
                is_host_tensor_tuple = isinstance(tt_out, tuple) and all(
                    tensor is None or isinstance(tensor, torch.Tensor)
                    for tensor in tt_out
                )
                if not (is_host_tensor or is_host_tensor_tuple):
                    raise AttributeError(
                        "TT model must implement read_decode_output() "
                        "unless decode_forward() already returns host tensors"
                    )
        return TTDecodeSubmission(
            tt_out=tt_out,
            read_events=read_events,
            batch_size_per_dp=batch_size_per_dp,
            sampling_params=sampling_params,
            perform_device_sampling=perform_device_sampling,
        )

    def finalize_decode(
        self,
        submission: TTDecodeSubmission,
    ) -> TTFinalizedDecode | None:
        runner = self.runner
        if submission.tt_out is None:
            return None

        if submission.read_events is not None:
            for read_event in submission.read_events:
                ttnn.event_synchronize(read_event)
            tt_out = submission.tt_out
        else:
            tt_out = submission.tt_out

        if hasattr(runner.model, "process_decode_output_host"):
            tt_out = runner.model.process_decode_output_host(
                tt_out,
                is_tokens=submission.perform_device_sampling,
            )
        else:
            is_host_tensor = isinstance(tt_out, torch.Tensor)
            is_host_tensor_tuple = isinstance(tt_out, tuple) and all(
                tensor is None or isinstance(tensor, torch.Tensor) for tensor in tt_out
            )
            if not (is_host_tensor or is_host_tensor_tuple):
                raise AttributeError(
                    "TT model must implement process_decode_output_host() "
                    "unless decode output is already a torch tensor"
                )

        tt_log_probs = None
        assert isinstance(submission.sampling_params.enable_log_probs, torch.Tensor)
        if (
            submission.perform_device_sampling
            and submission.sampling_params.enable_log_probs.any()
        ):
            assert isinstance(tt_out, tuple) and len(tt_out) == 2
            tt_out, tt_log_probs = tt_out
        elif isinstance(tt_out, tuple):
            tt_out, _ = tt_out

        return TTFinalizedDecode(tt_out=tt_out, tt_log_probs=tt_log_probs)
