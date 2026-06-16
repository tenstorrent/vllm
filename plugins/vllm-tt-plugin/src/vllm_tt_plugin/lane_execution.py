# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-process multi-lane (lane-DP) execution for ``TTModelRunner``.

In lane mode ``TTModelRunner.input_batch`` is a ``TTLaneInputBatch`` whose
persistent rows are the device decode slots: a request at lane ``l`` local slot
``s`` lives at row ``l * per_lane + s``, and that row IS its device slot for the
request's whole lifetime. So the runner never splits, scatters, or merges per
lane: it reads the slot-ordered rows straight into one merged device input,
executes, and samples the whole slot batch once.

``TTLaneStepExecutor`` holds a back-reference to its ``TTModelRunner`` and is
otherwise stateless. The split of ownership is:
  - ``TTModelRunner`` owns the canonical request map (``requests``), the shared
    submission/sampling helpers, and the persistent batch.
  - ``TTLaneInputBatch`` owns stable row placement and merged sampling state.
  - ``TTStepPlan`` (scheduler-owned, read-only here) carries the scheduled rows
    and per-DP batch sizes for one step.
The executor applies the plan to runner request state and lane batch state, then
drives one merged step. ``extract_step`` is also called from
``TTAsyncDecodeController`` (the only entry point beyond ``execute_model``).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm_tt_plugin.async_decode import AsyncTTModelRunnerOutput, SubmittedStepContext
from vllm_tt_plugin.input_batch import CachedRequestState
from vllm_tt_plugin.lane_scheduler import TTStepPlan, get_tt_step_plan
from vllm_tt_plugin.logprobs import build_logprobs_from_topk
from vllm_tt_plugin.model_input import TTModelInput, TTSamplingParams
from vllm_tt_plugin.structured_output import reorder_grammar_bitmask_for_tt_batch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm_tt_plugin.model_runner import TTModelRunner


class TTLaneStepExecutor:
    """Execute single-process multi-lane TT steps for a ``TTModelRunner``.

    Stateless apart from the ``runner`` back-reference: all persistent state
    (the ``TTLaneInputBatch``, ``requests``, the layout-changed flag, the async
    decode controller) lives on the runner. See the module docstring for the
    row == lane * per_lane + slot layout and the ownership split.
    """

    def __init__(self, runner: TTModelRunner):
        self.runner = runner

    @torch.no_grad()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | AsyncTTModelRunnerOutput:
        """Execute one merged multi-lane TT step in a single process.

        Falls back to the runner's non-DP ``execute_model`` when the scheduler
        attached no lane step plan (i.e. this is not a lane-DP step).
        """
        runner = self.runner
        plan = get_tt_step_plan(scheduler_output)
        if plan is None:
            return runner.execute_model(
                scheduler_output, grammar_output, intermediate_tensors
            )

        # Lane-DP lays requests out at sparse stable slots. Request-specific
        # RoPE (mrope/vision models, ``model_config.uses_mrope``) instead
        # assumes front-packed request rows, so the two are incompatible until
        # the RoPE delta mapping is made slot-aware. No lane-DP model needs it.
        if runner.request_specific_rope:
            raise NotImplementedError(
                "lane-DP does not support request-specific RoPE "
                "(mrope/vision models) yet"
            )

        runner.async_decode.apply_ready_completed_decode_steps()
        steady_decode_candidate = (
            runner.async_decode.can_attempt_steady_dp_decode_from_scheduler(
                scheduler_output, grammar_output
            )
        )
        if runner.async_decode.must_drain_pending_async_steps(steady_decode_candidate):
            runner.async_decode.wait_for_all_pending_async_steps()

        self._update_states(scheduler_output, plan)
        if not scheduler_output.total_num_scheduled_tokens:
            return EMPTY_MODEL_RUNNER_OUTPUT

        scheduled_rows = list(plan.scheduled_rows)
        if not scheduled_rows:
            return EMPTY_MODEL_RUNNER_OUTPUT

        model_input = self._build_input(scheduler_output, grammar_output, plan)
        if plan.is_decode:
            if runner.non_dp_async_scheduling:
                context = self._capture_context(scheduled_rows)
                return runner.async_decode.submit_async_lane_decode(
                    model_input, context, scheduled_rows
                )
            submission = runner.async_decode.submit_decode(
                model_input, read_from_device=True, async_read=False
            )
            finalized = runner.async_decode.finalize_decode(submission)
            assert finalized is not None
            sampled, logprobs = self.extract_step(
                finalized.tt_out,
                finalized.tt_log_probs,
                model_input,
                scheduled_rows,
                is_decode=True,
            )
        else:
            tt_out = runner.submit_prefill(model_input, model_input.unpadded_batch_size)
            tt_log_probs = None
            assert isinstance(
                model_input.tt_sampling_params.enable_log_probs, torch.Tensor
            )
            if (
                model_input.perform_device_sampling
                and model_input.tt_sampling_params.enable_log_probs.any()
            ):
                assert isinstance(tt_out, tuple) and len(tt_out) == 2
                tt_out, tt_log_probs = tt_out
            elif isinstance(tt_out, tuple):
                tt_out, _ = tt_out
            sampled, logprobs = self.extract_step(
                tt_out, tt_log_probs, model_input, scheduled_rows, is_decode=False
            )

        return self._finalize_output(sampled, logprobs, scheduled_rows)

    def _update_states(
        self, scheduler_output: SchedulerOutput, plan: TTStepPlan
    ) -> None:
        """Update cached states and the stable-slot batch from the step plan.

        Unlike the base ``TTModelRunner._update_states`` this does **not** evict
        merely-unscheduled requests: a prefill step can leave running decodes
        unscheduled, and freeing their stable device slot would disturb the
        on-device per-slot seed RNG. Only finished requests, and resumed
        requests whose KV was rebuilt, release their slot. There is no condense.
        """
        runner = self.runner
        lane_batch = runner.lane_batch
        layout_changed = False

        # Finished requests release their slot.
        for req_id in scheduler_output.finished_req_ids:
            runner.requests.pop(req_id, None)
            if lane_batch.remove_request(req_id) is not None:
                layout_changed = True

        # Free cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            runner.encoder_cache.pop(mm_hash, None)

        req_ids_to_add: list[str] = []
        for new_req_data in scheduler_output.scheduled_new_reqs:
            assert new_req_data.sampling_params is not None, (
                "Pooling is not supported for TT yet"
            )
            if new_req_data.prompt_token_ids is None:
                raise NotImplementedError(
                    "TT backend does not support prompt_embeds yet"
                )
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None
            runner.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=None,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
                prompt_embeds=new_req_data.prompt_embeds,
            )
            req_ids_to_add.append(req_id)

        # Running / resumed requests.
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = runner.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            req_state.num_computed_tokens = num_computed_tokens
            if resumed_from_preemption:
                # KV was freed and is being rebuilt; replace block IDs and
                # re-add fresh (drop the stale slot first). The slot may differ
                # afterwards -- acceptable under the exceptional preemption
                # path, which re-prefills the request anyway.
                assert new_block_ids is not None
                req_state.block_ids = new_block_ids
                if lane_batch.remove_request(req_id) is not None:
                    layout_changed = True
                req_ids_to_add.append(req_id)
                continue
            if new_block_ids is not None:
                for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                    block_ids.extend(new_ids)
            req_index = lane_batch.req_id_to_index.get(req_id)
            if req_index is None:
                req_ids_to_add.append(req_id)
                continue
            lane_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                lane_batch.block_table.append_row(new_block_ids, req_index)

        # Place new / resumed requests at scheduler-owned stable rows.
        for req_id in req_ids_to_add:
            lane_batch.add_request_to_row(
                runner.requests[req_id], plan.req_id_to_row[req_id]
            )
            layout_changed = True

        if layout_changed:
            runner._decode_layout_changed_since_last_decode = True
        lane_batch.refresh_logitsprocs()

    def _build_input(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
        plan: TTStepPlan,
    ) -> TTModelInput:
        """Build the merged device input for one lane step (decode or prefill)."""
        if plan.is_decode:
            return self._build_decode_input(scheduler_output, grammar_output, plan)
        return self._build_prefill_input(scheduler_output, grammar_output, plan)

    def _grammar_bitmask(
        self, grammar_output: GrammarOutput | None, batch_length: int
    ) -> torch.Tensor | None:
        """Reorder the scheduler grammar bitmask into a full slot-batch tensor.

        Each structured-output request's bitmask row is placed at its device
        slot row; every other slot is left all-ones (all tokens allowed), so
        the mask lines up with the full slot logits the host sampler reads.
        """
        runner = self.runner
        if grammar_output is None or grammar_output.grammar_bitmask is None:
            return None
        bitmask = torch.from_numpy(grammar_output.grammar_bitmask)
        return reorder_grammar_bitmask_for_tt_batch(
            bitmask=bitmask,
            structured_output_request_ids=grammar_output.structured_output_request_ids,
            req_id_to_index=runner.lane_batch.req_id_to_index,
            req_indices=list(range(batch_length)),
            batch_length=batch_length,
        )

    def _has_structured_outputs(
        self,
        scheduler_output: SchedulerOutput,
        bitmask: torch.Tensor | None,
    ) -> bool:
        runner = self.runner
        if bitmask is not None or scheduler_output.pending_structured_output_tokens:
            return True
        return any(
            (req := runner.requests.get(req_id)) is not None
            and req.sampling_params is not None
            and req.sampling_params.structured_outputs is not None
            for req_id in scheduler_output.num_scheduled_tokens
        )

    def _block_tables(
        self, rows: list[int], zero_gaps: bool, total: int
    ) -> list[torch.Tensor]:
        """Per-group block tables for ``rows`` (one row per slot), each padded
        to ``max_num_blocks_per_req``. When ``zero_gaps`` is set, rows of
        ``range(total)`` that are not in ``rows`` are zeroed (empty decode slots
        carry no blocks)."""
        runner = self.runner
        width = runner.max_num_blocks_per_req
        occupied = set(rows)
        out: list[torch.Tensor] = []
        for bt in runner.lane_batch.block_table.block_tables:
            sel = list(range(total)) if zero_gaps else rows
            bt_cpu = bt.get_cpu_tensor()[sel, :width].clone()
            if bt_cpu.shape[1] < width:
                pad = torch.zeros(
                    bt_cpu.shape[0], width - bt_cpu.shape[1], dtype=bt_cpu.dtype
                )
                bt_cpu = torch.cat([bt_cpu, pad], dim=1)
            if zero_gaps and len(occupied) < total:
                gap = torch.ones(total, dtype=torch.bool)
                gap[list(occupied)] = False
                bt_cpu[gap] = 0
            out.append(bt_cpu.contiguous())
        return out

    def _sampling_params(self, rows: list[int]) -> TTSamplingParams:
        """Slice the slot-ordered sampling tensors to ``rows``."""
        sp = self.runner.lane_batch.sampling
        idx = torch.as_tensor(rows, dtype=torch.long)
        num_logprobs = sp.num_logprobs[idx]
        return TTSamplingParams(
            temperature=sp.temperature[idx],
            top_k=sp.top_k[idx],
            top_p=sp.top_p[idx],
            presence_penalty=sp.presence_penalty[idx],
            frequency_penalty=sp.frequency_penalty[idx],
            repetition_penalty=sp.repetition_penalty[idx],
            seed=sp.seed[idx],
            num_logprobs=num_logprobs,
            enable_log_probs=num_logprobs >= 0,
        )

    def _build_decode_input(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
        plan: TTStepPlan,
    ) -> TTModelInput:
        """Build the merged decode input straight from the slot-ordered batch.

        Every slot is present (gaps padded), so this is the device decode batch
        with no scatter: row == device slot.
        """
        runner = self.runner
        lane_batch = runner.lane_batch
        total = plan.capacity
        occupied = lane_batch.occupied_rows()

        num_tokens = lane_batch.num_tokens
        positions_np = num_tokens[:total].astype(np.int32) - 1  # gaps -> -1
        input_positions = torch.from_numpy(positions_np)
        tokens_np = np.zeros((total, 1), dtype=np.int32)
        for row in occupied:
            tokens_np[row, 0] = lane_batch.token_ids_cpu[row, num_tokens[row] - 1]
        input_tokens = torch.from_numpy(tokens_np)

        block_tables_per_group = self._block_tables(
            occupied, zero_gaps=True, total=total
        )
        rows_all = list(range(total))
        tt_sampling_params = self._sampling_params(rows_all)

        bitmask = self._grammar_bitmask(grammar_output, total)
        has_structured = self._has_structured_outputs(scheduler_output, bitmask)
        perform_device_sampling = runner.check_perform_device_sampling(
            is_decode=True, has_structured_outputs=has_structured
        )

        # The prompt/output token tensors feed device-side penalties only. Host
        # sampling rebuilds them itself in ``build_merged_sampling_metadata``, so
        # building them here too would be dead work on the host path.
        prompt_tokens = output_tokens = None
        if perform_device_sampling and not lane_batch.no_penalties:
            prompt_tokens = lane_batch.make_prompt_token_ids_tensor(rows_all)
            output_tokens = lane_batch.make_output_token_ids_tensor(rows_all)
        reset_batch = runner._decode_layout_changed_since_last_decode
        runner._decode_layout_changed_since_last_decode = False
        slot_remap = lane_batch.pop_slot_remap()  # identity for stable slots

        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=None,
            block_tables=block_tables_per_group[0],
            block_tables_per_group=block_tables_per_group,
            block_tables_per_layer=runner._block_tables_per_layer(
                block_tables_per_group
            ),
            # Device decodes every slot; only used for the empty-batch guard.
            unpadded_batch_size=list(plan.batch_size_per_dp),
            tt_sampling_params=tt_sampling_params,
            multi_modal_kwargs={},
            perform_device_sampling=perform_device_sampling,
            grammar_bitmask=[bitmask],
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reset_batch=reset_batch,
            slot_remap=slot_remap,
            # Host sampling reads the merged batch directly (see
            # ``extract_step``); the per-rank sidecars are unused here.
            allowed_token_ids_mask_list=[None],
            bad_words_token_ids_list=[{}],
            max_num_logprobs=[lane_batch.max_num_logprobs],
            logitsprocs_list=[None],
            generators_list=[{}],
            prefill_empty_slots=None,
        )

    def _build_prefill_input(
        self,
        scheduler_output: SchedulerOutput,
        grammar_output: GrammarOutput | None,
        plan: TTStepPlan,
    ) -> TTModelInput:
        """Build the prefill input for the requests scheduled this step.

        Prefill rows are front-packed in scheduler plan order. The plan carries
        the stable slots (``prefill_empty_slots``) so ``submit_prefill`` seeds
        each user at the device row decode will later read from. The output is
        one token per prefilled request, in this same order.
        """
        runner = self.runner
        lane_batch = runner.lane_batch
        rows = list(plan.input_rows)
        rows_np = np.asarray(rows, dtype=np.int64)
        input_positions = torch.from_numpy(
            lane_batch.num_computed_tokens_cpu[rows_np].astype(np.int32)
        )
        prompt_lens = lane_batch.num_tokens[rows_np]
        max_prefill = int(prompt_lens.max())
        input_tokens = lane_batch.token_ids_cpu_tensor[rows_np, :max_prefill]

        block_tables_per_group = self._block_tables(rows, zero_gaps=False, total=0)
        tt_sampling_params = self._sampling_params(rows)
        batch_size_per_dp = list(plan.batch_size_per_dp)

        bitmask = self._grammar_bitmask(grammar_output, lane_batch.max_num_reqs)
        has_structured = self._has_structured_outputs(scheduler_output, bitmask)
        perform_device_sampling = runner.check_perform_device_sampling(
            is_decode=False, has_structured_outputs=has_structured
        )

        # Device-side penalties only; host sampling rebuilds these in
        # ``build_merged_sampling_metadata`` (over the full slot batch), so
        # building them here on the host path would be dead work.
        prompt_tokens = output_tokens = None
        if perform_device_sampling and not lane_batch.no_penalties:
            prompt_tokens = lane_batch.make_prompt_token_ids_tensor(rows)
            output_tokens = lane_batch.make_output_token_ids_tensor(rows)

        multi_modal_kwargs = (
            runner._gather_multi_modal_inputs(req_indices=list(rows))
            if runner.model_config.is_multimodal_model
            else {}
        )

        return TTModelInput(
            input_tokens=input_tokens,
            input_positions=input_positions,
            prompt_lens=prompt_lens,
            block_tables=block_tables_per_group[0],
            block_tables_per_group=block_tables_per_group,
            block_tables_per_layer=runner._block_tables_per_layer(
                block_tables_per_group
            ),
            unpadded_batch_size=batch_size_per_dp,
            tt_sampling_params=tt_sampling_params,
            multi_modal_kwargs=multi_modal_kwargs,
            perform_device_sampling=perform_device_sampling,
            grammar_bitmask=[bitmask],
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            reset_batch=False,
            slot_remap=None,
            allowed_token_ids_mask_list=[None],
            bad_words_token_ids_list=[{}],
            max_num_logprobs=[lane_batch.max_num_logprobs],
            logitsprocs_list=[None],
            generators_list=[{}],
            prefill_empty_slots=(
                list(plan.prefill_empty_slots)
                if plan.prefill_empty_slots is not None
                else None
            ),
        )

    def _capture_context(self, scheduled_rows: list[int]) -> SubmittedStepContext:
        """Snapshot the scheduled requests for deferred async state application.

        ``req_ids`` are the scheduled rows' requests in row order, which is the
        canonical merged output order.
        """
        runner = self.runner
        req_ids = [runner.lane_batch.req_ids[row] for row in scheduled_rows]
        return SubmittedStepContext(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            request_states=tuple(runner.requests[rid] for rid in req_ids),
            submit_time_ns=time.perf_counter_ns(),
        )

    def extract_step(
        self,
        tt_out: Any,
        tt_log_probs: Any,
        model_input: TTModelInput,
        scheduled_rows: list[int],
        is_decode: bool,
    ) -> tuple[torch.Tensor, LogprobsLists | None]:
        """Read back one merged lane step into per-request sampled tokens.

        Returns ``(sampled_token_ids[n, 1], logprobs)`` for the ``n``
        ``scheduled_rows`` in order. Device sampling reads the sampled tokens
        directly from each slot; host sampling runs **one** sampler call over
        the whole slot batch (so the builtin/custom logits processors stay
        row-aligned, with no per-lane slicing) and then picks the scheduled
        rows out of the result.

        Also called from ``TTAsyncDecodeController`` to finalize an async
        lane-decode step.
        """
        runner = self.runner
        n = len(scheduled_rows)
        rows_t = torch.as_tensor(scheduled_rows, dtype=torch.long)
        if model_input.perform_device_sampling:
            tokens = tt_out.reshape(-1) if isinstance(tt_out, torch.Tensor) else tt_out
            # Decode reads each scheduled slot; prefill returns one token per
            # scheduled request, already in row order.
            sampled = tokens[rows_t] if is_decode else tokens[:n]
            sampled = sampled.reshape(n, 1).to(torch.int32)
            logprobs = self._device_logprobs(
                tt_log_probs, model_input, scheduled_rows, sampled, is_decode
            )
            return sampled, logprobs

        # Host sampling over the full slot batch.
        total = runner.lane_batch.max_num_reqs
        logits = self._host_logits(tt_out, scheduled_rows, is_decode, total)
        bitmask = model_input.grammar_bitmask[0]
        if bitmask is not None:
            runner.apply_grammar_bitmask(logits, bitmask)
        sampling_metadata = runner.lane_batch.build_merged_sampling_metadata(
            scheduled_rows
        )
        sampler_output = runner.host_sampler(
            logits=logits, sampling_metadata=sampling_metadata
        )
        sampled = sampler_output.sampled_token_ids.reshape(-1)[rows_t].reshape(n, 1)
        logprobs = self._host_logprobs(sampler_output.logprobs_tensors, scheduled_rows)
        return sampled.to(torch.int32), logprobs

    def _host_logits(
        self, tt_out: Any, scheduled_rows: list[int], is_decode: bool, total: int
    ) -> torch.Tensor:
        """Full ``[total, vocab]`` slot logits for host sampling.

        Decode logits already cover every slot. Prefill logits cover only the
        scheduled requests (row order), so scatter them onto their slot rows;
        the unscheduled / gap rows are sampled harmlessly and dropped.
        """
        logits = tt_out[:, -1, :] if tt_out.dim() == 3 else tt_out
        if is_decode:
            return logits
        full = torch.zeros((total, logits.shape[-1]), dtype=logits.dtype)
        full[torch.as_tensor(scheduled_rows, dtype=torch.long)] = logits[
            : len(scheduled_rows)
        ]
        return full

    def _host_logprobs(
        self, logprobs_tensors: LogprobsTensors | None, scheduled_rows: list[int]
    ) -> LogprobsLists | None:
        if logprobs_tensors is None:
            return None
        rows_t = torch.as_tensor(scheduled_rows, dtype=torch.long)
        return LogprobsTensors(
            logprob_token_ids=logprobs_tensors.logprob_token_ids[rows_t],
            logprobs=logprobs_tensors.logprobs[rows_t],
            selected_token_ranks=logprobs_tensors.selected_token_ranks[rows_t],
        ).tolists()

    def _device_logprobs(
        self,
        tt_log_probs: Any,
        model_input: TTModelInput,
        scheduled_rows: list[int],
        sampled: torch.Tensor,
        is_decode: bool,
    ) -> LogprobsLists | None:
        """Build logprobs for device-sampled tokens, mirroring the gather-DP
        device logprobs path but over the scheduled slot rows."""
        n = len(scheduled_rows)
        assert isinstance(model_input.tt_sampling_params.enable_log_probs, torch.Tensor)
        enable = model_input.tt_sampling_params.enable_log_probs
        sel = (
            torch.as_tensor(scheduled_rows, dtype=torch.long)
            if is_decode
            else torch.arange(n, dtype=torch.long)
        )
        if not enable[sel].any():
            return None
        assert tt_log_probs is not None, "model should return logprobs when requested"
        max_lp = model_input.max_num_logprobs[0] or 0
        next_token_ids = sampled.reshape(n)
        if isinstance(tt_log_probs, tuple):
            top_k_logprobs, top_k_indices = tt_log_probs
            logprobs_tensors = build_logprobs_from_topk(
                top_k_logprobs=top_k_logprobs[sel],
                top_k_indices=top_k_indices[sel],
                sampled_token_ids=next_token_ids,
                max_num_logprobs=max_lp,
            )
        else:
            sampled_log_probs = tt_log_probs.reshape(-1)[sel].reshape(n)
            logprobs_tensors = LogprobsTensors(
                logprob_token_ids=next_token_ids.unsqueeze(-1).to(torch.int32),
                logprobs=sampled_log_probs.unsqueeze(-1).to(torch.float32),
                selected_token_ranks=torch.full((n,), -1, dtype=torch.int32),
            )
        return logprobs_tensors.tolists()

    def _finalize_output(
        self,
        sampled_token_ids: torch.Tensor,
        logprobs: LogprobsLists | None,
        scheduled_rows: list[int],
    ) -> ModelRunnerOutput:
        """Apply sampled tokens to the batch state and build the merged output.

        ``scheduled_rows`` are the persistent rows (== device slots) of the
        requests sampled this step; ``req_ids`` are taken from those rows, in
        order, giving the canonical merged ``req_id_to_index``.
        """
        runner = self.runner
        req_ids = [runner.lane_batch.req_ids[row] for row in scheduled_rows]
        runner._apply_sampled_tokens_to_state(
            sampled_token_ids=sampled_token_ids,
            req_ids=req_ids,
        )
        return runner._build_runner_output(
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs,
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        )
