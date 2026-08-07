# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
from vllm_tt_plugin.config import get_tt_output_tokens_per_step
from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)


@dataclass
class _PendingOutputs:
    """Counts a request's model outputs that are still in the pipeline.

    Async scheduling starts new decode steps before the tokens from earlier
    steps have come back. If a request is preempted, it has to redo its prefill
    from scratch, and outputs still in the pipeline must be discarded instead
    of being added to the request. One output can be one AR token or one
    multi-token physical canvas.

    ``outstanding`` counts outputs that were scheduled but have not come back
    yet. ``stale`` counts how many of those we have decided to throw away.
    """

    PENDING_ATTR: ClassVar[str] = "_tt_pending_outputs"

    outstanding: int = 0
    stale: int = 0

    def record(self) -> None:
        """Count one decode step; its model output returns later.

        A step counts once even if it speculates several tokens, because it
        still sends back a single output. A block model likewise counts one
        complete canvas as one output.
        """
        self.outstanding += 1

    def discard_outstanding(self) -> None:
        """On preempt, mark every in-flight output to be thrown away."""
        self.stale = self.outstanding

    def is_next_stale(self) -> bool:
        """Take the next returned output; True means throw it away.

        Outputs return in the order they were scheduled, so the stale ones
        always arrive before any fresh output from after the preempt.
        """
        if not self.outstanding:
            return False
        self.outstanding -= 1
        if not self.stale:
            return False
        self.stale -= 1
        return True

    @classmethod
    def for_request(cls, request: Request) -> "_PendingOutputs":
        pending = getattr(request, cls.PENDING_ATTR, None)
        if pending is None:
            pending = cls()
            setattr(request, cls.PENDING_ATTR, pending)
        return pending


class TTSchedulingMode(Enum):
    DEFAULT = "default"
    DECODE_ONLY = "decode_only"
    PREFILL_ONLY = "prefill_only"

    @classmethod
    def from_prefill_intent(cls, prefill_intent: int) -> "TTSchedulingMode":
        if prefill_intent == 0:
            return cls.DECODE_ONLY
        if prefill_intent == 1:
            return cls.PREFILL_ONLY
        raise ValueError(f"Invalid TT scheduling intent: {prefill_intent}")


class TTScheduler(AsyncScheduler):
    """Scheduler for the TT (Tenstorrent) platform.

    TT constraints:
    - No mixed prefill+decode batches: each batch is either all-prefill
      or all-decode.
    - Token-chunked prefill: a long prefill may be split across scheduler
      steps.  After a partial chunk the request moves to ``running`` with
      ``is_prefill_chunk=True``; subsequent steps schedule the next chunk
      until the prompt is fully computed.

    Inherits from AsyncScheduler to get num_output_placeholders support.
    TT uses this scheduler in both sync and async execution modes:
    - with async_scheduling=False, it behaves as the single TT scheduler
      without execution overlap
    - with async_scheduling=True, placeholders allow decode requests to be
      re-scheduled before update_from_output processes the previous step's
      results, enabling host/device overlap
    - under async_scheduling, preemption invalidates that request's
      scheduled-but-unreturned outputs (see ``_preempt_request``)

    Supports ``set_forced_mode`` for DP-gather coordination:
    - ``TTSchedulingMode.DECODE_ONLY`` forces decode-only (even if waiting
      queue is non-empty).
    - ``TTSchedulingMode.PREFILL_ONLY`` forces prefill-only (and may return an
      empty batch when waiting is empty).
    - ``TTSchedulingMode.DEFAULT`` uses the default policy: prefer prefill
      when pending prefill work exists (waiting queue or partial-prefill
      continuations), falling back to decode-only when prefill cannot make
      progress and running decode requests exist.
    """

    waiting: RequestQueue
    running: list[Request]
    max_num_running_reqs: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_mode = TTSchedulingMode.DEFAULT
        self._output_tokens_per_step = get_tt_output_tokens_per_step(self.vllm_config)
        self._is_block_output_model = self._output_tokens_per_step > 1
        # Autoregressive models retain the existing unconditional cache call.
        # Multi-token models cache only when they opt into vLLM APC; a
        # model-owned KV cache must not be recorded as a vLLM paged prefix.
        self._cache_block_outputs = (
            not self._is_block_output_model
            or self.vllm_config.cache_config.enable_prefix_caching
        )

    def set_forced_mode(self, mode: TTSchedulingMode) -> None:
        self._forced_mode = mode

    def _has_pending_prefill(self) -> bool:
        """Whether any request needs prefill work.

        True when the waiting queue is non-empty or any running request
        is a partial-prefill continuation (``is_prefill_chunk`` set by the
        base scheduler after the previous step).
        """
        return bool(self.waiting) or any(r.is_prefill_chunk for r in self.running)

    def _has_capacity_for_output_step(self, request: Request) -> bool:
        """Whether another physical model output can fit after in-flight work."""
        return (
            not self._is_block_output_model
            or request.num_tokens
            + request.num_output_placeholders
            + self._output_tokens_per_step
            <= self.max_model_len
        )

    def _finish_impossible_waiting_requests(self) -> None:
        """Reject bypassed admissions that cannot fit their first canvas."""
        if not self._is_block_output_model:
            return
        impossible_req_ids = [
            request.request_id
            for request in self.waiting
            if not self._has_capacity_for_output_step(request)
        ]
        if impossible_req_ids:
            logger.error(
                "Finishing block-output requests that bypassed frontend "
                "capacity validation and cannot fit a physical output: %s",
                impossible_req_ids,
            )
            self.finish_requests(
                impossible_req_ids, RequestStatus.FINISHED_LENGTH_CAPPED
            )

    def schedule(self) -> SchedulerOutput:
        self._finish_impossible_waiting_requests()

        # Async scheduling can have one or more complete physical outputs
        # reserved in ``num_output_placeholders``. Hide a running request when
        # another complete output would exceed capacity; its oldest in-flight
        # output is still dispatched and reconciled normally.
        deferred_running = [
            request
            for request in self.running
            if not self._has_capacity_for_output_step(request)
        ]
        if deferred_running:
            deferred_req_ids = {request.request_id for request in deferred_running}
            self.running = [
                request
                for request in self.running
                if request.request_id not in deferred_req_ids
            ]

        has_pending_prefill = self._has_pending_prefill()
        has_running = any(not r.is_prefill_chunk for r in self.running)
        mode = self._forced_mode
        result = None
        try:
            if mode == TTSchedulingMode.PREFILL_ONLY:
                result = self._schedule_prefill_only()
            elif mode == TTSchedulingMode.DECODE_ONLY:
                if has_pending_prefill:
                    # Hide waiting and partial-prefill continuations.
                    result = self._schedule_decode_only()
                else:
                    # No pending prefill: base scheduler naturally runs decode-only.
                    result = super().schedule()
            elif has_pending_prefill:
                # Default mode prefers prefill whenever there is pending
                # prefill work, falling back to decode when prefill cannot
                # make progress.
                result = self._schedule_prefill_only()
                if result.total_num_scheduled_tokens == 0 and has_running:
                    result = self._schedule_decode_only()
            else:
                # No pending prefill work: run decode-only naturally.
                result = super().schedule()
            return self._finalize_scheduler_output(result)
        finally:
            if deferred_running:
                active_req_ids = {request.request_id for request in self.running}
                self.running.extend(
                    request
                    for request in deferred_running
                    if request.request_id in self.requests
                    and not request.is_finished()
                    and request.request_id not in active_req_ids
                )

    def _finalize_scheduler_output(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        return scheduler_output

    def _schedule_prefill_only(self) -> SchedulerOutput:
        """Schedule prefill work: waiting requests + partial-prefill continuations.

        Hides running decode requests (``is_prefill_chunk=False``) so the base
        scheduler's running loop only processes partial-prefill continuations and
        the waiting loop admits new prefills.  Adjusts ``max_num_running_reqs`` to
        account for hidden decode slots.
        """
        pure_decodes = [r for r in self.running if not r.is_prefill_chunk]
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_max = self.max_num_running_reqs
        self.running = partial_prefills
        self.max_num_running_reqs = max(0, saved_max - len(pure_decodes))
        try:
            result = super().schedule()
        finally:
            self.running.extend(pure_decodes)
            self.max_num_running_reqs = saved_max
        return result

    def _schedule_decode_only(self) -> SchedulerOutput:
        """Schedule only running decode requests.

        Hides the waiting queue **and** any partial-prefill continuations
        so the base scheduler only sees decode-phase requests.  Preempted
        requests are merged back into the original waiting queue afterwards.
        """
        partial_prefills = [r for r in self.running if r.is_prefill_chunk]

        saved_waiting = self.waiting
        self.waiting = create_request_queue(self.policy)
        if partial_prefills:
            self.running = [r for r in self.running if not r.is_prefill_chunk]

        try:
            result = super().schedule()
        finally:
            if self.waiting:
                saved_waiting.prepend_requests(self.waiting)
            self.waiting = saved_waiting
            if partial_prefills:
                self.running.extend(partial_prefills)

        return result

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and drop any outputs still in the pipeline.

        The base class frees the request's KV cache and queues it to redo its
        prefill. Under async scheduling, the outputs it already scheduled are
        still on their way back; those were built on the now-freed cache, so we
        mark them to be thrown away and reset the placeholder count so the base
        scheduler treats the request as a fresh prefill. The reset is also
        required in synchronous mode because this class deliberately inherits
        AsyncScheduler's placeholder reservation in both execution modes.
        """
        super()._preempt_request(request, timestamp)

        if self.scheduler_config.async_scheduling:
            _PendingOutputs.for_request(request).discard_outstanding()
        request.num_output_placeholders = 0

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Advance scheduled inputs and reserve complete model outputs.

        ``AsyncScheduler`` first reserves one output placeholder. Block-output
        models then reserve the remaining physical positions and advance
        ``num_computed_tokens`` by the same remainder before dispatch. Under
        async execution, ``_PendingOutputs.record`` also remembers each
        scheduled-but-unreturned model output so a later preemption can discard
        every stale result built from the released cache.
        """
        super()._update_after_schedule(scheduler_output)

        extra_output_tokens = self._output_tokens_per_step - 1
        if extra_output_tokens:
            for req_id in scheduler_output.num_scheduled_tokens:
                request = self.requests[req_id]
                if request.is_prefill_chunk:
                    continue
                request.num_output_placeholders += extra_output_tokens
                request.num_computed_tokens += extra_output_tokens

        if self.scheduler_config.async_scheduling:
            for req_id in scheduler_output.num_scheduled_tokens:
                request = self.requests[req_id]
                if not request.is_prefill_chunk:
                    _PendingOutputs.for_request(request).record()

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        """Apply one returned output and reconcile its physical reservation.

        Autoregressive models delegate unchanged to ``AsyncScheduler``. For a
        block model, call the grandparent ``Scheduler`` implementation to append
        and trim the whole returned block, then manually mirror
        ``AsyncScheduler._update_request_with_output`` by consuming output
        placeholders and optionally caching blocks. This bypass is deliberate:
        AsyncScheduler assumes one placeholder per returned token and cannot
        reconcile a physically reserved block that was logically EOS/max-token
        trimmed. Future changes to that upstream override must be reflected
        here.
        """
        if (
            self.scheduler_config.async_scheduling
            and _PendingOutputs.for_request(request).is_next_stale()
        ):
            request.discard_latest_async_tokens = False
            return [], False

        if self._output_tokens_per_step == 1:
            return super()._update_request_with_output(request, new_token_ids)

        if len(new_token_ids) > self._output_tokens_per_step:
            raise ValueError(
                "Model returned more tokens than its output_tokens_per_step "
                f"contract: {len(new_token_ids)} > {self._output_tokens_per_step}"
            )

        if request.discard_latest_async_tokens:
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        # The base scheduler appends tokens and applies EOS/max-token trimming.
        # Reconcile against the surviving tokens, not the raw canvas.
        new_token_ids, stopped = Scheduler._update_request_with_output(
            self, request, new_token_ids
        )
        if (
            not stopped
            and request.num_tokens + self._output_tokens_per_step > self.max_model_len
        ):
            # End after the current physical output when no subsequent complete
            # output can fit. ``schedule`` independently gates both waiting and
            # running requests before model dispatch.
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            stopped = True
        unused_output_tokens = self._output_tokens_per_step - len(new_token_ids)
        request.num_computed_tokens -= unused_output_tokens
        request.num_output_placeholders -= unused_output_tokens
        request.num_output_placeholders -= len(new_token_ids)
        if request.num_output_placeholders < 0:
            raise RuntimeError(
                "Output placeholders underflowed after block-output reconciliation"
            )

        if status_before_update == RequestStatus.RUNNING and self._cache_block_outputs:
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens - request.num_output_placeholders,
            )
        return new_token_ids, stopped
