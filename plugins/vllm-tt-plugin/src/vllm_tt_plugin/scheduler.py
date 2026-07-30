# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus
from vllm_tt_plugin.logger import init_tt_logger

logger = init_tt_logger(__name__)

# Side-channel on Request: number of in-flight async output frames to drop
# after TT KV-preemption. Sized exactly to ``num_output_placeholders`` at
# preempt time (not a fixed queue depth): over-discarding drops the
# post-resume re-prefill sample and leaves phantom placeholders.
_TT_ASYNC_DISCARD_ATTR = "_tt_async_frames_to_discard"


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

    def set_forced_mode(self, mode: TTSchedulingMode) -> None:
        self._forced_mode = mode

    def _has_pending_prefill(self) -> bool:
        """Whether any request needs prefill work.

        True when the waiting queue is non-empty or any running request
        is a partial-prefill continuation (``is_prefill_chunk`` set by the
        base scheduler after the previous step).
        """
        return bool(self.waiting) or any(r.is_prefill_chunk for r in self.running)

    def schedule(self) -> SchedulerOutput:
        has_pending_prefill = self._has_pending_prefill()
        has_running = any(not r.is_prefill_chunk for r in self.running)
        mode = self._forced_mode

        if mode == TTSchedulingMode.PREFILL_ONLY:
            result = self._schedule_prefill_only()
            return self._finalize_scheduler_output(result)
        if mode == TTSchedulingMode.DECODE_ONLY:
            if has_pending_prefill:
                # Hide waiting and partial-prefill continuations.
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result)
            # No pending prefill: base scheduler naturally runs decode-only.
            result = super().schedule()
            return self._finalize_scheduler_output(result)

        # Default mode:
        # Prefer prefill whenever there is pending prefill work - either new
        # requests in the waiting queue or partial-prefill continuations in
        # the running list.
        if has_pending_prefill:
            prefill_result = self._schedule_prefill_only()
            # If prefill cannot make progress (e.g., KV pressure) but running
            # decode requests exist, fall back to decode-only so they can
            # advance and free capacity.
            if prefill_result.total_num_scheduled_tokens == 0 and has_running:
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result)
            return self._finalize_scheduler_output(prefill_result)

        # No pending prefill work: run decode-only naturally.
        result = super().schedule()
        return self._finalize_scheduler_output(result)

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
            # Only restore decode requests that were not preempted while hidden
            # is impossible today (they are not in ``running`` for allocate);
            # keep prior behavior: always re-attach the saved decode list.
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
        """Preempt and drop in-flight async frames (TT-specific).

        Upstream keeps placeholders on ordinary preempt because the request is
        not re-scheduled in the same step (vllm#38624). On TT, a preempted
        request is immediately eligible for full re-prefill while batch-queued
        / async-decode frames are still outstanding, so those frames must be
        discarded or they race with the re-prefill sample token.
        """
        pending = int(getattr(request, "num_output_placeholders", 0) or 0)
        super()._preempt_request(request, timestamp)
        if not self.scheduler_config.async_scheduling:
            return
        # At least one frame may still be in the batch queue even when the
        # placeholder counter already drained to 0 (output applied, execute
        # not yet finalized). Cap at tracked placeholders when known, else 1.
        pending = max(pending, 1)
        request.num_output_placeholders = 0
        # Frame counter only (do not also set discard_latest_async_tokens).
        setattr(
            request,
            _TT_ASYNC_DISCARD_ATTR,
            int(getattr(request, _TT_ASYNC_DISCARD_ATTR, 0) or 0) + pending,
        )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        """Advance computed tokens, then account async placeholders.

        Base vLLM sets
        ``is_prefill_chunk = computed < num_tokens + num_output_placeholders``.
        After KV-preempt + re-prefill, leftover placeholders keep that flag
        True even when the prompt is fully computed, so AsyncScheduler skips
        the placeholder ``+1`` while the TT chunked-prefill path still returns
        a sample token → ``num_output_placeholders`` underflow (same class as
        upstream vllm#35755). Classify prefill chunks by prompt progress only.
        """
        # Grandparent advances num_computed_tokens / encoder bookkeeping.
        Scheduler._update_after_schedule(self, scheduler_output)

        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            request.is_prefill_chunk = request.num_computed_tokens < request.num_tokens

        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            request.spec_token_ids = self._spec_token_placeholders

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        left = int(getattr(request, _TT_ASYNC_DISCARD_ATTR, 0) or 0)
        if left > 0:
            setattr(request, _TT_ASYNC_DISCARD_ATTR, left - 1)
            return [], False
        if request.discard_latest_async_tokens:
            request.discard_latest_async_tokens = False
            return [], False
        # Stale async frame while the request sits preempted in the waiting
        # queue (before re-prefill makes it RUNNING again).
        if request.status == RequestStatus.PREEMPTED:
            request.num_output_placeholders = 0
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = Scheduler._update_request_with_output(
            self, request, new_token_ids
        )

        request.num_output_placeholders -= len(new_token_ids)
        if request.num_output_placeholders < 0:
            # Last-resort clamp: must not kill EngineCore under residual races.
            logger.error(
                "TTScheduler: clamping num_output_placeholders=%s for req=%s",
                request.num_output_placeholders,
                request.request_id,
            )
            request.num_output_placeholders = 0

        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
