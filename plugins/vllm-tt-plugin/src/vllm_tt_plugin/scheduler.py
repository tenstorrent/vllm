# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from typing import cast

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


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
    - No chunked prefill: each prefill must be scheduled in full.

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
      when waiting is non-empty, but fall back to decode-only if prefill
      cannot admit any request and running decode requests exist.
    """

    waiting: RequestQueue
    running: list[Request]
    max_num_running_reqs: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_mode = TTSchedulingMode.DEFAULT

    def set_forced_mode(self, mode: TTSchedulingMode) -> None:
        self._forced_mode = mode

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        """#47488 scheduler half: generalize the 1-token async accounting to
        block-granular commits (block-diffusion models such as DiffusionGemma).

        AsyncScheduler models an autoregressive 1-in/1-out decode step: it
        reserves exactly one output placeholder per scheduled step and asserts
        ``num_output_placeholders >= 0`` after subtracting the committed count.
        A block-diffusion decode step commits a whole canvas of ``n`` tokens
        (n == ``canvas_length``, e.g. 256) in ONE model step whose K/V is written
        inside the model, so more than one committed token survives the stop-trim
        and AsyncScheduler underflows (``async_scheduler.py:53``).

        This override keeps the same intent while allowing ``n > 1``:
        * clamp the placeholder budget at 0 instead of asserting;
        * advance ``num_computed_tokens`` by the extra ``n - 1`` committed tokens
          so it keeps lagging the committed output by exactly one position — the
          invariant the running-loop scheduler math relies on to schedule the
          next decode step (``num_new_tokens == 1``);
        * skip the prefix-cache bookkeeping for block commits (prefix caching is
          force-disabled on this sliding-window / block-diffusion path and the
          committed block's slots are model-owned, not vLLM-paged).

        For ``n == 1`` (every autoregressive model, and DiffusionGemma requests
        that stop at the block's first token) this is byte-identical to
        ``AsyncScheduler._update_request_with_output``.
        """
        if request.discard_latest_async_tokens:
            # Force-preempted in reset_prefix_cache: discard the async token.
            request.discard_latest_async_tokens = False
            return [], False

        status_before_update = request.status
        # Grandparent (base Scheduler) does the append + stop-trim; we replace
        # AsyncScheduler's fixed 1-token placeholder/num_computed accounting.
        new_token_ids, stopped = Scheduler._update_request_with_output(
            self, request, new_token_ids
        )
        n = len(new_token_ids)

        request.num_output_placeholders -= n
        if request.num_output_placeholders < 0:
            # Block committed more tokens than the single reserved placeholder.
            request.num_output_placeholders = 0
        if n > 1:
            # _update_after_schedule already advanced num_computed by the one
            # scheduled input position; add the remaining committed tokens so
            # num_computed catches up to (num_tokens - 1).
            request.num_computed_tokens += n - 1

        if status_before_update == RequestStatus.RUNNING and n == 1:
            self.kv_cache_manager.cache_blocks(
                request,
                request.num_computed_tokens - request.num_output_placeholders,
            )
        return new_token_ids, stopped

    def schedule(self) -> SchedulerOutput:
        has_waiting = bool(self.waiting)
        has_running = bool(self.running)
        mode = self._forced_mode

        if mode == TTSchedulingMode.PREFILL_ONLY:
            # If waiting is empty, this intentionally returns an empty batch.
            result = self._schedule_prefill_only()
            return self._finalize_scheduler_output(result)
        if mode == TTSchedulingMode.DECODE_ONLY:
            if has_waiting:
                # Hide waiting so base scheduler cannot admit prefill.
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result)
            # No waiting requests: base scheduler naturally runs decode-only.
            result = super().schedule()
            return self._finalize_scheduler_output(result)

        # Default mode:
        # Prefer prefill whenever waiting is non-empty to admit new requests.
        if has_waiting:
            prefill_result = self._schedule_prefill_only()
            # If waiting is non-empty but prefill cannot be admitted (e.g. KV
            # pressure and no chunked prefill), do not stall decode progress.
            # Fall back to decode-only so running requests can advance and free
            # capacity for later full-prefill admission.
            if prefill_result.total_num_scheduled_tokens == 0 and has_running:
                result = self._schedule_decode_only()
                return self._finalize_scheduler_output(result)
            return self._finalize_scheduler_output(prefill_result)

        # No waiting requests in default mode: run decode-only naturally.
        result = super().schedule()
        return self._finalize_scheduler_output(result)

    def _finalize_scheduler_output(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        return scheduler_output

    def _schedule_prefill_only(self) -> SchedulerOutput:
        """Schedule only waiting (prefill) requests.

        Temporarily hides the running (decode) requests so the base
        scheduler's running loop iterates zero times and only the
        waiting loop executes.  Adjusts max_num_running_reqs so the
        waiting loop respects the true capacity.
        """
        saved_running = self.running
        saved_max = self.max_num_running_reqs
        self.running = cast(list[Request], [])
        self.max_num_running_reqs = max(0, saved_max - len(saved_running))
        try:
            result = super().schedule()
        finally:
            self.running = saved_running + self.running
            self.max_num_running_reqs = saved_max
        return result

    def _schedule_decode_only(self) -> SchedulerOutput:
        """Schedule only running (decode) requests.

        Temporarily hides the waiting queue so the base scheduler's
        waiting loop is a no-op.  Any requests that get preempted
        during decode scheduling are merged back into the original
        waiting queue afterwards.
        """
        saved_waiting = self.waiting
        self.waiting = create_request_queue(self.policy)
        try:
            result = super().schedule()
        finally:
            if self.waiting:
                saved_waiting.prepend_requests(self.waiting)
            self.waiting = saved_waiting
        return result
