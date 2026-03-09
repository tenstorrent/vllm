# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request

logger = init_logger(__name__)


class TTScheduler(AsyncScheduler):
    """Scheduler for the TT (Tenstorrent) platform.

    TT constraints:
    - No mixed prefill+decode batches: each batch is either all-prefill
      or all-decode.
    - No chunked prefill: each prefill must be scheduled in full.

    Inherits from AsyncScheduler to get num_output_placeholders support,
    which allows decode requests to be re-scheduled before
    update_from_output processes the previous step's results.  This is
    essential for overlapping CPU scheduling with device execution.

    Supports ``set_forced_mode`` for DP-gather coordination:
    - ``0`` forces decode-only (even if waiting queue is non-empty).
    - ``1`` forces prefill-only (and may return an empty batch when waiting
      is empty).
    - ``None`` uses the default policy: prefer prefill when waiting is
      non-empty, but fall back to decode-only if prefill cannot admit any
      request and running decode requests exist.
    """

    waiting: RequestQueue
    running: list[Request]
    max_num_running_reqs: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_mode: int | None = None

    def set_forced_mode(self, mode: int | None) -> None:
        assert mode in (None, 0, 1)
        self._forced_mode = mode

    def schedule(self) -> SchedulerOutput:
        has_waiting = bool(self.waiting)
        has_running = bool(self.running)
        mode = self._forced_mode

        # Forced mode from DP-gather coordination:
        #   mode == 1 -> prefill-only
        #   mode == 0 -> decode-only
        if mode == 1:
            # If waiting is empty, this intentionally returns an empty batch.
            return self._schedule_prefill_only()
        if mode == 0:
            if has_waiting:
                # Hide waiting so base scheduler cannot admit prefill.
                return self._schedule_decode_only()
            # No waiting requests: base scheduler naturally runs decode-only.
            return super().schedule()

        # Default mode (mode is None):
        # Prefer prefill whenever waiting is non-empty to admit new requests.
        if has_waiting:
            prefill_result = self._schedule_prefill_only()
            has_prefill_running = any(
                req.num_computed_tokens < req.num_prompt_tokens for req in self.running
            )
            # If waiting is non-empty but prefill cannot be admitted (e.g. KV
            # pressure and no chunked prefill), do not stall decode progress.
            # Fall back to decode-only so running requests can advance and free
            # capacity for later full-prefill admission.
            #
            # Guard: only apply this when running requests are already in
            # decode phase. If any running request is still in prefill phase,
            # forcing decode-only here can create scheduler/model-output
            # mismatches in async TT flow.
            if (
                prefill_result.total_num_scheduled_tokens == 0
                and has_running
                and bool(self.waiting)
                and not has_prefill_running
            ):
                return self._schedule_decode_only()
            return prefill_result

        # No waiting requests in default mode: run decode-only naturally.
        return super().schedule()

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
