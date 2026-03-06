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
    - ``1`` forces prefill-only (even if waiting queue is empty).
    - ``None`` uses the default policy: prefill-first, then decode.
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
        want_prefill = bool(self.waiting) and self._forced_mode != 0
        want_decode = not want_prefill and self._forced_mode != 1

        if want_decode:
            if self.waiting:
                # forced_mode=0 but waiting has requests.  Suppress the
                # waiting loop so the base scheduler only runs decode.
                return self._schedule_decode_only()
            # No waiting requests — base scheduler naturally does
            # decode-only (the waiting loop is a no-op).
            return super().schedule()

        if want_prefill:
            return self._schedule_prefill_only()

        # forced_mode=1 but nothing in waiting — return an empty batch.
        # Call super with hidden running so nothing gets scheduled.
        return self._schedule_prefill_only()

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
