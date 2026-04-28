# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum
from typing import cast

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, create_request_queue
from vllm.v1.request import Request

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
    - ``TTSchedulingMode.DEFAULT`` uses the default policy: prefer decode
      while requests are running, but fall back to prefill if decode cannot
      advance.
    """

    waiting: RequestQueue
    running: list[Request]
    max_num_running_reqs: int

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._forced_mode = TTSchedulingMode.DEFAULT

    def set_forced_mode(self, mode: TTSchedulingMode) -> None:
        self._forced_mode = mode

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
        # TT cannot mix prefill and decode in one batch. Once decode is active,
        # prioritize it so new prefills do not create long inter-token gaps.
        if has_running:
            decode_result = self._schedule_decode_only()
            if decode_result.total_num_scheduled_tokens > 0 or not has_waiting:
                return self._finalize_scheduler_output(decode_result)
            # If decode cannot advance, admit prefill work to avoid stalling.
            result = self._schedule_prefill_only()
            return self._finalize_scheduler_output(result)

        if has_waiting:
            prefill_result = self._schedule_prefill_only()
            if prefill_result.total_num_scheduled_tokens == 0:
                result = super().schedule()
                return self._finalize_scheduler_output(result)
            return self._finalize_scheduler_output(prefill_result)

        # No waiting requests in default mode: run decode-only naturally.
        result = super().schedule()
        return self._finalize_scheduler_output(result)

    def _finalize_scheduler_output(
        self, scheduler_output: SchedulerOutput
    ) -> SchedulerOutput:
        if not scheduler_output.pending_structured_output_tokens:
            self.get_grammar_bitmask(scheduler_output)
        return scheduler_output

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        grammar_output = super().get_grammar_bitmask(scheduler_output)
        if grammar_output is not None:
            scheduler_output.structured_output_request_ids = (
                grammar_output.structured_output_request_ids
            )
            scheduler_output.grammar_bitmask = grammar_output.grammar_bitmask
        else:
            scheduler_output.structured_output_request_ids = None
            scheduler_output.grammar_bitmask = None
        return grammar_output

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

