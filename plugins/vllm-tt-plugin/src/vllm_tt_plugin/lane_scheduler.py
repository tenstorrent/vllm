# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.request import Request
from vllm_tt_plugin.config import get_tt_data_parallel_size
from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


@dataclass
class LaneStepMetadata:
    """Per-step lane bookkeeping for runner merge/split."""

    lane_outputs: list[SchedulerOutput]
    batch_size_per_dp: list[int]
    is_decode: bool
    lane_req_ids: list[list[str]]
    lane_req_id_to_index: list[dict[str, int]]


def merge_lane_scheduler_outputs(
    lane_outputs: list[SchedulerOutput],
) -> SchedulerOutput:
    """Merge per-lane scheduler outputs into one engine-facing output."""
    if not lane_outputs:
        return SchedulerOutput.make_empty()

    if not any(out.total_num_scheduled_tokens > 0 for out in lane_outputs):
        finished: set[str] = set()
        free_encoder: list[str] = []
        for out in lane_outputs:
            finished |= out.finished_req_ids
            free_encoder.extend(out.free_encoder_mm_hashes)
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=finished,
            free_encoder_mm_hashes=free_encoder,
        )

    scheduled_new_reqs: list = []
    cached = CachedRequestData.make_empty()
    num_scheduled_tokens: dict[str, int] = {}
    scheduled_spec_decode_tokens: dict[str, list[int]] = {}
    scheduled_encoder_inputs: dict[str, list[int]] = {}
    num_common_prefix_blocks: list[int] = []
    finished_req_ids: set[str] = set()
    free_encoder_mm_hashes: list[str] = []
    has_structured_output_requests = False
    pending_structured_output_tokens = False
    num_invalid_spec_tokens: dict[str, int] | None = None

    for out in lane_outputs:
        scheduled_new_reqs.extend(out.scheduled_new_reqs)
        lane_cached = out.scheduled_cached_reqs
        if lane_cached.num_reqs > 0:
            cached.req_ids.extend(lane_cached.req_ids)
            cached.resumed_req_ids |= lane_cached.resumed_req_ids
            cached.new_token_ids.extend(lane_cached.new_token_ids)
            cached.all_token_ids.update(lane_cached.all_token_ids)
            cached.new_block_ids.extend(lane_cached.new_block_ids)
            cached.num_computed_tokens.extend(lane_cached.num_computed_tokens)
            cached.num_output_tokens.extend(lane_cached.num_output_tokens)
        num_scheduled_tokens.update(out.num_scheduled_tokens)
        scheduled_spec_decode_tokens.update(out.scheduled_spec_decode_tokens)
        scheduled_encoder_inputs.update(out.scheduled_encoder_inputs)
        if out.num_common_prefix_blocks:
            if not num_common_prefix_blocks:
                num_common_prefix_blocks = list(out.num_common_prefix_blocks)
            else:
                num_common_prefix_blocks = [
                    max(a, b)
                    for a, b in zip(
                        num_common_prefix_blocks,
                        out.num_common_prefix_blocks,
                        strict=False,
                    )
                ]
        finished_req_ids |= out.finished_req_ids
        free_encoder_mm_hashes.extend(out.free_encoder_mm_hashes)
        has_structured_output_requests |= out.has_structured_output_requests
        pending_structured_output_tokens |= out.pending_structured_output_tokens
        if out.num_invalid_spec_tokens:
            if num_invalid_spec_tokens is None:
                num_invalid_spec_tokens = {}
            num_invalid_spec_tokens.update(out.num_invalid_spec_tokens)

    total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=cached,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        scheduled_encoder_inputs=scheduled_encoder_inputs,
        num_common_prefix_blocks=num_common_prefix_blocks,
        finished_req_ids=finished_req_ids,
        free_encoder_mm_hashes=free_encoder_mm_hashes,
        has_structured_output_requests=has_structured_output_requests,
        pending_structured_output_tokens=pending_structured_output_tokens,
        num_invalid_spec_tokens=num_invalid_spec_tokens,
    )


def filter_scheduler_output_for_lane(
    scheduler_output: SchedulerOutput | None,
    lane: int,
    requests: dict[str, Request],
) -> SchedulerOutput | None:
    """Return a lane-local view of ``scheduler_output`` (or None if empty)."""
    if scheduler_output is None:
        return None

    def _lane_req(req_id: str) -> bool:
        req = requests.get(req_id)
        return req is not None and req.tt_lane == lane

    scheduled_new_reqs = [
        nr for nr in scheduler_output.scheduled_new_reqs if _lane_req(nr.req_id)
    ]
    cached = scheduler_output.scheduled_cached_reqs
    lane_req_ids = [rid for rid in cached.req_ids if _lane_req(rid)]
    if not scheduled_new_reqs and not lane_req_ids:
        if scheduler_output.total_num_scheduled_tokens == 0:
            return scheduler_output
        return None

    indices = [i for i, rid in enumerate(cached.req_ids) if _lane_req(rid)]
    lane_cached = CachedRequestData(
        req_ids=[cached.req_ids[i] for i in indices],
        resumed_req_ids={rid for rid in cached.resumed_req_ids if _lane_req(rid)},
        new_token_ids=[cached.new_token_ids[i] for i in indices],
        all_token_ids={
            rid: toks for rid, toks in cached.all_token_ids.items() if _lane_req(rid)
        },
        new_block_ids=[cached.new_block_ids[i] for i in indices],
        num_computed_tokens=[cached.num_computed_tokens[i] for i in indices],
        num_output_tokens=[cached.num_output_tokens[i] for i in indices],
    )
    num_scheduled_tokens = {
        rid: n
        for rid, n in scheduler_output.num_scheduled_tokens.items()
        if _lane_req(rid)
    }
    return SchedulerOutput(
        scheduled_new_reqs=scheduled_new_reqs,
        scheduled_cached_reqs=lane_cached,
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
        scheduled_spec_decode_tokens={
            rid: toks
            for rid, toks in scheduler_output.scheduled_spec_decode_tokens.items()
            if _lane_req(rid)
        },
        scheduled_encoder_inputs={
            rid: idxs
            for rid, idxs in scheduler_output.scheduled_encoder_inputs.items()
            if _lane_req(rid)
        },
        num_common_prefix_blocks=scheduler_output.num_common_prefix_blocks,
        finished_req_ids={
            rid for rid in scheduler_output.finished_req_ids if _lane_req(rid)
        },
        free_encoder_mm_hashes=scheduler_output.free_encoder_mm_hashes,
        has_structured_output_requests=scheduler_output.has_structured_output_requests,
        pending_structured_output_tokens=(
            scheduler_output.pending_structured_output_tokens
        ),
        num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
    )


class TTLaneCoordinator(TTScheduler):
    """Single-process multi-lane scheduler for TT gathered-batch execution."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_lanes = get_tt_data_parallel_size(self.vllm_config)
        self._per_lane_max = self.scheduler_config.max_num_seqs
        self._last_lane_metadata: LaneStepMetadata | None = None

    def _refresh_lane_counts(self) -> None:
        waiting_counts = [0] * self.num_lanes
        running_counts = [0] * self.num_lanes
        for req in self.waiting:
            if 0 <= req.tt_lane < self.num_lanes:
                waiting_counts[req.tt_lane] += 1
        for req in self.running:
            if 0 <= req.tt_lane < self.num_lanes:
                running_counts[req.tt_lane] += 1
        self._lane_waiting_counts = waiting_counts
        self._lane_running_counts = running_counts

    def _pick_lane(self) -> int:
        self._refresh_lane_counts()
        best_lane = 0
        best_score = None
        for lane in range(self.num_lanes):
            score = (
                self._lane_waiting_counts[lane] * 4 + self._lane_running_counts[lane]
            )
            if best_score is None or score < best_score:
                best_score = score
                best_lane = lane
        return best_lane

    def add_request(self, request: Request) -> None:
        if request.preferred_data_parallel_rank is not None:
            request.tt_lane = request.preferred_data_parallel_rank % self.num_lanes
        elif request.tt_lane < 0:
            request.tt_lane = self._pick_lane()
        super().add_request(request)

    def _lane_has_work(self, lane: int) -> bool:
        return (
            self._lane_waiting_counts[lane] > 0 or self._lane_running_counts[lane] > 0
        )

    def _local_prefill_intent(self, lane: int) -> int:
        has_waiting = self._lane_waiting_counts[lane] > 0
        has_running = self._lane_running_counts[lane] > 0
        has_capacity = self._lane_running_counts[lane] < self._per_lane_max
        return int(has_waiting and ((not has_running) or has_capacity))

    def _negotiate_forced_mode(self) -> TTSchedulingMode:
        intent = max(self._local_prefill_intent(lane) for lane in range(self.num_lanes))
        return TTSchedulingMode.from_prefill_intent(intent)

    @contextmanager
    def _visible_lane_only(self, lane: int) -> Iterator[None]:
        stashed_waiting = create_request_queue(self.policy)
        for req in list(self.waiting):
            if req.tt_lane != lane:
                self.waiting.remove_request(req)
                stashed_waiting.add_request(req)

        saved_running = self.running
        self.running = [r for r in saved_running if r.tt_lane == lane]
        stashed_running = [r for r in saved_running if r.tt_lane != lane]

        saved_max = self.max_num_running_reqs
        self.max_num_running_reqs = self._per_lane_max
        try:
            yield
        finally:
            self.running = self.running + stashed_running
            self.max_num_running_reqs = saved_max
            if stashed_waiting:
                self.waiting.prepend_requests(stashed_waiting)

    def schedule(self) -> SchedulerOutput:
        self._refresh_lane_counts()
        if not any(self._lane_has_work(lane) for lane in range(self.num_lanes)):
            self._last_lane_metadata = None
            # Preserve the base scheduler's "finished requests only" cleanup
            # step so the model runner can clear stale request state after the
            # last active lane drains.
            return super().schedule()

        forced_mode = self._negotiate_forced_mode()
        lane_outputs: list[SchedulerOutput] = []
        batch_size_per_dp: list[int] = []

        for lane in range(self.num_lanes):
            if not self._lane_has_work(lane):
                lane_outputs.append(SchedulerOutput.make_empty())
                batch_size_per_dp.append(0)
                continue
            self.set_forced_mode(forced_mode)
            with self._visible_lane_only(lane):
                lane_out = super().schedule()
            self.set_forced_mode(TTSchedulingMode.DEFAULT)
            lane_outputs.append(lane_out)
            batch_size_per_dp.append(len(lane_out.num_scheduled_tokens))

        merged = merge_lane_scheduler_outputs(lane_outputs)
        is_decode = forced_mode == TTSchedulingMode.DECODE_ONLY
        self._last_lane_metadata = LaneStepMetadata(
            lane_outputs=lane_outputs,
            batch_size_per_dp=batch_size_per_dp,
            is_decode=is_decode,
            lane_req_ids=[
                list(out.num_scheduled_tokens.keys()) for out in lane_outputs
            ],
            lane_req_id_to_index=[
                {rid: idx for idx, rid in enumerate(out.num_scheduled_tokens)}
                for out in lane_outputs
            ],
        )
        merged._tt_lane_step_metadata = self._last_lane_metadata
        self._refresh_lane_counts()
        return merged

    def get_last_lane_metadata(self) -> LaneStepMetadata | None:
        return self._last_lane_metadata
