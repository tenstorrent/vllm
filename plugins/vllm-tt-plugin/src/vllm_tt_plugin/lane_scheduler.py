# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-process, multi-lane scheduling for TT data-parallel execution.

TT Llama3 70B Galaxy model is single-weights and single-execute, but
keeps 4 data-parallel (DP) KV caches. In this layout,
one engine process drives all replicas: each replica is a
"lane" with its own slice of requests, but every step the lanes must execute in
lockstep against a single gathered batch on device.

This module bridges vLLM's single-queue scheduler to that layout:

- ``TTLaneCoordinator`` owns one independent :class:`TTScheduler` per lane (each
  with its own KV cache manager and request queues) and stitches the per-lane
  results into one engine-facing ``SchedulerOutput``. Because each lane drives a
  physically separate DP submesh KV cache, the per-lane schedulers allocate
  block IDs independently: block IDs repeat across lanes, which is correct since
  the model runner routes each lane's block table to its own submesh.
- ``merge_lane_scheduler_outputs`` performs that stitching.
- ``LaneStepMetadata`` rides along on the merged output so the model runner can
  later split the gathered batch back into per-lane pieces.

Because the device executes all lanes together, every lane in a step must agree
on a single scheduling mode (all-prefill or all-decode); the coordinator
negotiates that mode before scheduling any lane.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    SchedulerOutput,
)
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats
from vllm_tt_plugin.config import (
    get_tt_data_parallel_size,
    get_tt_per_lane_max_num_seqs,
)
from vllm_tt_plugin.scheduler import TTScheduler, TTSchedulingMode

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
    from vllm.v1.request import Request, RequestStatus
    from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


@dataclass
class LaneStepMetadata:
    """Per-step lane bookkeeping for the runner's merge/split.

    The coordinator merges every lane's work into one ``SchedulerOutput``, which
    erases the lane boundaries. This metadata records what those boundaries were
    so the model runner can scatter the gathered batch onto the right DP
    replicas and gather the results back, indexed by lane.

    The coordinator only fills in ``lane_outputs`` and ``is_decode`` — the
    minimal contract the runner needs. The runner derives the per-lane request
    ordering itself while preparing inputs and writes ``lane_req_ids`` /
    ``lane_req_id_to_index`` back onto its own copy; those are the authoritative
    values used to scatter device outputs back per request.
    """

    # The raw, unmerged output for each lane (empty for idle lanes), in lane
    # order. Lets the runner reconstruct per-lane inputs after the merge.
    lane_outputs: list[SchedulerOutput]
    # True if this is a decode-only step, False if prefill-only. Lanes never mix
    # the two within a step (a TT device constraint).
    is_decode: bool
    # Request IDs scheduled on each lane, in the order they appear in the batch,
    # plus the reverse lookup (request ID -> position within the lane). Left
    # empty by the coordinator; populated by the runner after it prepares the
    # per-lane inputs.
    lane_req_ids: list[list[str]] = field(default_factory=list)
    lane_req_id_to_index: list[dict[str, int]] = field(default_factory=list)


def merge_lane_scheduler_outputs(
    lane_outputs: list[SchedulerOutput],
) -> SchedulerOutput:
    """Merge per-lane scheduler outputs into one engine-facing output.

    Each field is combined according to its kind: list fields are concatenated
    in lane order, per-request dicts are merged (request IDs are globally
    unique, so keys never collide), set fields are unioned, and booleans are
    OR-ed. ``num_common_prefix_blocks`` is the elementwise max across lanes,
    since the gathered batch must reserve the largest common prefix any lane
    needs.

    Block IDs carried in ``scheduled_cached_reqs`` / ``scheduled_new_reqs`` are
    lane-local and may repeat across lanes; this is intentional — each lane
    indexes its own submesh KV cache — so they are concatenated verbatim
    without any global remapping.
    """
    if not lane_outputs:
        return SchedulerOutput.make_empty()

    # Fast path: no lane scheduled any tokens this step. We still need to
    # propagate finished-request and freed-encoder bookkeeping so the runner can
    # release that state, but everything else is empty.
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
        # CachedRequestData is a struct-of-arrays; concatenate each parallel
        # array so the merged object stays internally consistent. Skip empty
        # lanes to avoid touching the shared make_empty() sentinel.
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
        # Take the elementwise max so the merged batch reserves enough common
        # prefix blocks for the most demanding lane.
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
        # Stays None unless some lane reported invalid spec tokens, matching the
        # base output's "absent" representation rather than an empty dict.
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


class TTLaneCoordinator(SchedulerInterface):
    """Single-process multi-lane scheduler for TT gathered-batch execution.

    Owns one fully independent :class:`TTScheduler` per lane. Each lane
    scheduler has its own waiting/running queues and its own KV cache manager,
    so the coordinator behaves like several co-located gathered-DP engines: a
    request belongs to exactly one lane and only that lane's scheduler ever
    sees it. Every lane's KV cache manager is sized identically (the same
    ``kv_cache_config``) because each lane drives a physically separate, equally
    sized DP submesh cache; block IDs are therefore lane-local and repeat across
    lanes.

    The coordinator implements :class:`SchedulerInterface` by routing requests
    to their lane, negotiating the single shared scheduling mode each step,
    running every lane, and merging the per-lane results into one engine-facing
    ``SchedulerOutput`` tagged with :class:`LaneStepMetadata` so the runner can
    split it apart again.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.structured_output_manager = structured_output_manager
        self.log_stats = log_stats
        # Number of DP replicas (lanes) sharing this process.
        self.num_lanes = get_tt_data_parallel_size(vllm_config)
        # Max concurrent running requests a single lane may hold.
        self._per_lane_max = get_tt_per_lane_max_num_seqs(vllm_config)
        self._last_lane_metadata: LaneStepMetadata | None = None
        # No KV connector on TT; surfaced for engine-core attribute access.
        self.connector: KVConnectorBase_V1 | None = None

        # One independent scheduler per lane. Each gets the same kv_cache_config
        # (every submesh cache is the same size) and its own KV cache manager,
        # so lane block-ID spaces are independent. Per-lane stats are disabled;
        # the coordinator aggregates stats itself.
        self.lanes: list[TTScheduler] = [
            TTScheduler(
                vllm_config,
                kv_cache_config,
                structured_output_manager,
                block_size,
                mm_registry,
                include_finished_set,
                log_stats=False,
            )
            for _ in range(self.num_lanes)
        ]

    # ------------------------------------------------------------------
    # Lane selection / mode negotiation
    # ------------------------------------------------------------------

    def _pick_lane(self) -> int:
        """Choose the least-loaded lane for a newly arriving request.

        Scores each lane by load, weighting waiting requests more heavily than
        running ones (a queued request will cost a future prefill), and picks
        the lowest score. Ties resolve to the lowest lane index.
        """
        best_lane = 0
        best_score: int | None = None
        for lane, sched in enumerate(self.lanes):
            score = len(sched.waiting) * 4 + len(sched.running)
            if best_score is None or score < best_score:
                best_score = score
                best_lane = lane
        return best_lane

    def _lane_has_work(self, sched: TTScheduler) -> bool:
        """Whether a lane has any waiting or running requests to schedule."""
        return bool(sched.waiting) or bool(sched.running)

    def _local_prefill_intent(self, sched: TTScheduler) -> int:
        """Whether this lane *wants* to prefill this step (1) or not (0).

        A lane wants to prefill when it has queued requests and either nothing
        running (so it must prefill to make progress) or spare capacity to admit
        more alongside its running decodes.
        """
        has_waiting = bool(sched.waiting)
        has_running = bool(sched.running)
        has_capacity = len(sched.running) < self._per_lane_max
        return int(has_waiting and ((not has_running) or has_capacity))

    def _negotiate_forced_mode(self) -> TTSchedulingMode:
        """Pick the single mode (prefill- or decode-only) all lanes will run.

        The device executes all lanes together and cannot mix prefill with
        decode, so the lanes must agree. If *any* lane wants to prefill, the
        whole step is prefill-only; otherwise it is decode-only. Lanes without
        work for the chosen mode simply contribute an empty batch.
        """
        intent = max(self._local_prefill_intent(sched) for sched in self.lanes)
        return TTSchedulingMode.from_prefill_intent(intent)

    def _schedule_all_lanes(
        self, forced_mode: TTSchedulingMode
    ) -> list[SchedulerOutput]:
        """Run every lane scheduler under ``forced_mode``.

        Idle lanes are scheduled too (rather than short-circuited to an empty
        output) so each lane drains its own pending ``finished_req_ids`` for the
        runner's cleanup; an empty schedule for an idle lane is cheap.
        """
        lane_outputs: list[SchedulerOutput] = []
        for sched in self.lanes:
            sched.set_forced_mode(forced_mode)
            try:
                lane_outputs.append(sched.schedule())
            finally:
                sched.set_forced_mode(TTSchedulingMode.DEFAULT)
        return lane_outputs

    # ------------------------------------------------------------------
    # SchedulerInterface: scheduling
    # ------------------------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        forced_mode = self._negotiate_forced_mode()
        lane_outputs = self._schedule_all_lanes(forced_mode)
        merged = merge_lane_scheduler_outputs(lane_outputs)

        # Decode fallback: a forced prefill step can schedule zero tokens (no
        # chunked prefill + KV pressure means no full prefill fits). If any lane
        # has running decodes, falling back to a decode-only step keeps them
        # advancing — without this the step makes no global progress and the
        # engine livelocks. Mirrors the base scheduler's DEFAULT-mode fallback,
        # which the forced mode bypasses.
        if (
            forced_mode == TTSchedulingMode.PREFILL_ONLY
            and merged.total_num_scheduled_tokens == 0
            and any(sched.running for sched in self.lanes)
        ):
            # The discarded prefill pass already drained each lane's
            # finished/freed-encoder bookkeeping; carry it onto the decode pass
            # so the runner still releases that state.
            carried_finished = merged.finished_req_ids
            carried_free_encoder = merged.free_encoder_mm_hashes
            forced_mode = TTSchedulingMode.DECODE_ONLY
            lane_outputs = self._schedule_all_lanes(forced_mode)
            merged = merge_lane_scheduler_outputs(lane_outputs)
            merged.finished_req_ids |= carried_finished
            merged.free_encoder_mm_hashes = (
                carried_free_encoder + merged.free_encoder_mm_hashes
            )

        is_decode = forced_mode == TTSchedulingMode.DECODE_ONLY
        self._last_lane_metadata = LaneStepMetadata(
            lane_outputs=lane_outputs,
            is_decode=is_decode,
        )
        # Smuggle the metadata across to the runner on the output object itself.
        merged._tt_lane_step_metadata = self._last_lane_metadata
        return merged

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Mirrors the base scheduler, but over the union of every lane's
        # requests (request IDs are globally unique). Row order within the
        # bitmask is irrelevant: the runner remaps rows back to batch positions
        # by request ID via reorder_grammar_bitmask_for_tt_batch.
        requests: dict[str, Request] = {}
        for sched in self.lanes:
            requests.update(sched.requests)
        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := requests.get(req_id)) and req.use_structured_output
        ]
        if not structured_output_request_ids:
            return None
        bitmask = self.structured_output_manager.grammar_bitmask(
            requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    # ------------------------------------------------------------------
    # SchedulerInterface: output handling
    # ------------------------------------------------------------------

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        meta: LaneStepMetadata | None = getattr(
            scheduler_output, "_tt_lane_step_metadata", None
        )
        if meta is None:
            return {}

        # Each lane scheduler processes only its own SchedulerOutput. The merged
        # model_runner_output is passed through unchanged: its req_id_to_index
        # spans all lanes and the per-request dicts are keyed by (globally
        # unique) request ID, and the base update loop is driven by the lane's
        # num_scheduled_tokens, so a lane only ever touches its own requests.
        per_lane_outputs: list[dict[int, EngineCoreOutputs]] = [
            sched.update_from_output(lane_output, model_runner_output)
            for sched, lane_output in zip(self.lanes, meta.lane_outputs, strict=True)
        ]
        merged = self._merge_engine_core_outputs(per_lane_outputs)

        # Lanes run with stats disabled; attach the coordinator's aggregate
        # stats here, mirroring the base scheduler's placement.
        stats = self.make_stats()
        if stats is not None:
            eco = next(iter(merged.values()), None)
            if eco is None:
                merged[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats
        return merged

    @staticmethod
    def _merge_engine_core_outputs(
        per_lane_outputs: list[dict[int, EngineCoreOutputs]],
    ) -> dict[int, EngineCoreOutputs]:
        """Merge per-lane ``{client_index: EngineCoreOutputs}`` dicts.

        Concatenates each client's request outputs and unions its
        finished-request set. Lists/sets are rebound to fresh objects rather
        than mutated in place to avoid touching msgspec defaults.
        """
        merged: dict[int, EngineCoreOutputs] = {}
        for lane_dict in per_lane_outputs:
            for client_index, eco in lane_dict.items():
                existing = merged.get(client_index)
                if existing is None:
                    merged[client_index] = eco
                    continue
                if eco.outputs:
                    existing.outputs = existing.outputs + eco.outputs
                if eco.finished_requests:
                    if existing.finished_requests is None:
                        existing.finished_requests = set(eco.finished_requests)
                    else:
                        existing.finished_requests = (
                            existing.finished_requests | eco.finished_requests
                        )
        return merged

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for sched in self.lanes:
            sched.update_draft_token_ids(draft_token_ids)

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        meta: LaneStepMetadata | None = getattr(
            scheduler_output, "_tt_lane_step_metadata", None
        )
        if meta is None:
            return
        for sched, lane_output in zip(self.lanes, meta.lane_outputs, strict=True):
            sched.update_draft_token_ids_in_output(draft_token_ids, lane_output)

    # ------------------------------------------------------------------
    # SchedulerInterface: request lifecycle
    # ------------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        # Assign the request to a lane before handing it to that lane's
        # scheduler. An explicit preference wins (wrapped into range); otherwise
        # balance onto the least-loaded lane. A request that already carries a
        # valid lane keeps it.
        preferred_lane = getattr(request, "tt_preferred_lane", None)
        request_lane = getattr(request, "tt_lane", -1)
        if preferred_lane is not None:
            lane = preferred_lane % self.num_lanes
        elif 0 <= request_lane < self.num_lanes:
            lane = request_lane
        else:
            lane = self._pick_lane()
        request.tt_lane = lane
        self.lanes[lane].add_request(request)

    def finish_requests(
        self,
        request_ids: str | Iterable[str],
        finished_status: RequestStatus,
    ) -> None:
        # Broadcast to every lane; a lane no-ops for IDs it does not hold, so no
        # request->lane map is needed.
        for sched in self.lanes:
            sched.finish_requests(request_ids, finished_status)

    # ------------------------------------------------------------------
    # SchedulerInterface: queries / lifecycle
    # ------------------------------------------------------------------

    def get_num_unfinished_requests(self) -> int:
        return sum(sched.get_num_unfinished_requests() for sched in self.lanes)

    def has_finished_requests(self) -> bool:
        return any(sched.has_finished_requests() for sched in self.lanes)

    def get_request_counts(self) -> tuple[int, int]:
        num_running = 0
        num_waiting = 0
        for sched in self.lanes:
            running, waiting = sched.get_request_counts()
            num_running += running
            num_waiting += waiting
        return num_running, num_waiting

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        results = [
            sched.reset_prefix_cache(reset_running_requests, reset_connector)
            for sched in self.lanes
        ]
        return all(results)

    def reset_encoder_cache(self) -> None:
        for sched in self.lanes:
            sched.reset_encoder_cache()

    def make_stats(self) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        num_running, num_waiting = self.get_request_counts()
        kv_cache_usage = sum(
            sched.kv_cache_manager.usage for sched in self.lanes
        ) / len(self.lanes)
        return SchedulerStats(
            num_running_reqs=num_running,
            num_waiting_reqs=num_waiting,
            kv_cache_usage=kv_cache_usage,
        )

    def shutdown(self) -> None:
        for sched in self.lanes:
            sched.shutdown()

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return None
