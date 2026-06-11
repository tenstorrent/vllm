# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import numpy as np
import torch

from vllm.v1.sample.logits_processor import (
    BatchUpdateBuilder,
    LogitsProcessors,
    MoveDirectionality,
)
from vllm.v1.sample.logits_processor.builtin import (
    LogitBiasLogitsProcessor,
    MinPLogitsProcessor,
    MinTokensLogitsProcessor,
)
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState

# Sentinel value for None seed. vLLM treats -1 as equivalent to None
# (see SamplingParams.__post_init__), so we use -1 as the sentinel.
SEED_NONE_SENTINEL = -1

# Sentinel for logprobs=None (disabled). Can't use 0 because
# SamplingParams.logprobs=0 means "return the sampled token's logprob".
# Can't use -1 because SamplingParams.logprobs=-1 means "all vocab logprobs".
# although -1 gets remapped before writing to SamplingInputBatch.num_logprobs
LOGPROBS_NONE_SENTINEL = -2


class SamplingInputBatch:
    # Default values for padding sampling parameters in decode mode.
    DEFAULTS = {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "repetition_penalty": 1.0,
        "seed": SEED_NONE_SENTINEL,  # Sentinel represents None (no seed)
        "num_logprobs": LOGPROBS_NONE_SENTINEL,
    }

    def __init__(self, max_num_reqs: int, logitsprocs: LogitsProcessors | None = None):
        self.max_num_reqs = max_num_reqs
        # Initialize sampling parameter tensors with default values.
        default_tensors = self.create_default_tensors()
        # Set attributes explicitly for each parameter.
        self.temperature = default_tensors["temperature"]
        self.top_p = default_tensors["top_p"]
        self.top_k = default_tensors["top_k"]
        self.presence_penalty = default_tensors["presence_penalty"]
        self.frequency_penalty = default_tensors["frequency_penalty"]
        self.repetition_penalty = default_tensors["repetition_penalty"]
        self.seed = default_tensors["seed"]
        self.num_logprobs = default_tensors["num_logprobs"]
        # Asserting that all defaults have corresponding attributes.
        for name in self.DEFAULTS:
            assert hasattr(self, name), (
                f"Missing attribute '{name}' in SamplingInputBatch"
            )

        # req_index -> generator
        # NOTE: The indices of the requests that do not have their own
        # generator should not be included in the dictionary.
        self.generators: dict[int, torch.Generator] = {}

        # Internal representation of per-step batch state changes, used for
        # reordering persistent batch and generating logitsprocs batch state
        # updates. Should reset each step.
        self.batch_update_builder = BatchUpdateBuilder()

        # Loaded logits processors (builtin + optional custom), initialized by
        # the model runner and passed in here.
        self.logitsprocs = logitsprocs or LogitsProcessors()

        # Allowed token IDs tracking
        self.has_allowed_token_ids: set[str] = set()
        # NOTE: In the mask tensor, if the corresponding token is allowed,
        # the value is False. Since we use masked_fill_ to set -inf.
        self.allowed_token_ids_mask: torch.Tensor | None = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

    def has_active_logitsprocs(self) -> bool:
        """True if any logits processors have active per-request state."""
        for proc in self.logitsprocs.all:
            if isinstance(proc, MinPLogitsProcessor) and proc.min_p_count:
                return True
            if isinstance(proc, LogitBiasLogitsProcessor) and proc.biases:
                return True
            if isinstance(proc, MinTokensLogitsProcessor) and proc.min_toks:
                return True
        return False

    def create_default_tensors(self) -> dict[str, torch.Tensor]:
        """Create tensors filled with default values for all parameters in
        DEFAULTS."""
        # Map Python types to PyTorch dtypes
        # Note: torch.full infers dtype, but int defaults to int64, so we
        # explicitly specify int32.
        dtype_map = {
            float: torch.float32,
            int: torch.int32,
            bool: torch.bool,
        }
        result: dict[str, torch.Tensor] = {}
        for name, default_value in self.DEFAULTS.items():
            dtype = dtype_map[type(default_value)]
            result[name] = torch.full((self.max_num_reqs,), default_value, dtype=dtype)
        return result


class InputBatch:
    """Persistent input batch, based on InputBatch for GPU/TPU backends."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        block_sizes: list[int],  # The block_size of each kv cache group
        kernel_block_sizes: list[int],
        logitsprocs: LogitsProcessors | None = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size

        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}
        # Sampling fast-path bookkeeping (track by req_id like GPUInputBatch).
        # These are used to answer common "batch-wide" queries in O(1).
        self.random_reqs: set[str] = set()
        self.presence_penalties_reqs: set[str] = set()
        self.frequency_penalties_reqs: set[str] = set()
        self.repetition_penalties_reqs: set[str] = set()

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            dtype=torch.int32,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()

        self.num_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)

        # Block table.
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            pin_memory=False,
            device="cpu",
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
        )

        self.req_output_token_ids: list[list[int] | None] = []

        # Sampling-related.
        self.sampling = SamplingInputBatch(max_num_reqs, logitsprocs=logitsprocs)

        # Slot remap for seed manager: remap[i] = j means slot i's data came
        # from slot j after condense.  Identity when nothing moved.
        self._slot_remap = torch.arange(max_num_reqs, dtype=torch.int32)

    def pop_slot_remap(self) -> torch.Tensor:
        """Return pending slot remap and reset to identity."""
        remap = self._slot_remap
        self._slot_remap = torch.arange(self.max_num_reqs, dtype=torch.int32)
        return remap

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    @property
    def all_greedy(self) -> bool:
        """True iff all active requests are greedy (temperature == 0.0)."""
        return len(self.random_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        """True iff no active request has sampling penalties."""
        return (
            len(self.presence_penalties_reqs) == 0
            and len(self.frequency_penalties_reqs) == 0
            and len(self.repetition_penalties_reqs) == 0
        )

    def add_request(
        self,
        request: "CachedRequestState",
        req_index: int | None = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs, (
            f"req_index={req_index} >= max_num_reqs={self.max_num_reqs}"
        )

        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        prompt_token_ids = request.prompt_token_ids
        assert prompt_token_ids is not None, "prompt_embeds are not supported for TT"
        num_prompt_tokens = len(prompt_token_ids)
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        self.token_ids_cpu[req_index, :num_prompt_tokens] = prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index, start_idx:end_idx] = request.output_token_ids
        # Number of token ids in token_ids_cpu.
        self.num_tokens[req_index] = request.num_tokens

        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens
        self.block_table.add_row(request.block_ids, req_index)

        # Sampling-related.
        sampling_params = request.sampling_params
        assert sampling_params is not None, "pooling requests not supported yet"

        # Register with batch update builder for logits processors
        self.sampling.batch_update_builder.added.append(
            (
                req_index,
                sampling_params,
                request.prompt_token_ids,
                request.output_token_ids,
            )
        )

        self.sampling.temperature[req_index] = sampling_params.temperature
        top_p = sampling_params.top_p
        top_k = sampling_params.top_k
        if not (0 < top_k < self.vocab_size):
            # Normalize top_k <= 0 or >= vocab_size to vocab_size
            # (consider all tokens)
            top_k = self.vocab_size
        # Workaround for https://github.com/tenstorrent/tt-metal/issues/46827
        # top_k == 1 means greedy/argmax for this request. The on-device sampler
        # always builds a fixed top-32 candidate set and its per-user top_k does
        # NOT collapse that set to a single token before the RNG draw, so with
        # any top_p < 1.0 a multi-token nucleus survives and the random seed makes
        # top_k=1 non-deterministic (e.g. Qwen3's generation_config defaults
        # top_p=0.95). Force top_p to 0 so the nucleus keeps exactly the single
        # most-probable token (cum_prob > 0 keeps one), i.e. exact argmax and
        # RNG-independent. Per-request, so mixed-k batches are unaffected.
        if top_k == 1:
            top_p = 0.0
        self.sampling.top_p[req_index] = top_p
        self.sampling.top_k[req_index] = top_k
        self.sampling.presence_penalty[req_index] = sampling_params.presence_penalty
        self.sampling.frequency_penalty[req_index] = sampling_params.frequency_penalty
        self.sampling.repetition_penalty[req_index] = sampling_params.repetition_penalty
        # Store seed, using sentinel value for None
        self.sampling.seed[req_index] = (
            sampling_params.seed
            if sampling_params.seed is not None
            else SEED_NONE_SENTINEL
        )

        # Update fast-path bookkeeping sets.
        # NOTE: Use `discard()` because `req_id` can be reused (abort+resubmit)
        # and slots can be overwritten.
        if sampling_params.temperature == 0.0:
            self.random_reqs.discard(req_id)
        else:
            self.random_reqs.add(req_id)
        if sampling_params.presence_penalty == 0.0:
            self.presence_penalties_reqs.discard(req_id)
        else:
            self.presence_penalties_reqs.add(req_id)
        if sampling_params.frequency_penalty == 0.0:
            self.frequency_penalties_reqs.discard(req_id)
        else:
            self.frequency_penalties_reqs.add(req_id)
        if sampling_params.repetition_penalty == 1.0:
            self.repetition_penalties_reqs.discard(req_id)
        else:
            self.repetition_penalties_reqs.add(req_id)

        # Generator for random sampling
        if request.generator is not None:
            self.sampling.generators[req_index] = request.generator

        # Logprobs (-1 means all vocab logprobs, remap to vocab_size)
        if sampling_params.logprobs is not None:
            self.sampling.num_logprobs[req_index] = (
                self.vocab_size
                if sampling_params.logprobs == -1
                else sampling_params.logprobs
            )
        else:
            self.sampling.num_logprobs[req_index] = LOGPROBS_NONE_SENTINEL

        # Allowed token IDs
        if sampling_params.allowed_token_ids:
            self.sampling.has_allowed_token_ids.add(req_id)
            if self.sampling.allowed_token_ids_mask is None:
                # Lazy allocation for this tensor, which can be large.
                # True means we fill with -inf (disallowed).
                self.sampling.allowed_token_ids_mask = torch.zeros(
                    self.max_num_reqs, self.vocab_size, dtype=torch.bool, device="cpu"
                )
            self.sampling.allowed_token_ids_mask[req_index] = True
            # False means we don't fill with -inf (allowed).
            self.sampling.allowed_token_ids_mask[req_index][
                sampling_params.allowed_token_ids
            ] = False
        elif self.sampling.allowed_token_ids_mask is not None:
            # This request has no allowlist. The slot may have been reused from
            # a previous request that did, so its mask row could hold stale
            # "disallowed" bits. The mask is read as ``mask[req_indices]``
            # whenever *any* batched request has an allowlist, so a stale row
            # would wrongly constrain this request. Reset it.
            self.sampling.allowed_token_ids_mask[req_index] = False

        # Bad words
        if sampling_params.bad_words_token_ids:
            self.sampling.bad_words_token_ids[req_index] = (
                sampling_params.bad_words_token_ids
            )

    def remove_request(self, req_id: str) -> int | None:
        """This method must always be followed by a call to condense()."""

        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        self.sampling.batch_update_builder.removed_append(req_index)
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        # Update fast-path bookkeeping sets.
        self.random_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)

        # Clean up host-only sampling param tracking
        self.sampling.generators.pop(req_index, None)
        self.sampling.has_allowed_token_ids.discard(req_id)
        self.sampling.bad_words_token_ids.pop(req_index, None)
        # Clear the allowlist mask row so a stale "disallowed" set can never
        # survive into a request that later reuses this slot.
        if self.sampling.allowed_token_ids_mask is not None:
            self.sampling.allowed_token_ids_mask[req_index] = False

        return req_index

    def condense(self, empty_req_indices: list[int]) -> None:
        """Move non-empty requests down into lower, empty indices.

        Args:
            empty_req_indices: empty batch indices, sorted descending.
        """
        num_reqs = self.num_reqs
        if num_reqs == 0:
            # The batched states are empty.
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        # NOTE(woosuk): This function assumes that the empty_req_indices
        # is sorted in descending order.
        last_req_index = num_reqs + len(empty_req_indices) - 1
        while empty_req_indices:
            # Find the largest non-empty index.
            while last_req_index in empty_req_indices:
                last_req_index -= 1

            # Find the smallest empty index.
            empty_index = empty_req_indices.pop()
            if empty_index >= last_req_index:
                break

            # Track the move for logits processors
            self.sampling.batch_update_builder.moved.append(
                (last_req_index, empty_index, MoveDirectionality.UNIDIRECTIONAL)
            )
            # Track for on-device seed manager slot reindexing.
            self._slot_remap[empty_index] = self._slot_remap[last_req_index]

            # Swap the states.
            req_id = self._req_ids[last_req_index]
            output_token_ids = self.req_output_token_ids[last_req_index]
            assert req_id is not None
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = output_token_ids
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            num_tokens = self.num_tokens[last_req_index]
            self.token_ids_cpu[empty_index, :num_tokens] = self.token_ids_cpu[
                last_req_index, :num_tokens
            ]
            self.num_tokens[empty_index] = num_tokens
            self.num_prompt_tokens[empty_index] = self.num_prompt_tokens[last_req_index]
            self.num_computed_tokens_cpu[empty_index] = self.num_computed_tokens_cpu[
                last_req_index
            ]
            self.block_table.move_row(last_req_index, empty_index)

            # Sampling-related.
            sampling = self.sampling
            sampling.temperature[empty_index] = sampling.temperature[last_req_index]
            sampling.top_p[empty_index] = sampling.top_p[last_req_index]
            sampling.top_k[empty_index] = sampling.top_k[last_req_index]
            sampling.presence_penalty[empty_index] = sampling.presence_penalty[
                last_req_index
            ]
            sampling.frequency_penalty[empty_index] = sampling.frequency_penalty[
                last_req_index
            ]
            sampling.repetition_penalty[empty_index] = sampling.repetition_penalty[
                last_req_index
            ]
            sampling.seed[empty_index] = sampling.seed[last_req_index]
            sampling.num_logprobs[empty_index] = sampling.num_logprobs[last_req_index]

            # Move host-only sampling params
            if last_req_index in self.sampling.generators:
                self.sampling.generators[empty_index] = self.sampling.generators.pop(
                    last_req_index
                )

            if last_req_index in self.sampling.bad_words_token_ids:
                self.sampling.bad_words_token_ids[empty_index] = (
                    self.sampling.bad_words_token_ids.pop(last_req_index)
                )

            # Move allowed_token_ids_mask row
            if self.sampling.allowed_token_ids_mask is not None:
                self.sampling.allowed_token_ids_mask[empty_index] = (
                    self.sampling.allowed_token_ids_mask[last_req_index]
                )

            # Decrement last_req_index since it is now empty.
            last_req_index -= 1

        # Trim lists to the batch size.
        del self._req_ids[self.num_reqs :]
        del self.req_output_token_ids[self.num_reqs :]

    @property
    def max_num_logprobs(self) -> int | None:
        """Returns the max logprobs across requests, or None if none need logprobs."""
        if self.num_reqs == 0:
            return None
        max_val = int(self.sampling.num_logprobs[: self.num_reqs].max().item())
        if max_val < 0:
            return None
        return max_val

    @property
    def no_allowed_token_ids(self) -> bool:
        """True if no requests have allowed_token_ids set."""
        return len(self.sampling.has_allowed_token_ids) == 0

    def refresh_logitsprocs(self) -> None:
        """Update logits processors with batch state changes."""

        # For non-pooling models - generate and apply logitsprocs update;
        # reset batch update tracking.
        # Update sampling metadata if batch state is changed.
        batch_update = self.sampling.batch_update_builder.get_and_reset(self.num_reqs)
        for logit_proc in self.sampling.logitsprocs.all:
            logit_proc.update_state(batch_update)

    def make_prompt_token_ids_tensor(
        self, req_indices: list[int] | None = None
    ) -> torch.Tensor:
        """Create a tensor of prompt token IDs, padded with -1.

        ``req_indices`` selects which rows of the persistent batch to emit (one
        row per index, in order). ``None`` means the whole local batch
        (``range(num_reqs)``). Lane-DP passes one lane's indices so the result
        is attributed to that lane's requests rather than the merged batch's
        leading rows.

        NOTE: TT device sampling relies on -1 as the padding sentinel.
        If these tokens are passed to the host sampler for penalties, they must
        be canonicalized (cast to int64 and -1 replaced with vocab_size) before
        scatter operations.
        """
        rows = list(range(self.num_reqs)) if req_indices is None else list(req_indices)
        n = len(rows)
        idx = np.asarray(rows, dtype=np.int64)
        max_prompt_len = int(self.num_prompt_tokens[idx].max()) if n > 0 else 0
        prompt_token_ids_tensor = torch.full(
            (n, max_prompt_len),
            -1,
            device="cpu",
            dtype=torch.int32,
        )
        prompt_token_ids = prompt_token_ids_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[idx, :max_prompt_len]
        # Pad with -1 for positions beyond actual prompt length
        for row, i in enumerate(rows):
            prompt_token_ids[row, self.num_prompt_tokens[i] :] = -1
        return prompt_token_ids_tensor

    def make_output_token_ids_tensor(
        self, req_indices: list[int] | None = None
    ) -> torch.Tensor:
        """Create a tensor of output token IDs, padded with -1.

        ``req_indices`` selects which rows of the persistent batch to emit (see
        ``make_prompt_token_ids_tensor``).

        NOTE: TT device sampling relies on -1 as the padding sentinel.
        If these tokens are used by the host sampler penalties logic, -1 padding
        should be removed/handled before use.
        """
        rows = list(range(self.num_reqs)) if req_indices is None else list(req_indices)
        n = len(rows)
        idx = np.asarray(rows, dtype=np.int64)
        output_lens = self.num_tokens[idx] - self.num_prompt_tokens[idx]
        max_output_len = int(output_lens.max()) if n > 0 else 0

        output_token_ids_tensor = torch.full(
            (n, max_output_len),
            -1,
            device="cpu",
            dtype=torch.int32,
        )
        output_token_ids = output_token_ids_tensor.numpy()
        # Copy output tokens from token_ids_cpu
        for row, i in enumerate(rows):
            prompt_len = self.num_prompt_tokens[i]
            total_len = self.num_tokens[i]
            output_len = total_len - prompt_len
            if output_len > 0:
                output_token_ids[row, :output_len] = self.token_ids_cpu[
                    i, prompt_len:total_len
                ]
        return output_token_ids_tensor

    def advance_generators(self, req_indices: list[int] | None = None) -> None:
        # This relies on the fact, that for a torch all_gather_object,
        # the local object is also copied,
        # so the original object is not modified.
        # Otherwise, the generator at local_rank 0
        # would get out of sync with the others.
        #
        # ``req_indices`` restricts advancement to the build's own requests.
        # Each generator belongs to a single request, so lane-DP (which calls
        # this once per lane) passes the lane's indices to advance every
        # generator exactly once per step rather than once per lane. ``None``
        # advances all generators (whole-batch build, called once per step).
        if req_indices is None:
            generators = list(self.sampling.generators.values())
        else:
            generators = [
                self.sampling.generators[i]
                for i in req_indices
                if i in self.sampling.generators
            ]
        for generator in generators:
            # Sample once from the generator to advance its state.
            torch.rand(1, generator=generator)


class TTLaneInputBatch(InputBatch):
    """Persistent input batch for single-process multi-lane (lane-DP) execution.

    One engine process drives ``num_lanes`` data-parallel KV-cache replicas
    ("lanes") that execute in lockstep against a single gathered device batch.
    This batch owns the lane layout so the model runner does not: it lays the
    persistent rows out as ``num_lanes`` contiguous chunks of ``per_lane`` rows
    and binds each request to a stable row for its whole lifetime.

    Layout: lane ``l`` owns rows ``[l * per_lane, (l + 1) * per_lane)``. A
    request placed at lane-local slot ``s`` lives at persistent row
    ``l * per_lane + s``. **That persistent row IS the request's device decode
    slot**, so the merged device input is the batch's own row layout -- no
    scatter, and no separate ``req_id -> slot`` map. ``max_num_reqs`` is
    ``num_lanes * per_lane`` (the global ``max_num_seqs`` in lane mode).

    Stable slots: a request never moves once placed. Removing a request leaves
    its row as an empty gap to be reused by a later request in the same lane.
    There is no condense (``condense`` is a no-op): keeping every live request
    pinned to its row keeps the on-device per-slot seed RNG correct and makes
    the seed manager's ``slot_remap`` the identity. Gaps are reset to neutral
    sampling defaults so they cannot perturb batch-wide flags (``all_greedy`` /
    ``no_penalties``) or sample an invalid value.

    Merged host sampling: because rows are the device slots and gaps carry
    neutral defaults, the runner samples the whole ``max_num_reqs`` slot batch
    in one call against one :class:`SamplingMetadata` built here over every row
    (``build_merged_sampling_metadata``). The builtin/custom logits processors
    keep per-row state over this full slot batch (``refresh_logitsprocs`` passes
    ``max_num_reqs`` as the batch size), exactly like a normal single-engine
    vLLM batch -- so there is no per-lane slicing, no per-lane generator/penalty
    remap, and custom logits processors work unchanged. Pad rows sample greedy
    garbage that the runner drops when reading back the occupied rows.
    """

    def __init__(
        self,
        num_lanes: int,
        per_lane: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        vocab_size: int,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        logitsprocs: LogitsProcessors | None = None,
    ):
        if num_lanes < 1 or per_lane < 1:
            raise ValueError(
                f"num_lanes and per_lane must be >= 1, got num_lanes={num_lanes}, "
                f"per_lane={per_lane}"
            )
        self.num_lanes = num_lanes
        self.per_lane = per_lane
        super().__init__(
            max_num_reqs=num_lanes * per_lane,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            vocab_size=vocab_size,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            logitsprocs=logitsprocs,
        )
        # Rows are a fixed slot grid (lane-chunked), not a front-packed list:
        # pre-size so a request can occupy any slot in its lane's chunk, with
        # gaps, instead of always appending at ``num_reqs``.
        self._req_ids = [None] * self.max_num_reqs
        self.req_output_token_ids = [None] * self.max_num_reqs
        # req_id -> lane (static; set at admission, never changes).
        self._lane_of: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Lane geometry / membership
    # ------------------------------------------------------------------

    def lane_of(self, req_id: str) -> int:
        """Return the lane a request is bound to."""
        return self._lane_of[req_id]

    def lane_base_row(self, lane: int) -> int:
        """First persistent row of ``lane``'s chunk."""
        return lane * self.per_lane

    def occupied_rows(self) -> list[int]:
        """Persistent rows holding a live request, in ascending (lane-major,
        slot) order. This is the canonical merged order used for output."""
        return [row for row, rid in enumerate(self._req_ids) if rid is not None]

    # ------------------------------------------------------------------
    # Placement (stable lane-local slots)
    # ------------------------------------------------------------------

    def add_request_to_lane(self, request: "CachedRequestState", lane: int) -> int:
        """Place ``request`` at the lowest free slot in ``lane``'s chunk.

        The lane is decided by the scheduler and passed in; this method only
        records the membership and assigns the stable row. Returns the
        persistent row (== device decode slot).
        """
        if not (0 <= lane < self.num_lanes):
            raise ValueError(f"lane {lane} out of range [0, {self.num_lanes})")
        row = self._claim_free_slot(lane)
        super().add_request(request, row)
        self._lane_of[request.req_id] = lane
        return row

    def _claim_free_slot(self, lane: int) -> int:
        """Lowest free row in ``lane``'s chunk, reconciled with pending removals.

        If the chosen row was freed earlier in this same step it is still in the
        logitsproc batch-update ``removed`` list. Drop it from that list so the
        reused row is recorded only as an ``added`` update, not both -- mirroring
        upstream ``gpu_input_batch._register_add_request``'s ``pop_removed()`` so
        the builtin logits processors do not first set then clear the new
        request's per-row state.
        """
        base = self.lane_base_row(lane)
        builder = self.sampling.batch_update_builder
        for slot in range(self.per_lane):
            row = base + slot
            if self._req_ids[row] is None:
                if row in builder._removed:
                    builder._removed.remove(row)
                return row
        raise ValueError(f"lane {lane} has no free slot (capacity {self.per_lane})")

    def remove_request(self, req_id: str) -> int | None:
        row = super().remove_request(req_id)
        if row is not None:
            self._lane_of.pop(req_id, None)
            self._reset_slot(row)
        return row

    def _reset_slot(self, row: int) -> None:
        """Reset a freed row to neutral defaults so a gap never perturbs the
        merged batch's sampling. ``super().remove_request`` already clears the
        generator, bad-words and allowed-token-ids entries; this also resets the
        per-row sampling tensors and token counts (so the row reads as an empty,
        greedy, no-penalty request until it is reused)."""
        sampling = self.sampling
        for name, default in sampling.DEFAULTS.items():
            getattr(sampling, name)[row] = default
        self.num_tokens[row] = 0
        self.num_prompt_tokens[row] = 0
        self.num_computed_tokens_cpu[row] = 0

    def condense(self, empty_req_indices: list[int]) -> None:
        """No-op: lane slots are stable.

        The base class condenses by moving the highest live request into the
        lowest empty index. That would move requests across lane boundaries and
        shift their device slots, corrupting the on-device per-slot seed RNG.
        Lane mode instead leaves freed rows as gaps (reused in place by later
        requests in the same lane), so live requests never move and the seed
        manager's ``slot_remap`` stays the identity.
        """
        return

    # ------------------------------------------------------------------
    # Sampling layout (merged, over the full slot batch)
    # ------------------------------------------------------------------

    @property
    def max_num_logprobs(self) -> int | None:
        """Max logprobs across live requests, or None if none need logprobs.

        Computed over every slot row rather than ``[:num_reqs]`` because live
        rows are not front-packed; gap rows carry the ``LOGPROBS_NONE_SENTINEL``
        default (the minimum value), so the max over all rows equals the max
        over the live rows.
        """
        if self.num_reqs == 0:
            return None
        max_val = int(self.sampling.num_logprobs[: self.max_num_reqs].max().item())
        if max_val < 0:
            return None
        return max_val

    def refresh_logitsprocs(self) -> None:
        """Apply batch state changes to logits processors over the full slot
        batch. Passes ``max_num_reqs`` (not ``num_reqs``) as the batch size so
        each processor's per-row state spans every slot, matching the full slot
        logits the runner samples."""
        batch_update = self.sampling.batch_update_builder.get_and_reset(
            self.max_num_reqs
        )
        for logit_proc in self.sampling.logitsprocs.all:
            logit_proc.update_state(batch_update)

    def build_merged_sampling_metadata(self) -> SamplingMetadata:
        """Build one :class:`SamplingMetadata` over every slot row.

        Mirrors a normal single-engine vLLM ``SamplingMetadata`` build, but over
        the full ``max_num_reqs`` slot batch (live rows interleaved with neutral
        gap rows) so it lines up row-for-row with the full slot logits the
        runner hands the host sampler and with the per-row logits-processor
        state. Gap rows carry neutral defaults (greedy, no penalties), so they
        do not change ``all_greedy`` / ``no_penalties`` and sample harmless
        greedy tokens the runner discards.

        ``all_random`` is intentionally computed over the full tensor, so it is
        False whenever any gap exists; that keeps the sampler's div-by-zero
        guard active for the default-``temperature=0`` gap rows.
        """
        n = self.max_num_reqs
        sampling = self.sampling
        temperature = sampling.temperature[:n]
        all_greedy = bool((temperature == 0.0).all())
        all_random = bool((temperature != 0.0).all())
        presence = sampling.presence_penalty[:n]
        frequency = sampling.frequency_penalty[:n]
        repetition = sampling.repetition_penalty[:n]
        no_penalties = bool(
            (presence == 0.0).all()
            and (frequency == 0.0).all()
            and (repetition == 1.0).all()
        )
        rows = list(range(n))
        if not no_penalties:
            prompt_token_ids = self.make_prompt_token_ids_tensor(rows).to(torch.int64)
            prompt_token_ids = prompt_token_ids.masked_fill(
                prompt_token_ids == -1, self.vocab_size
            )
            output_rows = self.make_output_token_ids_tensor(rows)
            output_token_ids = [
                [tok for tok in row.tolist() if tok != -1] for row in output_rows
            ]
        else:
            prompt_token_ids = None
            output_token_ids = [[] for _ in range(n)]
        allowed_token_ids_mask = sampling.allowed_token_ids_mask
        if allowed_token_ids_mask is not None:
            allowed_token_ids_mask = allowed_token_ids_mask[:n]
        return SamplingMetadata(
            temperature=temperature if not all_greedy else None,
            all_greedy=all_greedy,
            all_random=all_random,
            top_p=sampling.top_p[:n],
            top_k=sampling.top_k[:n],
            generators=dict(sampling.generators),
            max_num_logprobs=self.max_num_logprobs,
            no_penalties=no_penalties,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=frequency,
            presence_penalties=presence,
            repetition_penalties=repetition,
            output_token_ids=output_token_ids,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=dict(sampling.bad_words_token_ids),
            logitsprocs=sampling.logitsprocs,
        )
