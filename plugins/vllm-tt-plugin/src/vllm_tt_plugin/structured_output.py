# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence

import torch


def reorder_grammar_bitmask_for_tt_batch(
    *,
    bitmask: torch.Tensor,
    structured_output_request_ids: Sequence[str],
    req_id_to_index: Mapping[str, int],
    req_indices: Sequence[int],
    batch_length: int,
) -> torch.Tensor:
    """Reorder scheduler bitmask rows into the TT lane-local batch layout."""
    grammar_bitmask_length = bitmask.shape[1]
    reordered_bitmask = torch.full(
        (batch_length, grammar_bitmask_length),
        -1,
        dtype=bitmask.dtype,
        device=bitmask.device,
    )

    req_id_to_bitmask_row: dict[str, int] = {
        req_id: i for i, req_id in enumerate(structured_output_request_ids)
    }
    req_index_to_local_row = {
        req_index: local_row for local_row, req_index in enumerate(req_indices)
    }
    for req_id, persistent_batch_index in req_id_to_index.items():
        scheduler_bitmask_row = req_id_to_bitmask_row.get(req_id)
        local_row = req_index_to_local_row.get(persistent_batch_index)
        if scheduler_bitmask_row is not None and local_row is not None:
            reordered_bitmask[local_row, :] = bitmask[scheduler_bitmask_row, :]

    return reordered_bitmask
