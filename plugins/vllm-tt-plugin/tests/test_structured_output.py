# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm_tt_plugin.structured_output import reorder_grammar_bitmask_for_tt_batch


def test_reorder_grammar_bitmask_uses_lane_local_rows():
    bitmask = torch.tensor(
        [
            [10, 11],
            [20, 21],
            [30, 31],
            [40, 41],
        ],
        dtype=torch.int32,
    )

    reordered = reorder_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=["req-0", "req-1", "req-2", "req-3"],
        req_id_to_index={"req-0": 0, "req-1": 3, "req-2": 7, "req-3": 10},
        req_indices=[3, 10],
        batch_length=2,
    )

    assert torch.equal(
        reordered,
        torch.tensor(
            [
                [20, 21],
                [40, 41],
            ],
            dtype=torch.int32,
        ),
    )


def test_reorder_grammar_bitmask_ignores_structured_requests_outside_lane():
    bitmask = torch.tensor(
        [
            [10, 11],
            [20, 21],
        ],
        dtype=torch.int32,
    )

    reordered = reorder_grammar_bitmask_for_tt_batch(
        bitmask=bitmask,
        structured_output_request_ids=["req-0", "req-1"],
        req_id_to_index={"req-0": 0, "req-1": 3},
        req_indices=[3, 5],
        batch_length=2,
    )

    assert torch.equal(
        reordered,
        torch.tensor(
            [
                [20, 21],
                [-1, -1],
            ],
            dtype=torch.int32,
        ),
    )
