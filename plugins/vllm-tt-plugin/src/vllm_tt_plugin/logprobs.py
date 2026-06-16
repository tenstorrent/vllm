# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.v1.outputs import LogprobsTensors


def build_logprobs_from_topk(
    top_k_logprobs: torch.Tensor,
    top_k_indices: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    max_num_logprobs: int,
) -> LogprobsTensors:
    """Build LogprobsTensors from device top-K logprobs.

    Device always computes top-32 logprobs sorted descending.
    This function trims to max_num_logprobs which is in range (0-20) to
    match the OpenAI API limit, then packs into LogprobsTensors format
    expected by the downstream vLLM pipeline.
    """
    sz = top_k_logprobs.shape[0]
    n = max_num_logprobs

    # Cast both tensors to int64 to avoid uint32/int32 promotion errors.
    if sampled_token_ids.dim() == 1:
        sampled_token_ids = sampled_token_ids.unsqueeze(-1)
    sampled_expanded = sampled_token_ids.to(torch.int64)
    # ttnn.sampling selects from the same fixed top-32 candidate set returned
    # here. If that device invariant changes, the sampled rank can be wrong.
    match_mask = top_k_indices.to(torch.int64) == sampled_expanded
    ranks = match_mask.int().argmax(dim=-1)

    if ranks.dim() < top_k_logprobs.dim():
        ranks = ranks.unsqueeze(-1)
    sampled_logprob = top_k_logprobs.gather(1, ranks.long())

    logprob_token_ids = torch.zeros(sz, n + 1, dtype=torch.int32)
    logprobs_values = torch.zeros(sz, n + 1, dtype=torch.float32)
    logprob_token_ids[:, 0] = sampled_token_ids.squeeze(-1)
    logprobs_values[:, 0] = sampled_logprob.squeeze(-1)
    logprob_token_ids[:, 1 : n + 1] = top_k_indices[:, :n].to(torch.int32)
    logprobs_values[:, 1 : n + 1] = top_k_logprobs[:, :n].to(torch.float32)

    selected_token_ranks = ranks.squeeze(-1).to(torch.int32)
    return LogprobsTensors(
        logprob_token_ids,
        logprobs_values,
        selected_token_ranks,
    )
