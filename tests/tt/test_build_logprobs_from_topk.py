# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for _build_logprobs_from_topk helper in tt_model_runner.

These tests use synthetic data only (no device required) to verify
that the helper correctly packs top-K logprobs into LogprobsTensors
for the downstream vLLM pipeline.
"""

import torch

from vllm.v1.worker.tt_model_runner import _build_logprobs_from_topk


class TestBuildLogprobsFromTopk:
    """Tests for _build_logprobs_from_topk."""

    def _make_sorted_topk(self, sz: int, k: int = 32):
        """Create synthetic sorted top-K data.

        Returns (top_k_logprobs[sz, k], top_k_indices[sz, k])
        where logprobs are sorted descending per row.
        """
        # Generate sorted descending logprobs
        logprobs = torch.sort(torch.randn(sz, k), dim=-1, descending=True).values
        # Generate unique token indices per row
        indices = torch.stack([torch.randperm(10000)[:k] for _ in range(sz)])
        return logprobs, indices.to(torch.int32)

    def test_basic_shape(self):
        """Output shape is [sz, max_num_logprobs + 1]."""
        sz, N = 4, 5
        logprobs, indices = self._make_sorted_topk(sz)
        sampled = indices[:, 0]  # sampled = top-1 token

        result = _build_logprobs_from_topk(logprobs, indices, sampled, N)

        assert result.logprob_token_ids.shape == (sz, N + 1)
        assert result.logprobs.shape == (sz, N + 1)
        assert result.selected_token_ranks.shape == (sz,)

    def test_sampled_token_at_col0(self):
        """Column 0 always contains the sampled token ID and logprob."""
        sz = 8
        logprobs, indices = self._make_sorted_topk(sz)
        # Sampled token is at rank 3 in the sorted list
        sampled = indices[:, 3].clone()

        result = _build_logprobs_from_topk(logprobs, indices, sampled, 5)

        assert torch.equal(result.logprob_token_ids[:, 0], sampled)
        expected_lps = logprobs[:, 3]
        assert torch.allclose(result.logprobs[:, 0], expected_lps, atol=1e-6)

    def test_top_n_at_cols_1_to_n(self):
        """Columns 1..N contain top-N tokens from sorted list."""
        sz, N = 4, 5
        logprobs, indices = self._make_sorted_topk(sz)
        sampled = indices[:, 0]

        result = _build_logprobs_from_topk(logprobs, indices, sampled, N)

        assert torch.equal(
            result.logprob_token_ids[:, 1 : N + 1],
            indices[:, :N].to(torch.int32),
        )
        assert torch.allclose(
            result.logprobs[:, 1 : N + 1],
            logprobs[:, :N].to(torch.float32),
            atol=1e-6,
        )

    def test_selected_token_ranks(self):
        """selected_token_ranks matches the sampled token's position."""
        sz = 4
        logprobs, indices = self._make_sorted_topk(sz)

        for rank in [0, 1, 5, 31]:
            sampled = indices[:, rank].clone()
            result = _build_logprobs_from_topk(logprobs, indices, sampled, 5)
            assert torch.all(result.selected_token_ranks == rank)

    def test_sampled_at_rank0(self):
        """Edge case: sampled token is the most likely (rank 0)."""
        sz = 4
        logprobs, indices = self._make_sorted_topk(sz)
        sampled = indices[:, 0]

        result = _build_logprobs_from_topk(logprobs, indices, sampled, 10)

        assert torch.all(result.selected_token_ranks == 0)
        assert torch.equal(result.logprob_token_ids[:, 0], sampled)

    def test_max_num_logprobs_zero(self):
        """max_num_logprobs=0 → shape [sz, 1] (sampled token only)."""
        sz = 4
        logprobs, indices = self._make_sorted_topk(sz)
        sampled = indices[:, 2]

        result = _build_logprobs_from_topk(logprobs, indices, sampled, 0)

        assert result.logprob_token_ids.shape == (sz, 1)
        assert result.logprobs.shape == (sz, 1)
        assert torch.equal(result.logprob_token_ids[:, 0], sampled)

    def test_dtypes(self):
        """Output dtypes match LogprobsTensors contract."""
        logprobs, indices = self._make_sorted_topk(4)
        sampled = indices[:, 0]

        result = _build_logprobs_from_topk(logprobs, indices, sampled, 5)

        assert result.logprob_token_ids.dtype == torch.int32
        assert result.logprobs.dtype == torch.float32
        assert result.selected_token_ranks.dtype == torch.int32

    def test_per_user_different_sampled(self):
        """Each user can have a different sampled token at different ranks."""
        sz = 4
        logprobs, indices = self._make_sorted_topk(sz)
        # User 0 sampled rank 0, user 1 rank 5, user 2 rank 31, user 3 rank 2
        ranks = [0, 5, 31, 2]
        sampled = torch.tensor(
            [indices[i, r].item() for i, r in enumerate(ranks)], dtype=torch.int32
        )

        result = _build_logprobs_from_topk(logprobs, indices, sampled, 10)

        for i, r in enumerate(ranks):
            assert result.selected_token_ranks[i].item() == r
            assert result.logprob_token_ids[i, 0].item() == sampled[i].item()
