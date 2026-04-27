# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared TT scheduling mode enum used by fork-only execution code.

The TT scheduler implementation lives in ``vllm_tt_plugin.scheduler``. This
module remains in-tree during Phase 1 because fork core code still imports
``TTSchedulingMode`` for gathered-DP execution coordination.
"""

from enum import Enum


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


__all__ = ["TTSchedulingMode"]
