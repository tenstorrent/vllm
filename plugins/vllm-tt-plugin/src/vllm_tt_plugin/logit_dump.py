# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Optional per-step host-sampling logit dump for structured-output debugging.

Enabled only when ``TT_LOGIT_DUMP_PATH`` is set. Writes JSONL: one line per
structured decode row per step, plus one prompt line the first time a request
id is seen. Every path is wrapped so a dump failure can never perturb
generation.

The point of the dump: at the host-sampling site the full-vocab logits are
already grammar-masked (forbidden ids are -inf), so the rank of a JSON closer
relative to the sampled token separates "grammar masked termination" from
"model preferred a repeat token".
"""

import contextlib
import json
import os

import torch

_PATH = os.environ.get("TT_LOGIT_DUMP_PATH")
_TOPK = int(os.environ.get("TT_LOGIT_DUMP_TOPK", "20"))
# Qwen3-VL single-token JSON closers: } " "} }\n ", "}\n  } \n <|im_end|>.
_DEFAULT_TERMINATORS = [92, 1, 9207, 532, 497, 16707, 335, 198, 151645]
_seen: set = set()


def enabled() -> bool:
    return _PATH is not None


def _terminators() -> list:
    env = os.environ.get("TT_LOGIT_DUMP_TERMINATORS")
    if env:
        return [int(x) for x in env.split(",") if x.strip()]
    return _DEFAULT_TERMINATORS


def _write(rec: dict) -> None:
    with open(_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")


def dump_rows(
    logits: torch.Tensor,
    next_token_ids: torch.Tensor,
    req_ids: list | None = None,
    prompt_tokens: torch.Tensor | None = None,
    output_tokens: torch.Tensor | None = None,
) -> None:
    """logits: [sz, vocab] host tensor, POST grammar mask."""
    if _PATH is None:
        return
    try:
        terms = _terminators()
        lp = torch.log_softmax(logits.float(), dim=-1)
        sz = logits.shape[0]
        for i in range(sz):
            req = req_ids[i] if req_ids and i < len(req_ids) else None
            if prompt_tokens is not None and req is not None and req not in _seen:
                _seen.add(req)
                row = prompt_tokens[i]
                ids = [int(t) for t in row.tolist() if int(t) >= 0]
                _write({"req": req, "kind": "prompt", "prompt_ids": ids})
            lv = logits[i]
            lpv = lp[i]
            sampled = int(next_token_ids[i])
            k = min(_TOPK, lv.shape[-1])
            _, tids = torch.topk(lv, k)
            step = None
            if output_tokens is not None:
                step = int((output_tokens[i] >= 0).sum())
            rec = {
                "req": req,
                "kind": "step",
                "step": step,
                "sampled": sampled,
                "sampled_logprob": float(lpv[sampled]),
                "topk": [[int(t), round(float(lpv[int(t)]), 4)] for t in tids],
                "terms": {},
            }
            for t in terms:
                val = lv[t]
                masked = bool(torch.isinf(val) and val < 0)
                rec["terms"][t] = {
                    "rank": int((lv > val).sum()),
                    "logprob": round(float(lpv[t]), 4),
                    "masked": masked,
                }
            _write(rec)
    except Exception as e:  # a debug dump must never break generation
        with contextlib.suppress(Exception):
            _write({"dump_error": repr(e)})
