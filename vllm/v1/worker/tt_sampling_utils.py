from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch

SAMPLING_FIELDS: tuple[tuple[str, type, Any], ...] = (
    ("temperature", float, 0.0),
    ("top_k", int, 1),
    ("top_p", float, 1.0),
    ("presence_penalty", float, 0.0),
    ("frequency_penalty", float, 0.0),
    ("repetition_penalty", float, 1.0),
    ("n", int, 1),
    ("seed", Optional[int], None),
)


@dataclass(frozen=True)
class SamplingDefaults:
    temperature: float = 0.0
    top_k: int = 1
    top_p: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    n: int = 1
    seed: Optional[int] = None


DEFAULTS = SamplingDefaults()


def empty_sampling_lists():
    return {name: [] for name, _, _ in SAMPLING_FIELDS}


def sampling_lists_from_numpy(sampling, length: int):
    lists = empty_sampling_lists()
    lists["temperature"] = sampling.temperature_cpu[:length].tolist()
    lists["top_k"] = sampling.top_k_cpu[:length].astype(int).tolist()
    lists["top_p"] = sampling.top_p_cpu[:length].tolist()
    lists["presence_penalty"] = (
        sampling.presence_penalty_cpu[:length].tolist())
    lists["frequency_penalty"] = (
        sampling.frequency_penalty_cpu[:length].tolist())
    lists["repetition_penalty"] = (
        sampling.repetition_penalty_cpu[:length].tolist())
    lists["n"] = sampling.n_cpu[:length].astype(int).tolist()
    lists["seed"] = [
        int(value) if int(value) >= 0 else None
        for value in sampling.seed_cpu[:length].tolist()
    ]
    return lists


def coerce_sampling_lists(
    params,
    defaults: SamplingDefaults = DEFAULTS,
):
    lists = empty_sampling_lists()
    if isinstance(params.temperature, list):
        length = len(params.temperature)
    else:
        length = 1

    def _coerce(field, default):
        if isinstance(field, list):
            if len(field) != length:
                raise ValueError(
                    f"Expected list of length {length}, got {len(field)}")
            return list(field)
        value = default if field is None else field
        return [value] * length

    lists["temperature"] = _coerce(params.temperature, defaults.temperature)
    lists["top_k"] = [int(v) for v in _coerce(params.top_k, defaults.top_k)]
    lists["top_p"] = _coerce(params.top_p, defaults.top_p)
    lists["presence_penalty"] = _coerce(
        params.presence_penalty, defaults.presence_penalty)
    lists["frequency_penalty"] = _coerce(
        params.frequency_penalty, defaults.frequency_penalty)
    lists["repetition_penalty"] = _coerce(
        params.repetition_penalty, defaults.repetition_penalty)
    lists["n"] = [int(v) for v in _coerce(params.n, defaults.n)]
    lists["seed"] = _coerce(params.seed, defaults.seed)
    return lists


def pad_sampling_lists(
    lists,
    target_len: int,
    defaults: SamplingDefaults = DEFAULTS,
):
    current_len = len(lists["temperature"])
    if current_len >= target_len:
        return
    pad = target_len - current_len
    lists["temperature"].extend([defaults.temperature] * pad)
    lists["top_k"].extend([defaults.top_k] * pad)
    lists["top_p"].extend([defaults.top_p] * pad)
    lists["presence_penalty"].extend([defaults.presence_penalty] * pad)
    lists["frequency_penalty"].extend([defaults.frequency_penalty] * pad)
    lists["repetition_penalty"].extend([defaults.repetition_penalty] * pad)
    lists["n"].extend([defaults.n] * pad)
    lists["seed"].extend([defaults.seed] * pad)


def lists_to_tt_params(lists, cls):
    return cls(
        temperature=lists["temperature"],
        top_k=lists["top_k"],
        top_p=lists["top_p"],
        presence_penalty=lists["presence_penalty"],
        frequency_penalty=lists["frequency_penalty"],
        repetition_penalty=lists["repetition_penalty"],
        n=lists["n"],
        seed=lists["seed"],
    )


def flatten_sampling_lists(lists):
    ints = torch.cat(
        [
            torch.tensor(lists["top_k"], dtype=torch.int32),
            torch.tensor(lists["n"], dtype=torch.int32),
            torch.tensor(
                [seed if seed is not None else -1 for seed in lists["seed"]],
                dtype=torch.int32,
            ),
        ],
        dim=0,
    )
    floats = torch.cat(
        [
            torch.tensor(lists["temperature"], dtype=torch.float32),
            torch.tensor(lists["top_p"], dtype=torch.float32),
            torch.tensor(lists["presence_penalty"], dtype=torch.float32),
            torch.tensor(lists["frequency_penalty"], dtype=torch.float32),
            torch.tensor(lists["repetition_penalty"], dtype=torch.float32),
        ],
        dim=0,
    )
    return ints, floats


def sampling_lists_from_flat(ints: torch.Tensor, floats: torch.Tensor, length: int):
    off = 0
    top_k = ints[off:off + length].tolist()
    off += length
    n_values = ints[off:off + length].tolist()
    off += length
    seed_values = [
        int(val) if int(val) >= 0 else None
        for val in ints[off:off + length].tolist()
    ]

    f_off = 0
    temperature = floats[f_off:f_off + length].tolist()
    f_off += length
    top_p = floats[f_off:f_off + length].tolist()
    f_off += length
    presence = floats[f_off:f_off + length].tolist()
    f_off += length
    frequency = floats[f_off:f_off + length].tolist()
    f_off += length
    repetition = floats[f_off:f_off + length].tolist()

    return {
        "temperature": temperature,
        "top_k": [int(v) for v in top_k],
        "top_p": top_p,
        "presence_penalty": presence,
        "frequency_penalty": frequency,
        "repetition_penalty": repetition,
        "n": [int(v) for v in n_values],
        "seed": seed_values,
    }

