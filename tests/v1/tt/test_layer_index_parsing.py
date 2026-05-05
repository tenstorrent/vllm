# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``_parse_layer_index`` used by the per-group KV cache
allocator to map ``KVCacheGroupSpec.layer_names`` back to model-side
layer indices.
"""

import pytest


@pytest.mark.parametrize(
    "name,expected",
    [
        ("layers.0", 0),
        ("layers.7", 7),
        ("model.layers.5.self_attn", 5),
        ("model.language_model.layers.42.self_attn", 42),
        ("foo.layers.10", 10),
        ("layers.999.attn.q_proj", 999),
    ],
)
def test_parse_layer_index_valid(name, expected):
    from vllm_tt_plugin.model_runner import _parse_layer_index

    assert _parse_layer_index(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        "foo",
        "layer.0",  # singular "layer" — typo not a valid pattern
        "layers",
        "layers.",
        "model.layers.x.self_attn",  # non-numeric
        "",
    ],
)
def test_parse_layer_index_invalid(name):
    from vllm_tt_plugin.model_runner import _parse_layer_index

    with pytest.raises(ValueError, match="parse a layer index"):
        _parse_layer_index(name)
