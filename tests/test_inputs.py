# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.config import ModelConfig
from vllm.inputs.preprocess import InputPreprocessor

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("model_id", ["facebook/chameleon-7b"])
@pytest.mark.parametrize("prompt", ["", {"prompt_token_ids": []}])
@pytest.mark.skip(
    reason=(
        "Applying huggingface processor on text inputs results in "
        "significant performance regression for multimodal models. "
        "See https://github.com/vllm-project/vllm/issues/26320"
    )
)
def test_preprocessor_always_mm_code_path(model_id, prompt):
    model_config = ModelConfig(model=model_id)
    input_preprocessor = InputPreprocessor(model_config)

    # HF processor adds sep token
    tokenizer = input_preprocessor.get_tokenizer()
    sep_token_id = tokenizer.vocab[tokenizer.sep_token]

    processed_inputs = input_preprocessor.preprocess(prompt)
    assert sep_token_id in processed_inputs["prompt_token_ids"]


def test_truncate_tokens_prompt():
    """Test that truncation is applied to pre-tokenized (TokensPrompt) inputs.

    Regression test for tenstorrent/vllm#362: _prompt_to_llm_inputs was not
    passing tokenization_kwargs to _process_tokens, so truncation was silently
    skipped for any path that supplies prompt_token_ids directly (e.g. the
    Harmony/GPT-OSS chat template renderer).
    """
    from unittest.mock import MagicMock

    from vllm.inputs.preprocess import InputPreprocessor

    mock_tokenizer = MagicMock()
    mock_tokenizer.truncation_side = "left"

    preprocessor = InputPreprocessor.__new__(InputPreprocessor)
    preprocessor.tokenizer = mock_tokenizer
    preprocessor.model_config = MagicMock()
    preprocessor.model_config.is_encoder_decoder = False
    preprocessor.model_config.multimodal_config = None
    preprocessor.mm_processor_cache = None
    preprocessor.mm_cache_stats = None

    original_ids = list(range(50))
    prompt = dict(prompt_token_ids=original_ids)

    # Without truncation kwargs
    result = preprocessor.preprocess(prompt)
    assert result["prompt_token_ids"] == original_ids

    # With truncation kwargs - should left-truncate to 20 tokens
    trunc_kwargs = dict(truncation=True, max_length=20)
    result = preprocessor.preprocess(prompt, tokenization_kwargs=trunc_kwargs)
    assert len(result["prompt_token_ids"]) == 20
    assert result["prompt_token_ids"] == original_ids[-20:]

    # With right truncation
    mock_tokenizer.truncation_side = "right"
    result = preprocessor.preprocess(prompt, tokenization_kwargs=trunc_kwargs)
    assert len(result["prompt_token_ids"]) == 20
    assert result["prompt_token_ids"] == original_ids[:20]

    # Prompt shorter than max_length - unchanged
    short_ids = list(range(10))
    prompt_short = dict(prompt_token_ids=short_ids)
    result = preprocessor.preprocess(prompt_short, tokenization_kwargs=trunc_kwargs)
    assert result["prompt_token_ids"] == short_ids
