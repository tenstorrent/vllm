# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Coherence guard: the model must echo an exact sentence verbatim."""

from tests.tt.utils import RequestConfig, run_concurrent_batch


class TestCoherence:
    def test_coherence_verbatim_echo(self, tt_server, tt_model_name, max_batch_size):
        """Coherence guard: the model must echo an exact sentence verbatim.

        The other tests here only check structural/determinism properties (n,
        max_tokens, stop, seed, logprobs, penalties). A backend regression that
        corrupts the forward pass (e.g. a broken on-device decode trace) still
        emits deterministic, well-formed-but-garbage tokens, so all of those
        pass while the model is actually producing gibberish. Requiring a
        verbatim echo catches that class of bug directly: a model that cannot
        reproduce a simple sentence is not generating coherent text.
        """
        sentence = "The quick brown fox jumps over the lazy dog."
        configs = [
            RequestConfig(
                prompt=(
                    "Repeat the following sentence exactly, with no extra words, "
                    f"no quotes, and no commentary: {sentence}"
                ),
                max_tokens=32,
                temperature=0,
            )
        ]
        results = run_concurrent_batch(tt_server, tt_model_name, configs, use_chat=True)
        assert len(results) == len(configs)

        output_text = results[0]
        assert output_text is not None, "coherence guard got no output"
        assert sentence in output_text, (
            "Coherence guard failed: model did not echo the sentence verbatim "
            "(likely gibberish from a corrupted forward pass). Expected to find "
            f"{sentence!r} in output, got: {output_text!r}"
        )
