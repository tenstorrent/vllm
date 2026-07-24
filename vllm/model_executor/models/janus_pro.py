# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Janus-Pro multimodal processor for vLLM.

Processor-only module (no GPU model class). Tenstorrent's TT plugin maps the HF
architecture ``JanusForConditionalGeneration`` onto the tt-metal generator in
``models/experimental/janus_pro/tt/generator_vllm.py``; that generator registers
the classes defined here via ``MULTIMODAL_REGISTRY``.

Janus-Pro (``deepseek-community/Janus-Pro-7B``) uses a fixed-resolution vision
tower: the HF ``JanusProcessor`` resizes+pads every image to
``vision_config.image_size`` (384) and expands each ``<image_placeholder>`` into
``<begin_of_image>`` + ``num_image_tokens`` (576) × ``<image_placeholder>`` +
``<end_of_image>``. The vLLM processor therefore:

1. Calls the HF processor to produce ``pixel_values``.
2. Expands each single ``image_token`` placeholder into that same
   ``boi + N×image + eoi`` block (via ``PromptUpdateDetails.select_token_id`` so
   only the N inner image tokens receive vision embeddings).

Upstream status: vLLM has no native Janus support (tracking issues
vllm-project/vllm #12479 / #12492 / #12538 were closed unimplemented). This
file is the Tenstorrent-fork Step-0 precondition for serving Janus-Pro over
vLLM.
"""

from collections.abc import Mapping, Sequence

from transformers import BatchFeature
from transformers.models.janus.configuration_janus import JanusConfig
from transformers.models.janus.processing_janus import JanusProcessor

from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)


class JanusProProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(JanusConfig)

    def get_hf_processor(self, **kwargs: object):
        return self.ctx.get_hf_processor(JanusProcessor, **kwargs)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # Janus-Pro understanding path accepts an arbitrary number of images;
        # each contributes a fixed ``num_image_tokens`` block.
        return {"image": None}

    def get_num_image_tokens(self) -> int:
        # Sequence-length contribution of one image: boi + N image tokens + eoi.
        # Matches HF ``JanusProcessor.replace_image_token`` / the original
        # DeepSeek Janus ``add_image_token``. Only the N inner image tokens
        # receive vision embeddings (see ``_get_prompt_updates``).
        processor = self.get_hf_processor()
        return int(processor.num_image_tokens) + 2

    def get_image_size_with_most_features(self) -> ImageSize:
        # Fixed-resolution tower: the max-feature size is just the configured
        # square image_size (default 384).
        hf_config = self.get_hf_config()
        image_size = hf_config.vision_config.image_size
        if isinstance(image_size, (list, tuple)):
            height, width = int(image_size[0]), int(image_size[-1])
        else:
            height = width = int(image_size)
        return ImageSize(width=width, height=height)


class JanusProDummyInputsBuilder(BaseDummyInputsBuilder[JanusProProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        processor = self.info.get_hf_processor()
        return processor.image_token * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
        mm_processor_kwargs: Mapping[str, object] | None = None,
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        max_image_size = self.info.get_image_size_with_most_features()
        image_overrides = mm_options.get("image") if mm_options else None
        return {
            "image": self._get_dummy_images(
                width=max_image_size.width,
                height=max_image_size.height,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class JanusProMultiModalProcessor(BaseMultiModalProcessor[JanusProProcessingInfo]):
    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            # Text-only path: skip the vision processor entirely.
            tokenizer = self.info.get_tokenizer()
            return tokenizer(prompt, add_special_tokens=True, return_tensors="pt")

        # HF JanusProcessor does `for sample in text` when prepending the default
        # system prompt. A bare str is iterated as characters → batch size
        # len(prompt) and BaseMultiModalProcessor's
        # `(prompt_ids,) = input_ids.tolist()` raises. Always pass a 1-element list.
        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=[prompt], **mm_data),
            dict(**mm_kwargs, **tok_kwargs),
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # HF JanusProcessor yields ``pixel_values`` shaped [num_images, 3, H, W]
        # (one padded square per image). Batched-per-image is the convention
        # the TT generator's ``_extract_vision_images`` expects.
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        tokenizer = hf_processor.tokenizer

        # Resolve ids from the tokenizer so we stay consistent with the
        # ``<image_placeholder>`` / ``<begin_of_image>`` / ``<end_of_image>``
        # strings the chat template and HF processor use (config.image_token_id
        # for the community Janus-Pro-7B checkpoint is 100594 = placeholder).
        image_token_id = int(tokenizer.convert_tokens_to_ids(hf_processor.image_token))
        boi_token_id = int(
            tokenizer.convert_tokens_to_ids(hf_processor.image_start_token)
        )
        eoi_token_id = int(
            tokenizer.convert_tokens_to_ids(hf_processor.image_end_token)
        )
        num_image_tokens = int(hf_processor.num_image_tokens)

        # Official Janus expansion: boi + N x image + eoi. Only the N image
        # token positions get vision features via masked_scatter / multimodal
        # embedding injection; boi/eoi keep their text embeddings.
        full = (
            [boi_token_id]
            + [image_token_id] * num_image_tokens
            + [eoi_token_id]
        )
        replacement = PromptUpdateDetails.select_token_id(full, image_token_id)

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=replacement,
            )
        ]


# Aliases without the ``Pro`` suffix so the tt-metal generator's
# ``from vllm.model_executor.models.janus_pro import JanusMultiModalProcessor, ...``
# import (and any future upstream rename) stays stable.
JanusProcessingInfo = JanusProProcessingInfo
JanusDummyInputsBuilder = JanusProDummyInputsBuilder
JanusMultiModalProcessor = JanusProMultiModalProcessor
