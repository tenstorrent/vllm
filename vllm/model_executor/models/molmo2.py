# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Molmo2 multimodal processor for the reference vllm fork.

Supports both image and video modalities:
  - Video: calls HF Molmo2VideoProcessor → pixel_values_videos [n_frames, 729, 588]
           + video_token_pooling [N_pooled, 9] (k_pool=9, pooling=[3,3])
  - Image: calls HF Molmo2ImageProcessor → pixel_values [n_crops, 729, 588]
           + image_token_pooling [N_pooled, 4] (k_pool=4, pooling=[2,2])

Both paths produce tensors that TtMolmo2Model.forward_prefill() already handles.
"""

from collections.abc import Mapping, Sequence
from typing import Optional

import torch
from transformers import BatchFeature

from vllm.multimodal import MULTIMODAL_REGISTRY  # noqa: F401
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)
try:
    from vllm.multimodal.profiling import BaseDummyInputsBuilder, MultiModalDataDict
except ImportError:
    from vllm.multimodal.processing import BaseDummyInputsBuilder
    from vllm.inputs import MultiModalDataDict

# Molmo2 token IDs (from allenai/Molmo2-8B config)
_IMAGE_PATCH_ID = 151938        # image_patch_id — replaces both <|video|> and <|image|>
_VIDEO_PLACEHOLDER_ID = 151945  # <|video|> token in the prompt
_IMAGE_PLACEHOLDER_ID = 151941  # <|image|> token in the prompt

_DEFAULT_N_OUT_PER_FRAME = 81   # video: 9×9 pool = 81 tokens per frame (pooling=[3,3])
_DEFAULT_N_OUT_PER_IMAGE_CROP = 196  # image: 14×14 pool = 196 tokens per crop (pooling=[2,2])
_DEFAULT_IMAGE_SIZE = 378
_DEFAULT_PATCH_SIZE = 14


class Molmo2ProcessingInfo(BaseProcessingInfo):
    """ProcessingInfo for Molmo2: supports image and video."""

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # Both video and image supported.
        # TT override (_TT_Molmo2ProcessingInfo) raises image limit to 23.
        return {"video": 1, "image": 1}

    def get_data_parser(self):
        """Use video_needs_metadata=True so vLLM preserves fps/duration metadata.

        With this flag, VideoProcessorItems stores (frames, metadata_dict) tuples
        instead of raw arrays. _call_hf_processor uses the metadata to enable
        do_sample_frames=True in the HF Molmo2VideoProcessor, reproducing the
        same frame sampling as the demo (processor(videos=file_path, ...)).
        """
        from vllm.multimodal.parse import MultiModalDataParser
        return MultiModalDataParser(
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_num_image_tokens(self, *, image_width: int, image_height: int, **kwargs) -> int:
        return (_DEFAULT_IMAGE_SIZE // _DEFAULT_PATCH_SIZE) ** 2  # 729

    def get_image_size_with_most_features(self):
        from vllm.multimodal.parse import ImageSize
        return ImageSize(width=_DEFAULT_IMAGE_SIZE, height=_DEFAULT_IMAGE_SIZE)


class Molmo2DummyInputsBuilder(BaseDummyInputsBuilder[Molmo2ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<|video|>" * mm_counts.get("video", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options=None,
    ) -> MultiModalDataDict:
        import numpy as np
        num_videos = mm_counts.get("video", 0)
        num_images = mm_counts.get("image", 0)
        result: MultiModalDataDict = {}
        if num_videos:
            # Provide (array, metadata_dict) tuples since video_needs_metadata=True.
            # Include frames_indices so HF processor can compute timestamps.
            import numpy as _np
            n_dummy_frames = 4
            dummy_fps = 2.0
            dummy_array = _np.random.randint(64, 192, (n_dummy_frames, _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, 3), dtype=_np.uint8)
            dummy_metadata = {
                "fps": dummy_fps,
                "total_num_frames": n_dummy_frames,
                "duration": n_dummy_frames / dummy_fps,
                "video_backend": "numpy",
                "frames_indices": list(range(n_dummy_frames)),
                "do_sample_frames": False,
            }
            result["video"] = [(dummy_array, dummy_metadata) for _ in range(num_videos)]
        if num_images:
            result["image"] = self._get_dummy_images(
                width=_DEFAULT_IMAGE_SIZE, height=_DEFAULT_IMAGE_SIZE, num_images=num_images
            )
        return result


class Molmo2MultiModalProcessor(BaseMultiModalProcessor[Molmo2ProcessingInfo]):
    """Calls HF Molmo2Processor to produce pre-processed video patches."""

    def _apply_hf_processor_tokens_only(self, prompt_tokens):
        """Keep tokens as-is — Molmo2 processor handles BOS internally."""
        return prompt_tokens

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Call HF Molmo2Processor with full prompt + video together.

        This matches the demo's behavior: processor(text=fmt, videos=str(vpath), ...)
        The processor produces ALL Molmo2-specific tokens (patches, frame markers,
        column/row separators) in one pass, giving the correct input_ids.
        """
        mm_data = dict(mm_data)
        hf_processor = self.info.get_hf_processor(**mm_kwargs)

        # ---- Image path (ImageProcessorItems.get_processor_data() → {"images": [...]}) ----
        images = mm_data.pop("images", [])
        if images:
            # Only pass text + images — NOT remaining mm_data which may contain "videos"
            # (HF Molmo2Processor cannot process images and videos simultaneously).
            combined_out = self.info.ctx.call_hf_processor(
                hf_processor,
                {"text": prompt, "images": images},
                {},
            )
            result = dict(combined_out)
            # HF image processor already provides image_num_crops as a per-image tensor
            # (e.g. tensor([2, 3]) for 2 images with 2 and 3 crops respectively).
            # DO NOT overwrite it — the per-image counts are needed for flat_from_sizes batching.
            img_num_crops = result.get("image_num_crops")
            if img_num_crops is not None:
                crops_t = (
                    img_num_crops
                    if isinstance(img_num_crops, torch.Tensor)
                    else torch.tensor(img_num_crops)
                )
                # Pooled patches per image = crops_per_image * pooled_per_crop (196 for [2,2] pool)
                result["image_num_pooled_patches"] = crops_t.long() * _DEFAULT_N_OUT_PER_IMAGE_CROP
            return BatchFeature(result)

        try:
            from transformers.video_utils import VideoMetadata as _VideoMetadata
        except ImportError:
            _VideoMetadata = None

        videos = mm_data.pop("videos", [])

        if videos:
            # With video_backend="molmo2" and video_needs_metadata=True, vLLM provides
            # (frames, metadata_dict) where metadata_dict includes:
            #   - fps, duration, total_num_frames (original video properties)
            #   - frames_indices (which frames were extracted — key for HF timestamps)
            #   - do_sample_frames=False (frames already sampled at correct rate)
            # Molmo2VideoBackend implements the same sampling as the HF processor,
            # so frames + frames_indices match exactly what processor(videos=path) produces.

            for item in videos:
                video_array, metadata_dict = item if (isinstance(item, (list, tuple)) and len(item) == 2) else (item, None)
                if video_array is None:
                    continue

                n_frames = video_array.shape[0] if hasattr(video_array, "shape") else 4

                if _VideoMetadata is not None and isinstance(metadata_dict, dict):
                    # Build VideoMetadata with frames_indices so HF processor can compute
                    # correct timestamps — identical to the demo path.
                    metadata = _VideoMetadata(
                        total_num_frames=metadata_dict.get("total_num_frames", n_frames),
                        fps=metadata_dict.get("fps", 2.0),
                        duration=metadata_dict.get("duration", n_frames / 2.0),
                        video_backend=metadata_dict.get("video_backend", "numpy"),
                        frames_indices=metadata_dict.get("frames_indices"),
                    )
                    do_sample_frames = metadata_dict.get("do_sample_frames", False)
                else:
                    # Fallback for dummy/unknown metadata
                    fps = 2.0
                    if _VideoMetadata is not None:
                        metadata = _VideoMetadata(total_num_frames=n_frames, fps=fps, duration=n_frames/fps, video_backend="numpy")
                    else:
                        metadata = {"fps": fps}
                    do_sample_frames = False

                # Call the HF processor with BOTH the full prompt AND video together.
                # This produces ALL Molmo2 tokens (patches + frame markers + separators),
                # matching exactly what the demo does: processor(text=fmt, videos=path, ...)
                combined_out = self.info.ctx.call_hf_processor(
                    hf_processor,
                    {"text": prompt, "videos": [[video_array]], "video_metadata": [[metadata]], **mm_data},
                    {"do_sample_frames": do_sample_frames},
                )
                result = dict(combined_out)
                # Add size fields for _get_mm_fields_config
                pv = result.get("pixel_values_videos")
                pool = result.get("video_token_pooling")
                if pv is not None:
                    result["video_num_crops"] = torch.tensor([pv.shape[0]])
                if pool is not None:
                    result["video_num_pooled_patches"] = torch.tensor([pool.shape[0]])

                # Store the full video token sequence (including frame markers <im_start>/<im_end>)
                # so _get_prompt_updates can use it as the PromptReplacement content.
                # This gives the server the same S=2701 (with frame markers) as the direct demo,
                # vs S=2481 from PromptReplacement (patches only). Key for ~98% accuracy.
                combined_ids = result.get("input_ids")
                if combined_ids is not None:
                    ids_1d = combined_ids.squeeze(0) if combined_ids.dim() > 1 else combined_ids
                    result["video_input_ids"] = ids_1d.long()
                    result["video_num_input_tokens"] = torch.tensor([ids_1d.shape[0]])
                return BatchFeature(result)

        # Text-only: just process the prompt without any video
        text_out = self.info.ctx.call_hf_processor(hf_processor, {"text": prompt, **mm_data}, {})
        return BatchFeature(dict(text_out))

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        video_num_crops = hf_inputs.get("video_num_crops", torch.empty(0))
        video_num_pooled = hf_inputs.get("video_num_pooled_patches", torch.empty(0))
        video_num_tokens = hf_inputs.get("video_num_input_tokens", torch.empty(0))
        config = {}
        if video_num_crops.numel() > 0:
            config["pixel_values_videos"] = MultiModalFieldConfig.flat_from_sizes(
                "video", video_num_crops
            )
        if video_num_pooled.numel() > 0:
            config["video_token_pooling"] = MultiModalFieldConfig.flat_from_sizes(
                "video", video_num_pooled
            )
        if video_num_crops.numel() > 0:
            config["video_num_crops"] = MultiModalFieldConfig.batched("video")
        if video_num_pooled.numel() > 0:
            config["video_num_pooled_patches"] = MultiModalFieldConfig.batched("video")
        if video_num_tokens.numel() > 0:
            config["video_input_ids"] = MultiModalFieldConfig.flat_from_sizes(
                "video", video_num_tokens
            )
            config["video_num_input_tokens"] = MultiModalFieldConfig.batched("video")

        # ---- Image fields (from native image path) ----
        # image_num_crops is a per-image tensor from the HF image processor (e.g. [2, 3] for
        # 2 images with 2 and 3 crops). flat_from_sizes batches pixel_values accordingly.
        img_num_crops_raw = hf_inputs.get("image_num_crops")
        img_num_pooled_raw = hf_inputs.get("image_num_pooled_patches")
        if img_num_crops_raw is not None:
            img_num_crops = (
                img_num_crops_raw
                if isinstance(img_num_crops_raw, torch.Tensor)
                else torch.tensor(img_num_crops_raw)
            )
            if img_num_crops.numel() > 0:
                config["pixel_values"] = MultiModalFieldConfig.flat_from_sizes(
                    "image", img_num_crops
                )
                config["image_num_crops"] = MultiModalFieldConfig.batched("image")
        if img_num_pooled_raw is not None:
            img_num_pooled = (
                img_num_pooled_raw
                if isinstance(img_num_pooled_raw, torch.Tensor)
                else torch.tensor(img_num_pooled_raw)
            )
            if img_num_pooled.numel() > 0:
                config["image_token_pooling"] = MultiModalFieldConfig.flat_from_sizes(
                    "image", img_num_pooled
                )
                config["image_num_pooled_patches"] = MultiModalFieldConfig.batched("image")
        return config

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Locate video tokens in the (already-updated) prompt token sequence.

        When mm_processor_cache_gb=0 (no cache), the HF Molmo2Processor is called
        with the FULL text+video together, producing input_ids that already have
        image_patch_id tokens (no <|video|> placeholder left). We set target to the
        entire image_patch_id run so vLLM's _find_mm_placeholders can locate them
        in the updated input_ids, setting the correct PlaceholderRange.

        When the cache is active (cached path), the HF processor only produces
        pixel_values/pooling, and PromptReplacement replaces <|video|> normally.
        """

        def get_replacement(item_idx: int) -> PromptUpdateDetails:
            video_items = out_mm_kwargs.get("video", [])
            if item_idx < len(video_items):
                item_data = video_items[item_idx].data

                # Prefer the full video token sequence (with frame markers <im_start>/<im_end>)
                # stored in video_input_ids — this gives S matching the demo exactly (~2701 for
                # 30 frames) vs patches-only (~2481). Gives ~98% accuracy vs ~68%.
                vid_ids_elem = item_data.get("video_input_ids")
                if vid_ids_elem is not None:
                    vid_ids_t = vid_ids_elem.data if hasattr(vid_ids_elem, "data") else vid_ids_elem
                    if isinstance(vid_ids_t, torch.Tensor) and vid_ids_t.numel() > 0:
                        return PromptUpdateDetails.select_token_id(
                            vid_ids_t.long().tolist(),
                            embed_token_id=_IMAGE_PATCH_ID,
                        )

                # Fallback: patches-only replacement (cached path without video_input_ids)
                n_pooled = 4 * _DEFAULT_N_OUT_PER_FRAME  # safe default for 4-frame dummy
                np_elem = item_data.get("video_num_pooled_patches")
                if np_elem is not None:
                    np_tensor = np_elem.data if hasattr(np_elem, "data") else np_elem
                    if isinstance(np_tensor, torch.Tensor):
                        n_pooled = int(np_tensor.item())
                else:
                    pool_elem = item_data.get("video_token_pooling")
                    if pool_elem is not None:
                        pool_tensor = pool_elem.data if hasattr(pool_elem, "data") else pool_elem
                        if isinstance(pool_tensor, torch.Tensor):
                            n_pooled = pool_tensor.shape[0]
            else:
                n_pooled = 4 * _DEFAULT_N_OUT_PER_FRAME

            return PromptUpdateDetails.select_token_id(
                [_IMAGE_PATCH_ID] * n_pooled,
                embed_token_id=_IMAGE_PATCH_ID,
            )

        def get_image_replacement(item_idx: int) -> PromptUpdateDetails:
            image_items = out_mm_kwargs.get("image", [])
            n_pooled = _DEFAULT_N_OUT_PER_IMAGE_CROP  # 196 per crop fallback
            if item_idx < len(image_items):
                item_data = image_items[item_idx].data
                np_elem = item_data.get("image_num_pooled_patches")
                if np_elem is not None:
                    np_tensor = np_elem.data if hasattr(np_elem, "data") else np_elem
                    if isinstance(np_tensor, torch.Tensor) and np_tensor.numel() > 0:
                        n_pooled = int(np_tensor.item())
            return PromptUpdateDetails.select_token_id(
                [_IMAGE_PATCH_ID] * n_pooled,
                embed_token_id=_IMAGE_PATCH_ID,
            )

        return [
            PromptReplacement(
                modality="video",
                target=[_VIDEO_PLACEHOLDER_ID],
                replacement=get_replacement,
            ),
            PromptReplacement(
                modality="image",
                target=[_IMAGE_PLACEHOLDER_ID],
                replacement=get_image_replacement,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(
    Molmo2MultiModalProcessor,
    info=Molmo2ProcessingInfo,
    dummy_inputs=Molmo2DummyInputsBuilder,
)
class Molmo2ForConditionalGeneration(torch.nn.Module):
    """Stub class registered in vllm's model registry for architecture lookup.

    The actual computation is done by TTMolmo2ForConditionalGeneration
    (models.demos.molmo2.tt.generator_vllm.Molmo2ForConditionalGeneration).
    The TT platform prepends 'TT' to the architecture name, so this stub
    is only used for vllm's initial model inspection.
    """

    # Required for vllm to treat this as a multimodal model
    supports_multimodal: bool = True

    def __init__(self, vllm_config=None, prefix: str = "") -> None:
        super().__init__()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.float()

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, **kwargs):
        return input_ids.float()

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs):
        return hidden_states
