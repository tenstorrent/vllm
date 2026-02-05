# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2025 The vLLM team.
# Copyright 2025 The Qwen Team.
# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3VL model compatible with HuggingFace weights."""
from typing import Any, Optional

import numpy as np
import torch
from transformers.models.qwen2_vl import Qwen2VLImageProcessorFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers.models.qwen3_vl import (Qwen3VLProcessor,
                                          Qwen3VLVideoProcessor)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

from vllm.model_executor.models.qwen2_vl import Qwen2VLProcessingInfo
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.parse import ImageSize


class Qwen3VLProcessingInfo(Qwen2VLProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3VLConfig)

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_tokenizer(self):
        return self.ctx.tokenizer

    def get_image_processor(self,
                            **kwargs: object) -> Qwen2VLImageProcessorFast:
        return self.get_hf_processor(**kwargs).image_processor

    def get_video_processor(self, **kwargs: object) -> Qwen3VLVideoProcessor:
        return self.get_hf_processor(**kwargs).video_processor

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 2,
        do_resize: bool = True,
        image_processor: Optional[Qwen2VLImageProcessorFast],
    ) -> tuple[ImageSize, int]:
        if image_processor is None:
            image_processor = self.get_image_processor()

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        if do_resize:
            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=image_processor.size["shortest_edge"],
                max_pixels=image_processor.size["longest_edge"],
            )
            preprocessed_size = ImageSize(width=resized_width,
                                          height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width,
                                          height=image_height)

        padded_num_frames = num_frames + num_frames % temporal_patch_size

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens

    def _calculate_timestamps(self, indices: list[int] | torch.Tensor,
                              video_fps: float, merge_size: int):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            # don't update metadata's frames_indices directly
            indices = indices + [indices[-1]
                                 ] * (merge_size - len(indices) % merge_size)
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [(timestamps[i] + timestamps[i + merge_size - 1]) / 2
                      for i in range(0, len(timestamps), merge_size)]
        return timestamps

    def _get_video_second_idx(
            self,
            metadata: dict[str, Any],
            out_item: MultiModalKwargsItem,
            do_sample_frames: Optional[bool] = None,
            sampled_fps: Optional[float] = None) -> list[int]:
        video_processor = self.get_video_processor()
        merge_size = video_processor.merge_size
        indices = metadata["frames_indices"]

        # metadata["fps"] refers to the true fps of the input video.
        video_fps = metadata["fps"]
        if do_sample_frames is None:
            do_sample_frames = metadata.get("do_sample_frames", False)

        # If video frames are sampled in HF processor (instead of vLLM
        # video loader), we need to re-calculate the indices from original
        # metadata.
        if do_sample_frames:
            # here video_fps is the fps of the sampled video, and
            # metadata["fps"] refers to the fps of the original video.
            video_fps = sampled_fps if sampled_fps else video_processor.fps
            total_num_frames = metadata["total_num_frames"]
            num_frames = int(total_num_frames / metadata["fps"] * video_fps)
            num_frames = min(
                min(max(num_frames, video_processor.min_frames),
                    video_processor.max_frames), total_num_frames)
            indices = np.linspace(0, total_num_frames - 1,
                                  num_frames).round().astype(int).tolist()
        timestamps = self._calculate_timestamps(indices, video_fps, merge_size)
        return timestamps
