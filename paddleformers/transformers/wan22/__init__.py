# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
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
"""Wan2.2 Video Generation Model - PaddlePaddle Native Implementation"""

import sys
from typing import TYPE_CHECKING

from ...utils.lazy_import import _LazyModule

import_structure = {
    "configuration": ["Wan22Config", "Wan22VAEConfig", "Wan22DiTConfig"],
    "modeling": [
        "Wan22PretrainedModel",
        "Wan22Model",
        "Wan22ForTextToVideo",
        "Wan22ForImageToVideo",
        "Wan22VideoOutput",
        "Wan22PatchEmbed3D",
        "Wan22PatchEmbedUnpatch3D",
    ],
    "modeling_paddle": [
        "Wan22VideoOutput",
        "Wan22RMSNorm",
        "Wan22FP32LayerNorm",
        "Wan22Attention",
        "Wan22FeedForward",
        "Wan22TransformerBlock",
        "Wan22Timesteps",
        "Wan22TimestepEmbedding",
        "Wan22DiTModel",
        "get_1d_rotary_pos_embed",
        "apply_rotary_emb",
    ],
    "vae": [
        "DecoderOutput",
        "AutoencoderKLOutput",
        "DiagonalGaussianDistribution",
        "WanCausalConv3d",
        "WanRMSNorm",
        "WanGroupNorm",
        "WanResample",
        "WanResnetBlock3D",
        "WanAttentionBlock",
        "WanEncoder3D",
        "WanDecoder3D",
        "Wan22VAEModel",
    ],
    "scheduler": [
        "SchedulerOutput",
        "Wan22FlowMatchScheduler",
        "Wan22UniPCScheduler",
    ],
    "pipeline": [
        "Wan22PipelineOutput",
        "Wan22BasePipeline",
        "Wan22TextToVideoPipeline",
        "Wan22ImageToVideoPipeline",
        "save_video",
        "SIZE_CONFIGS",
    ],
    "weight_converter": [
        "Wan22ModelLoader",
        "load_safetensors",
        "load_sharded_safetensors",
        "convert_diffusers_transformer_to_paddle",
        "load_vae_from_diffusers",
    ],
    "processor": [
        "Wan22Processor",
        "Wan22ImageProcessor",
        "Wan22ProcessorOutput",
    ],
    "video_processor": [
        "Wan22VideoProcessor",
        "Wan22VideoMetadata",
        "Wan22TrainingProcessor",
        "smart_resize",
    ],
}

if TYPE_CHECKING:
    from .configuration import *
    from .modeling import *
    from .modeling_paddle import *
    from .pipeline import *
    from .processor import *
    from .scheduler import *
    from .vae import *
    from .video_processor import *
    from .weight_converter import *

else:
    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        import_structure,
        module_spec=__spec__,
    )
