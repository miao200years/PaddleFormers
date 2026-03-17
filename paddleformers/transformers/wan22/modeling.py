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
"""Wan2.2 Video Generation Models - PaddlePaddle Native Implementation

This module provides PaddleFormers-style API for Wan2.2 video generation models.
Uses pure PaddlePaddle implementation following the same patterns as qwen2_5_vl.

Architecture:
- Wan22PretrainedModel: Base class inheriting from PretrainedModel (nn.Layer)
- Wan22Model: Core DiT model for video generation
- Wan22ForTextToVideo: Text-to-Video generation
- Wan22ForImageToVideo: Image-to-Video generation

Supported model tasks:
- t2v-5B: Text-to-Video 5B parameters
- ti2v-5B: Text/Image-to-Video 5B parameters
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import paddle
import paddle.nn as nn
from paddle import Tensor
from PIL import Image

from ..model_outputs import ModelOutput
from ..model_utils import PretrainedModel, register_base_model
from .configuration import Wan22Config
from .modeling_paddle import (
    Wan22FP32LayerNorm,
    Wan22TimestepEmbedding,
    Wan22Timesteps,
    Wan22TransformerBlock,
    get_1d_rotary_pos_embed,
)

# =============================================================================
# Output Classes
# =============================================================================


@dataclass
class Wan22VideoOutput(ModelOutput):
    """
    Output class for Wan2.2 video generation.

    Args:
        video: Generated video tensor (B, C, T, H, W) or numpy array
        latents: Final latent representation
        prompt: Input prompt used for generation
        size: Video size (width, height)
        frame_num: Number of frames
        seed: Random seed used
    """

    video: Optional[Tensor] = None
    latents: Optional[Tensor] = None
    prompt: Optional[str] = None
    size: Optional[Tuple[int, int]] = None
    frame_num: Optional[int] = None
    seed: Optional[int] = None


# =============================================================================
# Patch Embedding for Video
# =============================================================================


class Wan22PatchEmbed3D(nn.Layer):
    """
    3D Patch Embedding for video.

    Converts video (B, C, T, H, W) to patch embeddings (B, num_patches, embed_dim).
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        embed_dim: int = 3072,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv3D(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias_attr=bias,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Args:
            hidden_states: (B, C, T, H, W)

        Returns:
            (B, T' * H' * W', embed_dim)
        """
        hidden_states = self.proj(hidden_states)  # (B, embed_dim, T', H', W')
        hidden_states = hidden_states.flatten(2).transpose([0, 2, 1])  # (B, T'*H'*W', embed_dim)
        return hidden_states


class Wan22PatchEmbedUnpatch3D(nn.Layer):
    """
    Unpatch 3D embeddings back to video.
    """

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        out_channels: int = 16,
        embed_dim: int = 3072,
        bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.proj = nn.Linear(embed_dim, out_channels * patch_size[0] * patch_size[1] * patch_size[2], bias_attr=bias)

    def forward(self, hidden_states: Tensor, T: int, H: int, W: int) -> Tensor:
        """
        Args:
            hidden_states: (B, T' * H' * W', embed_dim)
            T, H, W: Original video dimensions

        Returns:
            (B, C, T, H, W)
        """
        batch_size = hidden_states.shape[0]
        p_t, p_h, p_w = self.patch_size

        # Project to patch values
        hidden_states = self.proj(hidden_states)  # (B, num_patches, C * p_t * p_h * p_w)

        # Reshape to patches
        T_p, H_p, W_p = T // p_t, H // p_h, W // p_w
        hidden_states = hidden_states.reshape([batch_size, T_p, H_p, W_p, self.out_channels, p_t, p_h, p_w])

        # Rearrange to video
        hidden_states = hidden_states.transpose([0, 4, 1, 5, 2, 6, 3, 7])
        hidden_states = hidden_states.reshape([batch_size, self.out_channels, T, H, W])

        return hidden_states


# =============================================================================
# Base Model Class
# =============================================================================


class Wan22PretrainedModel(PretrainedModel):
    """
    Base class for Wan2.2 models.

    Inherits from PretrainedModel (nn.Layer) to follow PaddleFormers conventions.
    All Wan22 models share this base class which provides common functionality
    like weight loading, saving, and configuration management.
    """

    config_class = Wan22Config
    base_model_prefix = "wan22"
    main_input_name = "hidden_states"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Wan22TransformerBlock"]
    _keys_to_ignore_on_load_unexpected = [r".*_diffusers.*"]

    # Size configurations for video generation
    SIZE_CONFIGS = {
        "1280*720": (1280, 720),
        "720*1280": (720, 1280),
        "1104*832": (1104, 832),
        "832*1104": (832, 1104),
        "960*960": (960, 960),
        "1280*704": (1280, 704),
        "704*1280": (704, 1280),
        "960*544": (960, 544),
        "544*960": (544, 960),
        "832*480": (832, 480),
        "480*832": (480, 832),
    }

    def _init_weights(self, layer):
        """Initialize the weights."""
        if isinstance(layer, nn.Linear):
            layer.weight.set_value(paddle.tensor.normal(mean=0.0, std=0.02, shape=layer.weight.shape))
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))
        elif isinstance(layer, nn.LayerNorm):
            layer.bias.set_value(paddle.zeros_like(layer.bias))
            layer.weight.set_value(paddle.ones_like(layer.weight))
        elif isinstance(layer, nn.Conv2D) or isinstance(layer, nn.Conv3D):
            layer.weight.set_value(paddle.tensor.normal(mean=0.0, std=0.02, shape=layer.weight.shape))
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))

    @staticmethod
    def _best_output_size(w: int, h: int, dw: int, dh: int, expected_area: int) -> Tuple[int, int]:
        """
        Calculate best output size maintaining aspect ratio.
        """
        ratio = w / h
        ow = (expected_area * ratio) ** 0.5
        oh = expected_area / ow

        ow1 = int(ow // dw * dw)
        oh1 = int(expected_area / ow1 // dh * dh)
        ratio1 = ow1 / oh1

        oh2 = int(oh // dh * dh)
        ow2 = int(expected_area / oh2 // dw * dw)
        ratio2 = ow2 / oh2

        if max(ratio / ratio1, ratio1 / ratio) < max(ratio / ratio2, ratio2 / ratio):
            return ow1, oh1
        else:
            return ow2, oh2

    @staticmethod
    def save_video(
        video: Union[Tensor, np.ndarray, List],
        output_path: str,
        fps: int = 24,
    ):
        """
        Save generated video to file.

        Args:
            video: Video tensor (B, C, T, H, W) or (T, H, W, C) numpy array
            output_path: Output file path
            fps: Frames per second
        """
        import imageio

        # Convert paddle tensor to numpy
        if isinstance(video, Tensor):
            video = video.numpy()

        # Handle different shapes
        if isinstance(video, np.ndarray):
            if video.ndim == 5:  # (B, C, T, H, W)
                video = video[0]  # Take first batch
                video = np.transpose(video, (1, 2, 3, 0))  # (T, H, W, C)
            elif video.ndim == 4:
                if video.shape[1] == 3:  # (T, C, H, W)
                    video = np.transpose(video, (0, 2, 3, 1))

            # Normalize to [0, 255]
            if video.dtype != np.uint8:
                if video.max() <= 1.0:
                    video = (video * 255).clip(0, 255).astype(np.uint8)
                else:
                    video = video.clip(0, 255).astype(np.uint8)

            imageio.mimsave(output_path, video, fps=fps)
            logging.info(f"Video saved to {output_path}")
            return

        # Handle list of PIL Images
        if isinstance(video, list) and len(video) > 0:
            if isinstance(video[0], Image.Image):
                frames = [np.array(frame) for frame in video]
                imageio.mimsave(output_path, frames, fps=fps)
                logging.info(f"Video saved to {output_path}")
                return

        raise ValueError(f"Unsupported video type: {type(video)}")


# =============================================================================
# Core Wan22 Model (DiT)
# =============================================================================


@register_base_model
class Wan22Model(Wan22PretrainedModel):
    """
    Wan2.2 Diffusion Transformer Model.

    This is the core model that performs the denoising process for video generation.
    It takes noisy latents and predicts the noise to be removed.
    """

    def __init__(self, config: Wan22Config):
        super().__init__(config)

        self.config = config
        dit_config = config.dit_config

        # Extract dimensions
        self.inner_dim = dit_config.dim
        self.num_heads = dit_config.num_heads
        self.head_dim = self.inner_dim // self.num_heads
        self.patch_size = dit_config.patch_size

        # Get cross attention dim (text encoder hidden size) from main config
        cross_attention_dim = config.dit_cross_attention_dim  # T5 hidden size (4096)
        in_channels = config.vae_config.z_dim if config.vae_config else config.dit_in_channels

        # Patch embedding
        self.patch_embed = Wan22PatchEmbed3D(
            patch_size=self.patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
        )

        # Time embedding
        self.time_proj = Wan22Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = Wan22TimestepEmbedding(256, self.inner_dim * 4)

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(cross_attention_dim, self.inner_dim * 4),
            nn.SiLU(),
            nn.Linear(self.inner_dim * 4, self.inner_dim * 6),
        )

        # Transformer blocks
        self.transformer_blocks = nn.LayerList(
            [
                Wan22TransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_heads,
                    attention_head_dim=self.head_dim,
                    cross_attention_dim=cross_attention_dim,
                    eps=dit_config.eps,
                    ff_inner_dim=dit_config.ffn_dim,
                    added_kv_proj_dim=config.dit_added_kv_proj_dim,  # For I2V image conditioning
                    cross_attention_dim_head=config.dit_cross_attention_dim_head,
                )
                for _ in range(dit_config.num_layers)
            ]
        )

        # Output projection
        self.norm_out = Wan22FP32LayerNorm(self.inner_dim, eps=dit_config.eps)
        self.unpatch = Wan22PatchEmbedUnpatch3D(
            patch_size=self.patch_size,
            out_channels=in_channels,
            embed_dim=self.inner_dim,
        )

        # AdaLN scale/shift for output
        self.scale_shift_table = self.create_parameter(
            shape=[2, self.inner_dim],
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        encoder_attention_mask: Optional[Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Tensor, Dict]:
        """
        Forward pass of the Wan22 DiT model.

        Args:
            hidden_states: Noisy latent tensor (B, C, T, H, W)
            encoder_hidden_states: Text embeddings (B, seq_len, text_dim)
            timestep: Diffusion timestep (B,)
            encoder_attention_mask: Attention mask for text
            return_dict: Whether to return a dict

        Returns:
            Predicted noise tensor (B, C, T, H, W)
        """
        batch_size, channels, num_frames, height, width = hidden_states.shape

        # Patch embedding
        hidden_states = self.patch_embed(hidden_states)  # (B, num_patches, inner_dim)

        # Time embedding
        t_emb = self.time_proj(timestep)
        t_emb = self.time_embedding(t_emb)  # (B, inner_dim * 4)

        # Text conditioning - produces (B, inner_dim * 6)
        text_emb = self.text_proj(encoder_hidden_states.mean(axis=1))  # (B, inner_dim * 6)

        # Combine time and text embeddings
        # Expand t_emb to match text_emb dimension by padding or repeating
        # timestep_proj needs to be (B, 6 * inner_dim) for transformer blocks
        timestep_proj = text_emb.clone()
        timestep_proj[..., : t_emb.shape[-1]] = timestep_proj[..., : t_emb.shape[-1]] + t_emb

        # Get rotary embeddings
        rotary_emb = get_1d_rotary_pos_embed(
            self.head_dim,
            hidden_states.shape[1],
            use_real=True,
        )

        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep_proj=timestep_proj,
                rotary_emb=rotary_emb,
            )

        # Output with AdaLN
        shift, scale = (
            self.scale_shift_table[None] + timestep_proj[:, : self.inner_dim * 2].reshape([-1, 2, self.inner_dim])
        ).unbind(axis=1)
        hidden_states = self.norm_out(hidden_states.astype("float32")) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = hidden_states.astype(self.unpatch.proj.weight.dtype)

        # Unpatch to video
        hidden_states = self.unpatch(hidden_states, num_frames, height, width)

        if return_dict:
            return {"sample": hidden_states}
        return hidden_states


# =============================================================================
# Video Generation Models
# =============================================================================


class Wan22ForTextToVideo(Wan22PretrainedModel):
    """
    Wan2.2 model for text-to-video generation.

    This model generates videos from text descriptions using a diffusion process.

    Example:
    ```python
    >>> from paddleformers.transformers import Wan22ForTextToVideo

    >>> model = Wan22ForTextToVideo.from_pretrained("path/to/wan22-t2v")

    >>> output = model.generate(
    ...     prompt="A cat walks on the grass",
    ...     height=480,
    ...     width=832,
    ...     num_frames=49,
    ... )

    >>> model.save_video(output.video, "output.mp4")
    ```
    """

    def __init__(self, config: Wan22Config):
        super().__init__(config)

        # Core DiT model
        self.transformer = Wan22Model(config)

        # VAE will be loaded separately or use existing implementation
        self.vae = None

        # Text encoder will be loaded separately (T5)
        self.text_encoder = None

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        **kwargs,
    ) -> Dict:
        """Forward pass through the transformer."""
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            **kwargs,
        )

    @paddle.no_grad()
    def generate(
        self,
        prompt: str,
        height: int = 480,
        width: int = 832,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: str = "",
        seed: int = -1,
        **kwargs,
    ) -> Wan22VideoOutput:
        """
        Generate video from text prompt.

        Args:
            prompt: Text description for video generation
            height: Video height (must be divisible by 16)
            width: Video width (must be divisible by 16)
            num_frames: Number of frames (must be 4n+1)
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt
            seed: Random seed (-1 for random)

        Returns:
            Wan22VideoOutput containing generated video
        """
        # This is a placeholder - full implementation requires:
        # 1. Text encoding with T5
        # 2. Diffusion sampling loop
        # 3. VAE decoding
        raise NotImplementedError(
            "Full generation pipeline requires VAE and text encoder. "
            "Use the pipeline module for complete generation."
        )


class Wan22ForImageToVideo(Wan22PretrainedModel):
    """
    Wan2.2 model for image-to-video generation.

    This model generates videos from an input image and text description.

    Example:
    ```python
    >>> from paddleformers.transformers import Wan22ForImageToVideo
    >>> from PIL import Image

    >>> model = Wan22ForImageToVideo.from_pretrained("path/to/wan22-i2v")

    >>> image = Image.open("input.jpg")
    >>> output = model.generate(
    ...     image=image,
    ...     prompt="The scene comes alive",
    ...     height=480,
    ...     width=832,
    ...     num_frames=49,
    ... )

    >>> model.save_video(output.video, "output.mp4")
    ```
    """

    def __init__(self, config: Wan22Config):
        super().__init__(config)

        # Core DiT model (same architecture, different weights)
        self.transformer = Wan22Model(config)

        # Image encoder for first frame conditioning
        self.image_encoder = None

        # VAE
        self.vae = None

        # Text encoder
        self.text_encoder = None

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        image_embeds: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        """Forward pass through the transformer with image conditioning."""
        # Concatenate image embeddings with text embeddings for I2V
        if image_embeds is not None:
            encoder_hidden_states = paddle.concat([image_embeds, encoder_hidden_states], axis=1)

        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            **kwargs,
        )

    @paddle.no_grad()
    def generate(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        height: int = 480,
        width: int = 832,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        negative_prompt: str = "",
        seed: int = -1,
        **kwargs,
    ) -> Wan22VideoOutput:
        """
        Generate video from image and text prompt.

        Args:
            image: Input image (PIL Image or path)
            prompt: Text description
            height: Video height
            width: Video width
            num_frames: Number of frames
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt
            seed: Random seed

        Returns:
            Wan22VideoOutput containing generated video
        """
        raise NotImplementedError(
            "Full generation pipeline requires VAE, image encoder, and text encoder. "
            "Use the pipeline module for complete generation."
        )


__all__ = [
    "Wan22PretrainedModel",
    "Wan22Model",
    "Wan22ForTextToVideo",
    "Wan22ForImageToVideo",
    "Wan22VideoOutput",
    "Wan22PatchEmbed3D",
    "Wan22PatchEmbedUnpatch3D",
]
