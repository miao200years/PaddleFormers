# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
"""
Wan2.2 Video Generation Models - PaddlePaddle Native Implementation

This module provides a pure PaddlePaddle implementation of Wan2.2 video generation models,
following the same patterns as other PaddleFormers models (e.g., qwen3_vl).

Architecture:
- Wan22VAE: 3D VAE for video encoding/decoding
- Wan22DiT: Diffusion Transformer for video generation
- Wan22TextEncoder: T5-based text encoder
- Wan22Pipeline: End-to-end video generation pipeline
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

from .configuration import Wan22Config


# Base class for models - simplified version to avoid import chain
class Wan22BaseModel(nn.Layer):
    """Base class for Wan22 models"""

    config_class = None
    base_model_prefix = ""

    def __init__(self, config):
        super().__init__()
        self.config = config


# =============================================================================
# Output Classes
# =============================================================================


@dataclass
class Wan22VideoOutput:
    """Output class for Wan2.2 video generation"""

    video: Optional[Tensor] = None  # (B, C, T, H, W)
    latents: Optional[Tensor] = None
    prompt: Optional[str] = None
    size: Optional[Tuple[int, int]] = None
    frame_num: Optional[int] = None
    seed: Optional[int] = None


# =============================================================================
# Basic Building Blocks
# =============================================================================


class Wan22RMSNorm(nn.Layer):
    """RMS Normalization layer"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[hidden_size],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype("float32")
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * paddle.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).astype(input_dtype)


class Wan22FP32LayerNorm(nn.Layer):
    """FP32 Layer Normalization"""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = self.create_parameter(
                shape=[normalized_shape],
                default_initializer=nn.initializer.Constant(1.0),
            )
            self.bias = self.create_parameter(
                shape=[normalized_shape],
                default_initializer=nn.initializer.Constant(0.0),
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        x = x.astype("float32")
        x = F.layer_norm(x, [self.normalized_shape], self.weight, self.bias, self.eps)
        return x.astype(orig_dtype)


class Wan22SiLU(nn.Layer):
    """SiLU activation (Swish)"""

    def forward(self, x: Tensor) -> Tensor:
        return F.silu(x)


class Wan22GELU(nn.Layer):
    """GELU activation with approximate option"""

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        if self.approximate == "tanh":
            return F.gelu(x, approximate=True)
        return F.gelu(x)


# =============================================================================
# Timestep Embedding
# =============================================================================


class Wan22Timesteps(nn.Layer):
    """Sinusoidal timestep embeddings"""

    def __init__(self, num_channels: int, flip_sin_to_cos: bool = False, downscale_freq_shift: float = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: Tensor) -> Tensor:
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * paddle.arange(0, half_dim, dtype="float32")
        exponent = exponent / (half_dim - self.downscale_freq_shift)

        emb = paddle.exp(exponent)
        emb = timesteps[:, None].astype("float32") * emb[None, :]

        if self.flip_sin_to_cos:
            emb = paddle.concat([paddle.cos(emb), paddle.sin(emb)], axis=-1)
        else:
            emb = paddle.concat([paddle.sin(emb), paddle.cos(emb)], axis=-1)

        if self.num_channels % 2 == 1:
            emb = F.pad(emb, [0, 1, 0, 0])

        return emb


class Wan22TimestepEmbedding(nn.Layer):
    """MLP for timestep embedding projection"""

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)

        if act_fn == "silu":
            self.act = Wan22SiLU()
        elif act_fn == "gelu":
            self.act = Wan22GELU()
        else:
            self.act = Wan22SiLU()

        self.linear_2 = nn.Linear(time_embed_dim, out_dim or time_embed_dim)

    def forward(self, sample: Tensor) -> Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


# =============================================================================
# Rotary Position Embedding
# =============================================================================


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[Tensor, int],
    theta: float = 10000.0,
    use_real: bool = False,
    freqs_dtype: str = "float32",
) -> Tuple[Tensor, Tensor]:
    """
    Get 1D rotary position embeddings.

    Args:
        dim: Embedding dimension
        pos: Position indices or maximum position
        theta: Base for frequency computation
        use_real: Whether to return real-valued tensors
        freqs_dtype: Dtype for frequency tensors

    Returns:
        Tuple of (cos, sin) embeddings
    """
    if isinstance(pos, int):
        pos = paddle.arange(pos, dtype=freqs_dtype)

    freqs = 1.0 / (theta ** (paddle.arange(0, dim, 2, dtype=freqs_dtype) / dim))

    # Outer product: (seq_len, dim//2)
    freqs = paddle.outer(pos.astype(freqs_dtype), freqs)

    if use_real:
        freqs_cos = freqs.cos().tile([1, 2])
        freqs_sin = freqs.sin().tile([1, 2])
        return freqs_cos, freqs_sin
    else:
        freqs_cis = paddle.complex(freqs.cos(), freqs.sin())
        return freqs_cis


def apply_rotary_emb(
    hidden_states: Tensor,
    freqs_cos: Tensor,
    freqs_sin: Tensor,
) -> Tensor:
    """Apply rotary embedding to hidden states

    Args:
        hidden_states: (batch, seq, heads, head_dim) or (batch, heads, seq, head_dim)
        freqs_cos: (seq, head_dim) or (seq, head_dim/2)
        freqs_sin: (seq, head_dim) or (seq, head_dim/2)
    """
    # Get dimensions

    # Adjust freqs shape to match hidden_states
    # freqs shape: (seq, dim) -> needs to broadcast with (batch, seq, heads, dim)
    if freqs_cos.ndim == 2:
        # Shape: (seq, dim) -> (1, seq, 1, dim)
        freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2)
        freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    # Apply rotation using complex multiplication pattern
    # Split into pairs for rotation
    x1 = hidden_states[..., 0::2]
    x2 = hidden_states[..., 1::2]

    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 0::2]  # Changed from 1::2 to 0::2

    # Apply rotation: out = x * cos + rotate(x) * sin
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    # Interleave back
    out = paddle.stack([out1, out2], axis=-1).flatten(-2)

    return out.astype(hidden_states.dtype)


# =============================================================================
# Attention Components
# =============================================================================


class Wan22Attention(nn.Layer):
    """
    Multi-head attention with support for:
    - Self-attention and cross-attention
    - Rotary position embeddings
    - QK normalization
    - Added KV projections for I2V
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
        cross_attention_dim: Optional[int] = None,  # Input dim for K/V in cross-attention
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.cross_attention_dim = cross_attention_dim or dim  # Default to dim for self-attention

        # KV dimension (different for cross-attention)
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        # QKV projections - K and V use cross_attention_dim as input
        self.to_q = nn.Linear(dim, self.inner_dim, bias_attr=True)
        self.to_k = nn.Linear(self.cross_attention_dim, self.kv_inner_dim, bias_attr=True)
        self.to_v = nn.Linear(self.cross_attention_dim, self.kv_inner_dim, bias_attr=True)

        # QK normalization
        self.norm_q = Wan22RMSNorm(self.inner_dim // heads, eps=eps)
        self.norm_k = Wan22RMSNorm(self.kv_inner_dim // heads, eps=eps)

        # Output projection - may need to handle different kv dimensions
        if self.kv_inner_dim != self.inner_dim:
            # If kv_dim != q_dim, add a projection to match
            self.kv_to_inner = nn.Linear(self.kv_inner_dim, self.inner_dim, bias_attr=True)
        else:
            self.kv_to_inner = None

        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim, bias_attr=True),
            nn.Dropout(dropout),
        )

        # Additional KV projections for I2V (image context)
        if added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.kv_inner_dim, bias_attr=True)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.kv_inner_dim, bias_attr=True)
            self.norm_added_k = Wan22RMSNorm(self.kv_inner_dim // heads, eps=eps)
        else:
            self.add_k_proj = None
            self.add_v_proj = None
            self.norm_added_k = None

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Handle image context for I2V
        encoder_hidden_states_img = None
        if self.add_k_proj is not None and encoder_hidden_states is not None:
            # 512 is the context length of the text encoder
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        # Compute Q, K, V
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        # Apply QK normalization
        query = self.norm_q(query.reshape([batch_size, -1, self.heads, self.dim_head]))
        key = self.norm_k(key.reshape([batch_size, -1, self.heads, self.kv_inner_dim // self.heads]))

        # Reshape for attention
        query = query.reshape([batch_size, -1, self.heads, self.dim_head])
        key = key.reshape([batch_size, -1, self.heads, self.kv_inner_dim // self.heads])
        value = value.reshape([batch_size, -1, self.heads, self.kv_inner_dim // self.heads])

        # Apply rotary embeddings
        if rotary_emb is not None:
            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # Transpose for attention: (B, heads, seq, dim_head)
        query = query.transpose([0, 2, 1, 3])
        key = key.transpose([0, 2, 1, 3])
        value = value.transpose([0, 2, 1, 3])

        # Handle I2V image attention
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = self.add_k_proj(encoder_hidden_states_img)
            value_img = self.add_v_proj(encoder_hidden_states_img)

            key_img = self.norm_added_k(key_img.reshape([batch_size, -1, self.heads, self.kv_inner_dim // self.heads]))
            key_img = key_img.reshape([batch_size, -1, self.heads, self.kv_inner_dim // self.heads])
            value_img = value_img.reshape([batch_size, -1, self.heads, self.kv_inner_dim // self.heads])

            key_img = key_img.transpose([0, 2, 1, 3])
            value_img = value_img.transpose([0, 2, 1, 3])

            # Scaled dot-product attention for image
            hidden_states_img = F.scaled_dot_product_attention(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )
            hidden_states_img = hidden_states_img.transpose([0, 2, 1, 3])
            hidden_states_img = hidden_states_img.reshape([batch_size, -1, self.inner_dim])

        # Manual scaled dot-product attention (more compatible with cross-attention)
        scale = 1.0 / math.sqrt(query.shape[-1])
        attn_weights = paddle.matmul(query, key.transpose([0, 1, 3, 2])) * scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, axis=-1)
        hidden_states = paddle.matmul(attn_weights, value)

        # Reshape back - hidden_states shape: (B, heads, seq, kv_dim_head)
        hidden_states = hidden_states.transpose([0, 2, 1, 3])
        hidden_states = hidden_states.reshape([batch_size, -1, self.kv_inner_dim])

        # Project to inner_dim if kv_inner_dim is different
        if self.kv_to_inner is not None:
            hidden_states = self.kv_to_inner(hidden_states)

        # Add image attention if present
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        # Output projection
        hidden_states = self.to_out(hidden_states)

        return hidden_states


# =============================================================================
# Feed Forward Network
# =============================================================================


class Wan22FeedForward(nn.Layer):
    """Feed-forward network with GELU activation"""

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
    ):
        super().__init__()

        inner_dim = hidden_dim or dim * mult

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim, bias_attr=True),
            Wan22GELU(approximate="tanh" if "approximate" in activation_fn else "none"),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias_attr=True),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.net(hidden_states)


# =============================================================================
# Transformer Block
# =============================================================================


class Wan22TransformerBlock(nn.Layer):
    """
    Wan2.2 Transformer block with:
    - Self-attention with rotary embeddings
    - Cross-attention for text conditioning
    - Feed-forward network
    - Adaptive layer norm (AdaLN) for timestep conditioning
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int = None,
        eps: float = 1e-5,
        dropout: float = 0.0,
        ff_inner_dim: Optional[int] = None,
        added_kv_proj_dim: Optional[int] = None,
        cross_attention_dim_head: Optional[int] = None,
    ):
        super().__init__()

        self.dim = dim

        # Self-attention (K,V from hidden_states, so cross_attention_dim=dim)
        self.norm1 = Wan22FP32LayerNorm(dim, eps=eps)
        self.attn1 = Wan22Attention(
            dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            eps=eps,
            dropout=dropout,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim=dim,  # Self-attention: K/V input dim = dim
        )

        # Cross-attention
        self.norm2 = Wan22FP32LayerNorm(dim, eps=eps)
        self.attn2 = Wan22Attention(
            dim=dim,
            heads=num_attention_heads,
            dim_head=cross_attention_dim_head or attention_head_dim,
            eps=eps,
            dropout=dropout,
            cross_attention_dim_head=cross_attention_dim_head,
            cross_attention_dim=cross_attention_dim,  # For K/V projection from encoder_hidden_states
        )

        # Feed-forward
        self.norm3 = Wan22FP32LayerNorm(dim, eps=eps)
        self.ff = Wan22FeedForward(
            dim=dim,
            hidden_dim=ff_inner_dim,
            dropout=dropout,
        )

        # AdaLN modulation
        self.scale_shift_table = self.create_parameter(
            shape=[6, dim],
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep_proj: Tensor,
        rotary_emb: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        # Get modulation parameters from timestep
        # timestep_proj: (B, 6*dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep_proj.reshape([-1, 6, self.dim])
        ).unbind(axis=1)

        # Self-attention with AdaLN
        norm_hidden_states = (
            self.norm1(hidden_states.astype("float32")) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        )
        norm_hidden_states = norm_hidden_states.astype(hidden_states.dtype)
        attn_output = self.attn1(norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = hidden_states + gate_msa[:, None] * attn_output

        # Cross-attention
        norm_hidden_states = self.norm2(hidden_states.astype("float32")).astype(hidden_states.dtype)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # Feed-forward with AdaLN
        norm_hidden_states = (
            self.norm3(hidden_states.astype("float32")) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        )
        norm_hidden_states = norm_hidden_states.astype(hidden_states.dtype)
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate_mlp[:, None] * ff_output

        return hidden_states


# =============================================================================
# DiT (Diffusion Transformer) Model
# =============================================================================


class Wan22DiTModel(Wan22BaseModel):
    """
    Wan2.2 Diffusion Transformer (DiT) for video generation.

    This is the core denoising model that takes noisy latents and predicts noise.
    """

    config_class = Wan22Config
    base_model_prefix = "wan22_dit"

    def __init__(self, config: Wan22Config):
        super().__init__(config)

        self.config = config

        # Patch embedding
        self.patch_size = config.dit_patch_size
        self.in_channels = config.dit_in_channels
        inner_dim = config.dit_num_attention_heads * config.dit_attention_head_dim

        # Patch projection
        self.patch_proj = nn.Conv2D(
            self.in_channels,
            inner_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias_attr=True,
        )

        # Time embedding
        self.time_proj = Wan22Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = Wan22TimestepEmbedding(256, inner_dim * 6)  # 6 for scale/shift/gate pairs

        # Text projection (must match time_embedding output for addition)
        self.text_proj = nn.Linear(config.dit_cross_attention_dim, inner_dim * 6)

        # Transformer blocks
        self.transformer_blocks = nn.LayerList(
            [
                Wan22TransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=config.dit_num_attention_heads,
                    attention_head_dim=config.dit_attention_head_dim,
                    cross_attention_dim=config.dit_cross_attention_dim,
                    eps=config.dit_norm_eps,
                    ff_inner_dim=config.dit_ff_inner_dim,
                    added_kv_proj_dim=config.dit_added_kv_proj_dim,
                    cross_attention_dim_head=config.dit_cross_attention_dim_head,
                )
                for _ in range(config.dit_num_layers)
            ]
        )

        # Output layers
        self.norm_out = Wan22FP32LayerNorm(inner_dim, eps=config.dit_norm_eps)
        self.proj_out = nn.Linear(inner_dim, self.patch_size**2 * self.in_channels)

        # Scale/shift for output
        self.scale_shift_table = self.create_parameter(
            shape=[2, inner_dim],
            default_initializer=nn.initializer.Normal(std=0.02),
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        return_dict: bool = True,
    ) -> Union[Tensor, Dict]:
        """
        Forward pass of the DiT model.

        Args:
            hidden_states: Noisy latent tensor (B, C, T, H, W)
            encoder_hidden_states: Text embeddings from T5
            timestep: Diffusion timestep

        Returns:
            Predicted noise
        """
        batch_size, channels, num_frames, height, width = hidden_states.shape

        # Reshape to (B*T, C, H, W) for patch embedding
        hidden_states = hidden_states.transpose([0, 2, 1, 3, 4])
        hidden_states = hidden_states.reshape([batch_size * num_frames, channels, height, width])

        # Patch embedding
        hidden_states = self.patch_proj(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose([0, 2, 1])

        # Reshape back to (B, T*H*W, D)
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.reshape([batch_size, num_frames * seq_len, -1])

        # Time embedding
        t_emb = self.time_proj(timestep)
        t_emb = self.time_embedding(t_emb)

        # Text embedding
        text_emb = self.text_proj(encoder_hidden_states.mean(axis=1))

        # Combined conditioning
        timestep_proj = t_emb + text_emb

        # Get rotary embeddings
        # (simplified - full implementation would compute proper 3D rotary embeddings)
        rotary_emb = get_1d_rotary_pos_embed(
            self.config.dit_attention_head_dim,
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

        # Output projection - extract shift/scale from timestep_proj (first 2*dim of 6*dim)
        inner_dim = self.scale_shift_table.shape[1]
        timestep_proj_out = timestep_proj[:, : 2 * inner_dim].reshape([-1, 2, inner_dim])
        shift, scale = (self.scale_shift_table[None] + timestep_proj_out).unbind(axis=1)
        hidden_states = self.norm_out(hidden_states.astype("float32")) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = hidden_states.astype(self.proj_out.weight.dtype)
        hidden_states = self.proj_out(hidden_states)

        # Reshape to video
        hidden_states = hidden_states.reshape(
            [batch_size, num_frames, height // self.patch_size, width // self.patch_size, -1]
        )
        hidden_states = hidden_states.transpose([0, 4, 1, 2, 3])

        # Unpatchify
        hidden_states = hidden_states.reshape(
            [
                batch_size,
                channels,
                num_frames,
                height // self.patch_size,
                self.patch_size,
                width // self.patch_size,
                self.patch_size,
            ]
        )
        hidden_states = hidden_states.transpose([0, 1, 2, 3, 5, 4, 6])
        hidden_states = hidden_states.reshape([batch_size, channels, num_frames, height, width])

        if return_dict:
            return {"sample": hidden_states}
        return hidden_states


# =============================================================================
# Placeholder classes for VAE and Pipeline (to be implemented)
# =============================================================================


class Wan22VAEEncoder(nn.Layer):
    """VAE Encoder - converts video to latent space (TODO: full implementation)"""

    def __init__(self, config: Wan22Config):
        super().__init__()
        self.config = config
        # Placeholder - full implementation needed

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("VAE Encoder not yet implemented")


class Wan22VAEDecoder(nn.Layer):
    """VAE Decoder - converts latents back to video (TODO: full implementation)"""

    def __init__(self, config: Wan22Config):
        super().__init__()
        self.config = config
        # Placeholder - full implementation needed

    def forward(self, z: Tensor) -> Tensor:
        raise NotImplementedError("VAE Decoder not yet implemented")


class Wan22VAE(Wan22BaseModel):
    """
    Wan2.2 VAE for video encoding/decoding.

    Full implementation pending - requires ~1400 lines of code.
    """

    config_class = Wan22Config
    base_model_prefix = "wan22_vae"

    def __init__(self, config: Wan22Config):
        super().__init__(config)
        self.encoder = Wan22VAEEncoder(config)
        self.decoder = Wan22VAEDecoder(config)

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)


class Wan22TextEncoder(Wan22BaseModel):
    """
    Wan2.2 Text Encoder (T5-based).

    Full implementation pending - can potentially reuse existing T5 implementation.
    """

    config_class = Wan22Config
    base_model_prefix = "wan22_text_encoder"

    def __init__(self, config: Wan22Config):
        super().__init__(config)
        # Placeholder - reuse existing T5 or implement UMT5

    def forward(self, input_ids: Tensor, attention_mask: Tensor = None) -> Tensor:
        raise NotImplementedError("Text Encoder not yet implemented")


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Output
    "Wan22VideoOutput",
    # Building blocks
    "Wan22RMSNorm",
    "Wan22FP32LayerNorm",
    "Wan22Attention",
    "Wan22FeedForward",
    "Wan22TransformerBlock",
    "Wan22Timesteps",
    "Wan22TimestepEmbedding",
    # Models
    "Wan22DiTModel",
    "Wan22VAE",
    "Wan22TextEncoder",
    # Utilities
    "get_1d_rotary_pos_embed",
    "apply_rotary_emb",
]
