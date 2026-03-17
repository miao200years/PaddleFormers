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
Wan2.2 VAE (Variational Autoencoder) - PaddlePaddle Native Implementation

This module provides a 3D VAE for video encoding/decoding, supporting:
- Causal 3D convolutions for temporal consistency
- Feature caching for efficient inference
- Temporal and spatial downsampling/upsampling
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import Tensor

# Cache size for temporal processing
CACHE_T = 2


# =============================================================================
# Output Classes
# =============================================================================


@dataclass
class DecoderOutput:
    """VAE Decoder output"""

    sample: Tensor


@dataclass
class AutoencoderKLOutput:
    """VAE output with latent distribution"""

    latent_dist: "DiagonalGaussianDistribution"


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution for VAE latent space"""

    def __init__(self, parameters: Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = paddle.chunk(parameters, 2, axis=1)
        self.logvar = paddle.clip(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = paddle.exp(0.5 * self.logvar)
        self.var = paddle.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = paddle.zeros_like(self.mean)

    def sample(self, generator=None) -> Tensor:
        if generator is not None:
            sample = paddle.randn(self.mean.shape, dtype=self.mean.dtype)
        else:
            sample = paddle.randn(self.mean.shape, dtype=self.mean.dtype)
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None) -> Tensor:
        if self.deterministic:
            return paddle.to_tensor([0.0])

        if other is None:
            return 0.5 * paddle.sum(paddle.pow(self.mean, 2) + self.var - 1.0 - self.logvar, axis=[1, 2, 3, 4])
        else:
            return 0.5 * paddle.sum(
                paddle.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                axis=[1, 2, 3, 4],
            )

    def mode(self) -> Tensor:
        return self.mean


# =============================================================================
# 3D Downsampling/Upsampling Layers
# =============================================================================


class AvgDown3D(nn.Layer):
    """3D Average Pooling Downsampling"""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: Tensor) -> Tensor:
        # Pad temporal dimension
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        if pad_t > 0:
            x = F.pad(x, [0, 0, 0, 0, pad_t, 0], data_format="NCDHW")

        B, C, T, H, W = x.shape

        # Reshape for downsampling
        x = x.reshape(
            [
                B,
                C,
                T // self.factor_t,
                self.factor_t,
                H // self.factor_s,
                self.factor_s,
                W // self.factor_s,
                self.factor_s,
            ]
        )
        x = x.transpose([0, 1, 3, 5, 7, 2, 4, 6])
        x = x.reshape([B, C * self.factor, T // self.factor_t, H // self.factor_s, W // self.factor_s])
        x = x.reshape(
            [B, self.out_channels, self.group_size, T // self.factor_t, H // self.factor_s, W // self.factor_s]
        )
        x = x.mean(axis=2)

        return x


class DupUp3D(nn.Layer):
    """3D Duplication Upsampling"""

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: Tensor, first_chunk: bool = False) -> Tensor:
        # Repeat channels
        x = x.tile([1, self.repeats, 1, 1, 1])

        B = x.shape[0]
        T = x.shape[2]
        H = x.shape[3]
        W = x.shape[4]

        # Reshape for upsampling
        x = x.reshape([B, self.out_channels, self.factor_t, self.factor_s, self.factor_s, T, H, W])
        x = x.transpose([0, 1, 5, 2, 6, 3, 7, 4])
        x = x.reshape([B, self.out_channels, T * self.factor_t, H * self.factor_s, W * self.factor_s])

        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]

        return x


# =============================================================================
# Causal 3D Convolution
# =============================================================================


class WanCausalConv3d(nn.Layer):
    """
    Causal 3D Convolution with temporal padding for autoregressive generation.
    Ensures that outputs only depend on past and current frames.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        self.orig_padding = padding

        # Causal padding: pad only on the past side for temporal dimension
        # Format: (W_left, W_right, H_top, H_bottom, T_front, T_back)
        self._padding = (padding[2], padding[2], padding[1], padding[1], 2 * padding[0], 0)

        self.conv = nn.Conv3D(in_channels, out_channels, kernel_size, stride, padding=0, bias_attr=True)
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, x: Tensor, cache_x: Optional[Tensor] = None) -> Tensor:
        padding = list(self._padding)

        # Use cached frames if available
        if cache_x is not None and self._padding[4] > 0:
            x = paddle.concat([cache_x, x], axis=2)
            padding[4] -= cache_x.shape[2]

        # Apply causal padding
        if any(p > 0 for p in padding):
            # Pad format for 3D: (left, right, top, bottom, front, back)
            x = F.pad(x, padding, data_format="NCDHW")

        return self.conv(x)


# =============================================================================
# Normalization Layers
# =============================================================================


class WanRMSNorm(nn.Layer):
    """RMS Normalization for video data"""

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True, bias: bool = False):
        super().__init__()

        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = [dim] + list(broadcastable_dims) if channel_first else [dim]

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = self.create_parameter(shape=shape, default_initializer=nn.initializer.Constant(1.0))

        if bias:
            self.bias = self.create_parameter(shape=shape, default_initializer=nn.initializer.Constant(0.0))
        else:
            self.bias = 0.0

    def forward(self, x: Tensor) -> Tensor:
        norm_dim = 1 if self.channel_first else -1
        x_normalized = F.normalize(x, p=2, axis=norm_dim)
        return x_normalized * self.scale * self.gamma + self.bias


class WanGroupNorm(nn.Layer):
    """Group Normalization wrapper"""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels, epsilon=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.group_norm(x)


# =============================================================================
# Resampling Modules
# =============================================================================


class WanUpsample(nn.Layer):
    """Upsampling layer that preserves dtype"""

    def __init__(self, scale_factor: Tuple[float, float], mode: str = "nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: Tensor) -> Tensor:
        orig_dtype = x.dtype
        x = x.astype("float32")
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x.astype(orig_dtype)


class WanResample(nn.Layer):
    """2D/3D Resampling module with optional temporal processing"""

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None):
        super().__init__()
        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest"),
                nn.Conv2D(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest"),
                nn.Conv2D(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.Pad2D([0, 1, 0, 1]),
                nn.Conv2D(dim, dim, 3, stride=2),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.Pad2D([0, 1, 0, 1]),
                nn.Conv2D(dim, dim, 3, stride=2),
            )
            self.time_conv = WanCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x: Tensor, feat_cache: List = None, feat_idx: List = None) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]

        B, C, T, H, W = x.shape

        # Handle 3D upsampling with temporal expansion
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape([B, 2, C, T, H, W])
                    x = paddle.stack([x[:, 0], x[:, 1]], axis=3)
                    x = x.reshape([B, C, T * 2, H, W])

        # Apply 2D resampling frame by frame
        T = x.shape[2]
        x = x.transpose([0, 2, 1, 3, 4]).reshape([B * T, C, H, W])
        x = self.resample(x)
        x = x.reshape([B, T, x.shape[1], x.shape[2], x.shape[3]]).transpose([0, 2, 1, 3, 4])

        # Handle 3D downsampling with temporal reduction
        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = x[:, :, -1:, :, :].clone()
                    feat_idx[0] += 1

        return x


# =============================================================================
# ResNet Block
# =============================================================================


class WanResnetBlock3D(nn.Layer):
    """3D ResNet block with group normalization"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
    ):
        super().__init__()

        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = WanGroupNorm(groups, in_channels, eps=eps)
        self.conv1 = WanCausalConv3d(in_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))

        self.norm2 = WanGroupNorm(groups, out_channels, eps=eps)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = WanCausalConv3d(out_channels, out_channels, (3, 3, 3), padding=(1, 1, 1))

        self.nonlinearity = nn.Silu()

        # Shortcut connection
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = WanCausalConv3d(in_channels, out_channels, (1, 1, 1))
            else:
                self.nin_shortcut = WanCausalConv3d(in_channels, out_channels, (1, 1, 1))
        else:
            self.conv_shortcut = None
            self.nin_shortcut = None

    def forward(self, x: Tensor, feat_cache: List = None, feat_idx: List = None) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]

        residual = x

        # First conv block
        h = self.norm1(x)
        h = self.nonlinearity(h)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_h = h[:, :, -CACHE_T:, :, :].clone()
            h = self.conv1(h, feat_cache[idx])
            feat_cache[idx] = cache_h
            feat_idx[0] += 1
        else:
            h = self.conv1(h)

        # Second conv block
        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_h = h[:, :, -CACHE_T:, :, :].clone()
            h = self.conv2(h, feat_cache[idx])
            feat_cache[idx] = cache_h
            feat_idx[0] += 1
        else:
            h = self.conv2(h)

        # Shortcut
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                residual = self.conv_shortcut(residual)
            else:
                residual = self.nin_shortcut(residual)

        return h + residual


# =============================================================================
# Attention Block
# =============================================================================


class WanAttentionBlock(nn.Layer):
    """Self-attention block for VAE"""

    def __init__(self, channels: int, num_head_channels: int = None, groups: int = 32, eps: float = 1e-6):
        super().__init__()

        self.channels = channels
        self.num_heads = channels // num_head_channels if num_head_channels else 1

        self.group_norm = WanGroupNorm(groups, channels, eps=eps)
        self.query = nn.Conv1D(channels, channels, 1)
        self.key = nn.Conv1D(channels, channels, 1)
        self.value = nn.Conv1D(channels, channels, 1)
        self.proj_attn = nn.Conv1D(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        B, C, T, H, W = x.shape

        # Normalize
        x = self.group_norm(x)

        # Reshape to (B*T, C, H*W)
        x = x.transpose([0, 2, 1, 3, 4]).reshape([B * T, C, H * W])

        # QKV projections
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Multi-head attention
        head_dim = C // self.num_heads
        q = q.reshape([B * T, self.num_heads, head_dim, H * W]).transpose([0, 1, 3, 2])
        k = k.reshape([B * T, self.num_heads, head_dim, H * W]).transpose([0, 1, 3, 2])
        v = v.reshape([B * T, self.num_heads, head_dim, H * W]).transpose([0, 1, 3, 2])

        # Scaled dot-product attention
        scale = head_dim**-0.5
        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * scale
        attn = F.softmax(attn, axis=-1)

        # Apply attention
        out = paddle.matmul(attn, v)
        out = out.transpose([0, 1, 3, 2]).reshape([B * T, C, H * W])

        # Output projection
        out = self.proj_attn(out)

        # Reshape back to video
        out = out.reshape([B, T, C, H, W]).transpose([0, 2, 1, 3, 4])

        return out + residual


# =============================================================================
# Encoder
# =============================================================================


class WanEncoder3D(nn.Layer):
    """VAE Encoder for video"""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        base_channels: int = 160,
        num_res_blocks: int = 2,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        temporal_downsample: Tuple[bool, ...] = (False, True, True),
        groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initial convolution
        self.conv_in = WanCausalConv3d(in_channels, base_channels, (3, 3, 3), padding=(1, 1, 1))

        # Downsampling blocks
        self.down_blocks = nn.LayerList()

        in_ch = base_channels
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult

            # ResNet blocks
            for _ in range(num_res_blocks):
                block = WanResnetBlock3D(in_ch, out_ch, groups=groups, dropout=dropout)
                self.down_blocks.append(block)
                in_ch = out_ch

            # Downsampling
            if i < len(channel_multipliers) - 1:
                if temporal_downsample[i]:
                    resample = WanResample(in_ch, "downsample3d")
                else:
                    resample = WanResample(in_ch, "downsample2d")
                self.down_blocks.append(resample)

        # Middle blocks
        self.mid_block_1 = WanResnetBlock3D(in_ch, in_ch, groups=groups, dropout=dropout)
        self.mid_attn = WanAttentionBlock(in_ch, num_head_channels=in_ch // 4, groups=groups)
        self.mid_block_2 = WanResnetBlock3D(in_ch, in_ch, groups=groups, dropout=dropout)

        # Output
        self.norm_out = WanGroupNorm(groups, in_ch)
        self.conv_out = WanCausalConv3d(in_ch, 2 * out_channels, (3, 3, 3), padding=(1, 1, 1))

        self.nonlinearity = nn.Silu()

    def forward(self, x: Tensor, feat_cache: List = None, feat_idx: List = None) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]

        # Initial conv
        h = self.conv_in(x)

        # Down blocks
        for block in self.down_blocks:
            if isinstance(block, WanResnetBlock3D):
                h = block(h, feat_cache, feat_idx)
            else:
                h = block(h, feat_cache, feat_idx)

        # Middle
        h = self.mid_block_1(h, feat_cache, feat_idx)
        h = self.mid_attn(h)
        h = self.mid_block_2(h, feat_cache, feat_idx)

        # Output
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)

        return h


# =============================================================================
# Decoder
# =============================================================================


class WanDecoder3D(nn.Layer):
    """VAE Decoder for video"""

    def __init__(
        self,
        in_channels: int = 48,
        out_channels: int = 3,
        base_channels: int = 160,
        num_res_blocks: int = 2,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        temporal_upsample: Tuple[bool, ...] = (False, True, True),
        groups: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Initial channels (from highest multiplier)
        block_in_ch = base_channels * channel_multipliers[-1]

        # Initial convolution
        self.conv_in = WanCausalConv3d(in_channels, block_in_ch, (3, 3, 3), padding=(1, 1, 1))

        # Middle blocks
        self.mid_block_1 = WanResnetBlock3D(block_in_ch, block_in_ch, groups=groups, dropout=dropout)
        self.mid_attn = WanAttentionBlock(block_in_ch, num_head_channels=block_in_ch // 4, groups=groups)
        self.mid_block_2 = WanResnetBlock3D(block_in_ch, block_in_ch, groups=groups, dropout=dropout)

        # Upsampling blocks
        self.up_blocks = nn.LayerList()

        in_ch = block_in_ch
        reversed_mults = list(reversed(channel_multipliers))
        reversed_temp_up = list(reversed(temporal_upsample))

        for i, mult in enumerate(reversed_mults):
            out_ch = base_channels * mult

            # ResNet blocks (one extra for first level)
            for j in range(num_res_blocks + 1):
                block = WanResnetBlock3D(in_ch, out_ch, groups=groups, dropout=dropout)
                self.up_blocks.append(block)
                in_ch = out_ch

            # Upsampling (except last level)
            if i < len(reversed_mults) - 1:
                upsample_out = base_channels * reversed_mults[i + 1]
                if reversed_temp_up[i]:
                    resample = WanResample(in_ch, "upsample3d", upsample_out_dim=upsample_out)
                else:
                    resample = WanResample(in_ch, "upsample2d", upsample_out_dim=upsample_out)
                self.up_blocks.append(resample)
                in_ch = upsample_out

        # Output
        self.norm_out = WanGroupNorm(groups, in_ch)
        self.conv_out = WanCausalConv3d(in_ch, out_channels, (3, 3, 3), padding=(1, 1, 1))

        self.nonlinearity = nn.Silu()

    def forward(self, z: Tensor, feat_cache: List = None, feat_idx: List = None) -> Tensor:
        if feat_idx is None:
            feat_idx = [0]

        # Initial conv
        h = self.conv_in(z)

        # Middle
        h = self.mid_block_1(h, feat_cache, feat_idx)
        h = self.mid_attn(h)
        h = self.mid_block_2(h, feat_cache, feat_idx)

        # Up blocks
        for block in self.up_blocks:
            if isinstance(block, WanResnetBlock3D):
                h = block(h, feat_cache, feat_idx)
            else:
                h = block(h, feat_cache, feat_idx)

        # Output
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)

        return h


# =============================================================================
# Full VAE Model
# =============================================================================


class Wan22VAEModel(nn.Layer):
    """
    Wan2.2 Variational Autoencoder for video encoding/decoding.

    This VAE uses 3D causal convolutions to ensure temporal consistency
    and supports efficient inference via feature caching.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 48,
        base_channels: int = 160,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 4),
        temporal_downsample: Tuple[bool, ...] = (False, True, True),
        num_res_blocks: int = 2,
        groups: int = 32,
        dropout: float = 0.0,
        scaling_factor: float = 0.18215,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor

        # Encoder
        self.encoder = WanEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            groups=groups,
            dropout=dropout,
        )

        # Decoder
        self.decoder = WanDecoder3D(
            in_channels=latent_channels,
            out_channels=in_channels,
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_upsample=temporal_downsample,
            groups=groups,
            dropout=dropout,
        )

        # Quant/post-quant convolutions
        self.quant_conv = nn.Conv3D(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv3D(latent_channels, latent_channels, 1)

    def encode(self, x: Tensor, return_dict: bool = True) -> Union[AutoencoderKLOutput, DiagonalGaussianDistribution]:
        """Encode video to latent distribution"""
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)

        if return_dict:
            return AutoencoderKLOutput(latent_dist=posterior)
        return posterior

    def decode(self, z: Tensor, return_dict: bool = True) -> Union[DecoderOutput, Tensor]:
        """Decode latents to video"""
        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if return_dict:
            return DecoderOutput(sample=dec)
        return dec

    def forward(self, x: Tensor, sample_posterior: bool = True, return_dict: bool = True, generator=None):
        """Full forward pass: encode -> sample -> decode"""
        posterior = self.encode(x).latent_dist

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        dec = self.decode(z).sample

        if return_dict:
            return {"sample": dec, "latent_dist": posterior}
        return dec


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Output classes
    "DecoderOutput",
    "AutoencoderKLOutput",
    "DiagonalGaussianDistribution",
    # Building blocks
    "AvgDown3D",
    "DupUp3D",
    "WanCausalConv3d",
    "WanRMSNorm",
    "WanGroupNorm",
    "WanUpsample",
    "WanResample",
    "WanResnetBlock3D",
    "WanAttentionBlock",
    # Encoder/Decoder
    "WanEncoder3D",
    "WanDecoder3D",
    # Full model
    "Wan22VAEModel",
]
