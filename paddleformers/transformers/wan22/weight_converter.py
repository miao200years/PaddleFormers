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
Weight Converter for Wan2.2 Models

This module provides utilities to load pretrained weights from Diffusers format
and convert them to PaddlePaddle native format.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import paddle


def _convert_key_pytorch_to_paddle(key: str) -> str:
    """
    Convert PyTorch state dict key to PaddlePaddle format.

    Main differences:
    - PyTorch: model.layer.weight, model.layer.bias
    - PaddlePaddle: model.layer.weight, model.layer.bias (same for most cases)
    - Some layers may need renaming
    """
    # Most keys are the same, but handle specific cases
    conversions = [
        # LayerNorm
        (r"\.layer_norm\.weight", ".layer_norm.weight"),
        (r"\.layer_norm\.bias", ".layer_norm.bias"),
        # GroupNorm
        (r"\.group_norm\.weight", ".group_norm.weight"),
        (r"\.group_norm\.bias", ".group_norm.bias"),
        # Attention
        (r"\.to_q\.weight", ".to_q.weight"),
        (r"\.to_k\.weight", ".to_k.weight"),
        (r"\.to_v\.weight", ".to_v.weight"),
        (r"\.to_out\.0\.weight", ".to_out.0.weight"),
        (r"\.to_out\.0\.bias", ".to_out.0.bias"),
    ]

    new_key = key
    for pattern, replacement in conversions:
        new_key = re.sub(pattern, replacement, new_key)

    return new_key


def _convert_tensor_pytorch_to_paddle(tensor: np.ndarray, key: str) -> np.ndarray:
    """
    Convert PyTorch tensor format to PaddlePaddle format.

    Main differences:
    - Conv2D: PyTorch (out, in, H, W) -> PaddlePaddle (out, in, H, W) [same]
    - Conv3D: PyTorch (out, in, D, H, W) -> PaddlePaddle (out, in, D, H, W) [same]
    - Linear: PyTorch (out, in) -> PaddlePaddle (in, out) [transpose needed]
    """
    # Check if this is a linear layer weight that needs transposing
    if ".weight" in key and tensor.ndim == 2:
        # Linear layer weight: transpose from (out, in) to (in, out)
        # Note: PaddlePaddle's Linear expects (in, out) format
        return tensor.T

    return tensor


def load_safetensors(file_path: str) -> Dict[str, np.ndarray]:
    """Load weights from a safetensors file."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise ImportError("Please install safetensors: pip install safetensors")

    tensors = {}
    with safe_open(file_path, framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    return tensors


def load_sharded_safetensors(
    model_path: str, index_file: str = "diffusion_pytorch_model.safetensors.index.json"
) -> Dict[str, np.ndarray]:
    """Load weights from sharded safetensors files."""
    index_path = os.path.join(model_path, index_file)

    if not os.path.exists(index_path):
        # Try single file
        single_file = os.path.join(model_path, "diffusion_pytorch_model.safetensors")
        if os.path.exists(single_file):
            return load_safetensors(single_file)
        raise FileNotFoundError(f"No safetensors file found in {model_path}")

    with open(index_path, "r") as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Group weights by file
    file_to_keys: Dict[str, List[str]] = {}
    for key, filename in weight_map.items():
        if filename not in file_to_keys:
            file_to_keys[filename] = []
        file_to_keys[filename].append(key)

    # Load all weights
    all_tensors = {}
    for filename, keys in file_to_keys.items():
        file_path = os.path.join(model_path, filename)
        print(f"Loading {filename}...")
        tensors = load_safetensors(file_path)
        for key in keys:
            if key in tensors:
                all_tensors[key] = tensors[key]

    return all_tensors


def convert_diffusers_transformer_to_paddle(
    diffusers_state_dict: Dict[str, np.ndarray],
    paddle_model: paddle.nn.Layer,
    strict: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    Convert Diffusers transformer weights to PaddlePaddle model.

    Args:
        diffusers_state_dict: State dict loaded from Diffusers safetensors
        paddle_model: Target PaddlePaddle model
        strict: Whether to require all weights to match

    Returns:
        Tuple of (loaded_keys, missing_keys)
    """
    paddle_state_dict = paddle_model.state_dict()

    loaded_keys = []
    missing_keys = []
    unexpected_keys = []

    # Key mapping from Diffusers to our implementation
    key_mapping = {
        # Patch embedding
        "patch_embedding.proj.weight": "patch_proj.weight",
        "patch_embedding.proj.bias": "patch_proj.bias",
        # Time embedding
        "time_embedding.timestep_embedder.linear_1.weight": "time_embedding.linear_1.weight",
        "time_embedding.timestep_embedder.linear_1.bias": "time_embedding.linear_1.bias",
        "time_embedding.timestep_embedder.linear_2.weight": "time_embedding.linear_2.weight",
        "time_embedding.timestep_embedder.linear_2.bias": "time_embedding.linear_2.bias",
        # Text projection
        "text_embedding.text_embedder.linear_1.weight": "text_proj.weight",
        "text_embedding.text_embedder.linear_1.bias": "text_proj.bias",
        # Output
        "norm_out.linear.weight": "norm_out.weight",
        "norm_out.linear.bias": "norm_out.bias",
        "proj_out.weight": "proj_out.weight",
        "proj_out.bias": "proj_out.bias",
    }

    # Build transformer block mapping
    def build_block_mapping(block_idx: int) -> Dict[str, str]:
        prefix = f"blocks.{block_idx}"
        paddle_prefix = f"transformer_blocks.{block_idx}"

        return {
            # Self-attention
            f"{prefix}.attn1.to_q.weight": f"{paddle_prefix}.attn1.to_q.weight",
            f"{prefix}.attn1.to_q.bias": f"{paddle_prefix}.attn1.to_q.bias",
            f"{prefix}.attn1.to_k.weight": f"{paddle_prefix}.attn1.to_k.weight",
            f"{prefix}.attn1.to_k.bias": f"{paddle_prefix}.attn1.to_k.bias",
            f"{prefix}.attn1.to_v.weight": f"{paddle_prefix}.attn1.to_v.weight",
            f"{prefix}.attn1.to_v.bias": f"{paddle_prefix}.attn1.to_v.bias",
            f"{prefix}.attn1.to_out.0.weight": f"{paddle_prefix}.attn1.to_out.0.weight",
            f"{prefix}.attn1.to_out.0.bias": f"{paddle_prefix}.attn1.to_out.0.bias",
            f"{prefix}.attn1.norm_q.weight": f"{paddle_prefix}.attn1.norm_q.weight",
            f"{prefix}.attn1.norm_k.weight": f"{paddle_prefix}.attn1.norm_k.weight",
            # Cross-attention
            f"{prefix}.attn2.to_q.weight": f"{paddle_prefix}.attn2.to_q.weight",
            f"{prefix}.attn2.to_q.bias": f"{paddle_prefix}.attn2.to_q.bias",
            f"{prefix}.attn2.to_k.weight": f"{paddle_prefix}.attn2.to_k.weight",
            f"{prefix}.attn2.to_k.bias": f"{paddle_prefix}.attn2.to_k.bias",
            f"{prefix}.attn2.to_v.weight": f"{paddle_prefix}.attn2.to_v.weight",
            f"{prefix}.attn2.to_v.bias": f"{paddle_prefix}.attn2.to_v.bias",
            f"{prefix}.attn2.to_out.0.weight": f"{paddle_prefix}.attn2.to_out.0.weight",
            f"{prefix}.attn2.to_out.0.bias": f"{paddle_prefix}.attn2.to_out.0.bias",
            f"{prefix}.attn2.norm_q.weight": f"{paddle_prefix}.attn2.norm_q.weight",
            f"{prefix}.attn2.norm_k.weight": f"{paddle_prefix}.attn2.norm_k.weight",
            # Norms
            f"{prefix}.norm1.weight": f"{paddle_prefix}.norm1.weight",
            f"{prefix}.norm1.bias": f"{paddle_prefix}.norm1.bias",
            f"{prefix}.norm2.weight": f"{paddle_prefix}.norm2.weight",
            f"{prefix}.norm2.bias": f"{paddle_prefix}.norm2.bias",
            f"{prefix}.norm3.weight": f"{paddle_prefix}.norm3.weight",
            f"{prefix}.norm3.bias": f"{paddle_prefix}.norm3.bias",
            # FFN
            f"{prefix}.ffn.net.0.weight": f"{paddle_prefix}.ff.net.0.weight",
            f"{prefix}.ffn.net.0.bias": f"{paddle_prefix}.ff.net.0.bias",
            f"{prefix}.ffn.net.2.weight": f"{paddle_prefix}.ff.net.3.weight",
            f"{prefix}.ffn.net.2.bias": f"{paddle_prefix}.ff.net.3.bias",
            # Scale shift table
            f"{prefix}.scale_shift_table": f"{paddle_prefix}.scale_shift_table",
        }

    # Add block mappings for all layers
    num_layers = 30  # Default for 5B model
    for i in range(num_layers):
        key_mapping.update(build_block_mapping(i))

    # Convert and load weights
    for diffusers_key, tensor in diffusers_state_dict.items():
        # Try direct mapping
        paddle_key = key_mapping.get(diffusers_key)

        if paddle_key is None:
            # Try automatic conversion
            paddle_key = _convert_key_pytorch_to_paddle(diffusers_key)

        if paddle_key in paddle_state_dict:
            # Convert tensor format
            converted = _convert_tensor_pytorch_to_paddle(tensor, paddle_key)

            # Check shape compatibility
            expected_shape = list(paddle_state_dict[paddle_key].shape)
            actual_shape = list(converted.shape)

            if expected_shape == actual_shape:
                paddle_state_dict[paddle_key] = paddle.to_tensor(converted)
                loaded_keys.append(paddle_key)
            else:
                print(f"Shape mismatch for {paddle_key}: expected {expected_shape}, got {actual_shape}")
                missing_keys.append(paddle_key)
        else:
            unexpected_keys.append(diffusers_key)

    # Find missing keys
    for key in paddle_state_dict.keys():
        if key not in loaded_keys:
            missing_keys.append(key)

    # Load state dict
    paddle_model.set_state_dict(paddle_state_dict)

    print(f"Loaded {len(loaded_keys)} weights")
    print(f"Missing {len(missing_keys)} weights")
    if unexpected_keys:
        print(f"Unexpected {len(unexpected_keys)} weights (not loaded)")

    return loaded_keys, missing_keys


def load_vae_from_diffusers(
    vae_path: str,
    paddle_vae: paddle.nn.Layer,
) -> Tuple[List[str], List[str]]:
    """Load VAE weights from Diffusers format."""
    # Load weights
    weights = load_sharded_safetensors(vae_path)

    paddle_state_dict = paddle_vae.state_dict()
    loaded_keys = []
    missing_keys = list(paddle_state_dict.keys())

    # VAE has relatively simple key structure
    for diffusers_key, tensor in weights.items():
        paddle_key = _convert_key_pytorch_to_paddle(diffusers_key)

        if paddle_key in paddle_state_dict:
            converted = _convert_tensor_pytorch_to_paddle(tensor, paddle_key)

            expected_shape = list(paddle_state_dict[paddle_key].shape)
            actual_shape = list(converted.shape)

            if expected_shape == actual_shape:
                paddle_state_dict[paddle_key] = paddle.to_tensor(converted)
                loaded_keys.append(paddle_key)
                missing_keys.remove(paddle_key)

    paddle_vae.set_state_dict(paddle_state_dict)

    print(f"VAE: Loaded {len(loaded_keys)}, Missing {len(missing_keys)}")
    return loaded_keys, missing_keys


class Wan22ModelLoader:
    """
    Utility class to load Wan2.2 models from Diffusers checkpoints.

    Example:
        loader = Wan22ModelLoader("/path/to/Wan2.2-TI2V-5B-Diffusers")
        dit = loader.load_transformer()
        vae = loader.load_vae()
        scheduler = loader.load_scheduler()
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)

        # Verify path exists
        if not self.model_path.exists():
            raise ValueError(f"Model path does not exist: {model_path}")

        # Load model index
        index_path = self.model_path / "model_index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                self.model_index = json.load(f)
        else:
            self.model_index = {}

    def load_transformer_config(self) -> dict:
        """Load transformer configuration."""
        config_path = self.model_path / "transformer" / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return {}

    def load_vae_config(self) -> dict:
        """Load VAE configuration."""
        config_path = self.model_path / "vae" / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return {}

    def load_scheduler_config(self) -> dict:
        """Load scheduler configuration."""
        config_path = self.model_path / "scheduler" / "scheduler_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        return {}

    def create_transformer(self):
        """Create transformer model from config."""
        from .configuration import Wan22Config
        from .modeling_paddle import Wan22DiTModel

        diffusers_config = self.load_transformer_config()

        # Map Diffusers config to our config
        config = Wan22Config(
            dit_num_layers=diffusers_config.get("num_layers", 30),
            dit_num_attention_heads=diffusers_config.get("num_attention_heads", 24),
            dit_attention_head_dim=diffusers_config.get("attention_head_dim", 128),
            dit_in_channels=diffusers_config.get("in_channels", 48),
            dit_ff_inner_dim=diffusers_config.get("ffn_dim", 14336),
            dit_cross_attention_dim=diffusers_config.get("text_dim", 4096),
        )

        return Wan22DiTModel(config)

    def create_vae(self):
        """Create VAE model from config."""
        from .vae import Wan22VAEModel

        diffusers_config = self.load_vae_config()

        return Wan22VAEModel(
            in_channels=diffusers_config.get("in_channels", 3),
            latent_channels=diffusers_config.get("latent_channels", 48),
            base_channels=diffusers_config.get("base_channels", 160),
        )

    def create_scheduler(self):
        """Create scheduler from config."""
        from .scheduler import Wan22FlowMatchScheduler

        config = self.load_scheduler_config()

        return Wan22FlowMatchScheduler(
            num_train_timesteps=config.get("num_train_timesteps", 1000),
            shift=config.get("shift", 5.0),
        )


__all__ = [
    "load_safetensors",
    "load_sharded_safetensors",
    "convert_diffusers_transformer_to_paddle",
    "load_vae_from_diffusers",
    "Wan22ModelLoader",
]
