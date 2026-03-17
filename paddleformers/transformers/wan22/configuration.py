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
"""Wan2.2 model configuration"""

from ..configuration_utils import PretrainedConfig


class Wan22VAEConfig(PretrainedConfig):
    """Configuration for Wan2.2 VAE"""

    model_type = "wan22_vae"
    base_config_key = "vae_config"

    def __init__(
        self,
        z_dim: int = 48,
        c_dim: int = 160,
        dim_mult: list = None,
        temporal_downsample: list = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim_mult = dim_mult or [1, 2, 4, 4]
        self.temporal_downsample = temporal_downsample or [False, True, True]


class Wan22DiTConfig(PretrainedConfig):
    """Configuration for Wan2.2 DiT (Diffusion Transformer)"""

    model_type = "wan22_dit"
    base_config_key = "dit_config"

    def __init__(
        self,
        dim: int = 3072,
        ffn_dim: int = 14336,
        freq_dim: int = 256,
        num_heads: int = 24,
        num_layers: int = 30,
        window_size: tuple = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        patch_size: tuple = (1, 2, 2),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.patch_size = patch_size


class Wan22Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a Wan2.2 model.
    It is used to instantiate a Wan2.2 model for video generation according to the
    specified arguments.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs.

    Args:
        task (`str`, *optional*, defaults to `"ti2v-5B"`):
            The task type. Options: "t2v-A14B", "i2v-A14B", "ti2v-5B".
        text_len (`int`, *optional*, defaults to 512):
            Maximum text sequence length.
        vae_stride (`tuple`, *optional*, defaults to (4, 16, 16)):
            VAE temporal and spatial stride.
        patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
            Patch size for DiT.
        frame_num (`int`, *optional*, defaults to 121):
            Number of frames to generate.
        sample_fps (`int`, *optional*, defaults to 24):
            Output video FPS.
        sample_steps (`int`, *optional*, defaults to 50):
            Number of diffusion sampling steps.
        sample_shift (`float`, *optional*, defaults to 5.0):
            Noise schedule shift parameter.
        sample_guide_scale (`float`, *optional*, defaults to 5.0):
            Classifier-free guidance scale.
        num_train_timesteps (`int`, *optional*, defaults to 1000):
            Number of training timesteps for diffusion.
        t5_checkpoint (`str`, *optional*):
            Path to T5 encoder checkpoint.
        t5_tokenizer (`str`, *optional*):
            Path to T5 tokenizer.
        vae_checkpoint (`str`, *optional*):
            Path to VAE checkpoint.
        sample_neg_prompt (`str`, *optional*):
            Default negative prompt.

    Example:
    ```python
    >>> from paddleformers.transformers import Wan22Config, Wan22ForTextToVideo

    >>> # Initializing a Wan2.2 configuration
    >>> configuration = Wan22Config()

    >>> # Initializing a model from the configuration
    >>> model = Wan22ForTextToVideo(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "wan22"
    sub_configs = {
        "vae_config": Wan22VAEConfig,
        "dit_config": Wan22DiTConfig,
    }

    def __init__(
        self,
        task: str = "ti2v-5B",
        text_len: int = 512,
        vae_stride: tuple = (4, 16, 16),
        patch_size: tuple = (1, 2, 2),
        frame_num: int = 121,
        sample_fps: int = 24,
        sample_steps: int = 50,
        sample_shift: float = 5.0,
        sample_guide_scale: float = 5.0,
        num_train_timesteps: int = 1000,
        t5_checkpoint: str = "models_t5_umt5-xxl-enc-bf16.pth",
        t5_tokenizer: str = "google/umt5-xxl",
        vae_checkpoint: str = "Wan2.2_VAE.pth",
        sample_neg_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        vae_config: dict = None,
        dit_config: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.task = task
        self.text_len = text_len
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.frame_num = frame_num
        self.sample_fps = sample_fps
        self.sample_steps = sample_steps
        self.sample_shift = sample_shift
        self.sample_guide_scale = sample_guide_scale
        self.num_train_timesteps = num_train_timesteps
        self.t5_checkpoint = t5_checkpoint
        self.t5_tokenizer = t5_tokenizer
        self.vae_checkpoint = vae_checkpoint
        self.sample_neg_prompt = sample_neg_prompt

        # DiT model parameters (for native PaddlePaddle implementation)
        # These are derived from the 5B model configuration
        self.dit_in_channels = 48  # Same as VAE z_dim
        self.dit_patch_size = 2
        self.dit_num_attention_heads = 24
        self.dit_attention_head_dim = 128  # dim / num_heads = 3072 / 24
        self.dit_cross_attention_dim = 4096  # T5 hidden size
        self.dit_norm_eps = 1e-6
        self.dit_ff_inner_dim = 14336
        self.dit_num_layers = 30
        self.dit_added_kv_proj_dim = 4096  # For I2V image conditioning
        self.dit_cross_attention_dim_head = 64

        # Sub-configs
        if isinstance(vae_config, dict):
            self.vae_config = self.sub_configs["vae_config"](**vae_config)
        elif vae_config is None:
            self.vae_config = self.sub_configs["vae_config"]()
        else:
            self.vae_config = vae_config

        if isinstance(dit_config, dict):
            self.dit_config = self.sub_configs["dit_config"](**dit_config)
        elif dit_config is None:
            self.dit_config = self.sub_configs["dit_config"]()
        else:
            self.dit_config = dit_config


__all__ = ["Wan22Config", "Wan22VAEConfig", "Wan22DiTConfig"]
