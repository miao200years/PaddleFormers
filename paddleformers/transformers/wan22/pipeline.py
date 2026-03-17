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
Wan2.2 Pipeline - PaddlePaddle Native Implementation

This module provides end-to-end pipelines for video generation:
- Text-to-Video (T2V)
- Image-to-Video (I2V)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import Tensor
from PIL import Image

from .configuration import Wan22Config
from .modeling_paddle import Wan22DiTModel
from .scheduler import Wan22FlowMatchScheduler, Wan22UniPCScheduler
from .vae import Wan22VAEModel


@dataclass
class Wan22PipelineOutput:
    """Output of Wan22 pipeline"""

    videos: Union[List[np.ndarray], np.ndarray]
    latents: Optional[Tensor] = None


# Size configurations for different aspect ratios
SIZE_CONFIGS = {
    "1280*720": (1280, 720),
    "720*1280": (720, 1280),
    "1104*832": (1104, 832),
    "832*1104": (832, 1104),
    "960*960": (960, 960),
    "832*480": (832, 480),
    "480*832": (480, 832),
    "624*624": (624, 624),
}


def _best_output_size(w: int, h: int, dw: int = 32, dh: int = 32, expected_area: int = 704 * 544) -> Tuple[int, int]:
    """
    Calculate optimal output dimensions maintaining aspect ratio.

    Args:
        w: Input width
        h: Input height
        dw: Width divisor (must be multiple of this)
        dh: Height divisor (must be multiple of this)
        expected_area: Target pixel area

    Returns:
        Tuple of (output_width, output_height)
    """
    aspect_ratio = w / h
    target_height = math.sqrt(expected_area / aspect_ratio)
    target_width = target_height * aspect_ratio

    # Option 1: width-first
    ow1 = int(target_width // dw) * dw
    oh1 = int(ow1 / aspect_ratio // dh) * dh

    # Option 2: height-first
    oh2 = int(target_height // dh) * dh
    ow2 = int(oh2 * aspect_ratio // dw) * dw

    # Choose option closer to original aspect ratio
    ar1 = ow1 / oh1 if oh1 > 0 else 0
    ar2 = ow2 / oh2 if oh2 > 0 else 0

    if abs(ar1 - aspect_ratio) < abs(ar2 - aspect_ratio):
        return (ow1, oh1)
    return (ow2, oh2)


class Wan22BasePipeline:
    """Base class for Wan22 pipelines"""

    def __init__(
        self,
        vae: Wan22VAEModel,
        transformer: Wan22DiTModel,
        scheduler: Union[Wan22FlowMatchScheduler, Wan22UniPCScheduler],
        text_encoder: Any = None,
        tokenizer: Any = None,
        config: Wan22Config = None,
    ):
        self.vae = vae
        self.transformer = transformer
        self.scheduler = scheduler
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.config = config or Wan22Config()

        # VAE scaling factor
        self.vae_scale_factor_spatial = 16
        self.vae_scale_factor_temporal = 4

    def progress_bar(self, iterable=None, total=None, desc=None):
        """Simple progress bar wrapper"""
        if iterable is not None:
            return iterable
        return range(total) if total else []

    @staticmethod
    def numpy_to_pil(images: np.ndarray) -> List[Image.Image]:
        """Convert numpy array to PIL Images"""
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype(np.uint8)
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    @staticmethod
    def pil_to_numpy(images: Union[Image.Image, List[Image.Image]]) -> np.ndarray:
        """Convert PIL Images to numpy array"""
        if not isinstance(images, list):
            images = [images]
        images = [np.array(img).astype(np.float32) / 255.0 for img in images]
        return np.stack(images, axis=0)

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
    ) -> Tensor:
        """
        Encode text prompt using T5 encoder.

        Note: This is a placeholder. In the full implementation,
        this would use UMT5 or similar text encoder.
        """
        # For now, return random embeddings as placeholder
        # TODO: Implement proper text encoding with T5
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # T5 output dimension
        embed_dim = self.config.dit_cross_attention_dim
        seq_len = self.config.text_len

        # Placeholder embeddings
        prompt_embeds = paddle.randn([batch_size, seq_len, embed_dim])

        if do_classifier_free_guidance:
            negative_embeds = paddle.randn([batch_size, seq_len, embed_dim])
            prompt_embeds = paddle.concat([negative_embeds, prompt_embeds], axis=0)

        return prompt_embeds

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: str,
        generator: Any = None,
        latents: Optional[Tensor] = None,
    ) -> Tensor:
        """Prepare initial latent noise"""
        shape = (
            batch_size,
            num_channels_latents,
            num_frames,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            latents = paddle.randn(shape, dtype=dtype)

        # Scale by scheduler init sigma
        latents = latents * self.scheduler.init_noise_sigma

        return latents

    def decode_latents(self, latents: Tensor) -> np.ndarray:
        """Decode latents to video frames"""
        video = self.vae.decode(latents).sample

        # Convert to numpy (B, C, T, H, W) -> (B, T, H, W, C)
        video = video.transpose([0, 2, 3, 4, 1])
        video = (video / 2 + 0.5).clip(0, 1)
        video = video.numpy()

        return video

    def check_inputs(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        num_frames: int,
    ):
        """Validate input parameters"""
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(f"Height and width must be divisible by 32, got {height}x{width}")

        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")


class Wan22TextToVideoPipeline(Wan22BasePipeline):
    """
    Text-to-Video generation pipeline for Wan2.2.

    This pipeline generates videos from text prompts using:
    1. T5 text encoder for prompt embedding
    2. DiT transformer for denoising
    3. VAE decoder for latent-to-video conversion
    """

    @paddle.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator: Any = None,
        latents: Optional[Tensor] = None,
        output_type: str = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> Union[Wan22PipelineOutput, Tuple]:
        """
        Generate video from text prompt.

        Args:
            prompt: Text prompt(s) for generation
            negative_prompt: Negative prompt(s) for classifier-free guidance
            height: Output video height (must be divisible by 32)
            width: Output video width (must be divisible by 32)
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            generator: Random generator for reproducibility
            latents: Pre-generated latents (optional)
            output_type: Output format ("np", "pil", "latent")
            return_dict: Whether to return as dataclass
            callback: Progress callback function
            callback_steps: Callback frequency

        Returns:
            Generated video(s)
        """
        # Validate inputs
        self.check_inputs(prompt, height, width, num_frames)

        # Batch handling
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # Classifier-free guidance
        do_classifier_free_guidance = guidance_scale > 1.0

        # Encode prompt
        prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt=negative_prompt or self.config.sample_neg_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Prepare latents
        num_channels_latents = self.config.dit_in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            num_frames,
            height,
            width,
            "float32",
            generator,
            latents,
        )

        # Denoising loop

        for i, t in enumerate(self.progress_bar(timesteps, desc="Sampling")):
            # Expand latents for classifier-free guidance
            latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            noise_pred = self.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=t.tile([latent_model_input.shape[0]]),
                return_dict=False,
            )

            if isinstance(noise_pred, dict):
                noise_pred = noise_pred["sample"]

            # Classifier-free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Callback
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # Decode latents
        if output_type == "latent":
            video = latents
        else:
            video = self.decode_latents(latents)

            if output_type == "pil":
                video = [self.numpy_to_pil(v) for v in video]

        if not return_dict:
            return (video,)

        return Wan22PipelineOutput(videos=video, latents=latents if output_type == "latent" else None)


class Wan22ImageToVideoPipeline(Wan22BasePipeline):
    """
    Image-to-Video generation pipeline for Wan2.2.

    This pipeline generates videos from an input image and text prompt.
    """

    def encode_image(self, image: Union[Image.Image, np.ndarray, Tensor]) -> Tensor:
        """Encode input image to latent space"""
        # Convert PIL to tensor
        if isinstance(image, Image.Image):
            image = np.array(image).astype(np.float32) / 255.0
            image = paddle.to_tensor(image)
        elif isinstance(image, np.ndarray):
            image = paddle.to_tensor(image.astype(np.float32))

        # Ensure correct shape (B, C, H, W)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[-1] == 3:  # HWC -> CHW
            image = image.transpose([0, 3, 1, 2])

        # Normalize to [-1, 1]
        image = 2 * image - 1

        # Add temporal dimension for VAE (B, C, 1, H, W)
        image = image.unsqueeze(2)

        # Encode
        latent_dist = self.vae.encode(image).latent_dist
        image_latents = latent_dist.sample() * self.vae.scaling_factor

        return image_latents

    @paddle.no_grad()
    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, Tensor],
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = None,
        width: int = None,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        generator: Any = None,
        latents: Optional[Tensor] = None,
        output_type: str = "np",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> Union[Wan22PipelineOutput, Tuple]:
        """
        Generate video from image and text prompt.

        Args:
            image: Input image
            prompt: Text prompt(s) for generation
            negative_prompt: Negative prompt(s)
            height: Output height (auto-calculated if None)
            width: Output width (auto-calculated if None)
            num_frames: Number of frames to generate
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            generator: Random generator
            latents: Pre-generated latents
            output_type: Output format
            return_dict: Whether to return as dataclass
            callback: Progress callback
            callback_steps: Callback frequency

        Returns:
            Generated video(s)
        """
        # Get image dimensions
        if isinstance(image, Image.Image):
            img_w, img_h = image.size
        elif isinstance(image, np.ndarray):
            img_h, img_w = image.shape[:2]
        else:
            img_h, img_w = image.shape[-2:]

        # Calculate output size based on image aspect ratio
        if height is None or width is None:
            width, height = _best_output_size(img_w, img_h)

        # Validate inputs
        self.check_inputs(prompt, height, width, num_frames)

        # Batch handling
        batch_size = 1 if isinstance(prompt, str) else len(prompt)

        # Classifier-free guidance
        do_classifier_free_guidance = guidance_scale > 1.0

        # Resize image to target size
        if isinstance(image, Image.Image):
            image = image.resize((width, height), Image.LANCZOS)

        # Encode image
        image_latents = self.encode_image(image)

        # Encode prompt (with image context prepended)
        prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt=negative_prompt or self.config.sample_neg_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # Prepare latents
        num_channels_latents = self.config.dit_in_channels
        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            num_frames,
            height,
            width,
            "float32",
            generator,
            latents,
        )

        # Set first frame to image latent
        latents[:, :, 0:1, :, :] = image_latents

        # Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps, desc="Sampling")):
            # Expand latents for classifier-free guidance
            latent_model_input = paddle.concat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Predict noise
            noise_pred = self.transformer(
                latent_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=t.tile([latent_model_input.shape[0]]),
                return_dict=False,
            )

            if isinstance(noise_pred, dict):
                noise_pred = noise_pred["sample"]

            # Classifier-free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Keep first frame fixed
            latents[:, :, 0:1, :, :] = image_latents

            # Callback
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)

        # Decode latents
        if output_type == "latent":
            video = latents
        else:
            video = self.decode_latents(latents)

            if output_type == "pil":
                video = [self.numpy_to_pil(v) for v in video]

        if not return_dict:
            return (video,)

        return Wan22PipelineOutput(videos=video, latents=latents if output_type == "latent" else None)


def save_video(
    frames: Union[np.ndarray, List[np.ndarray], List[Image.Image]],
    output_path: str,
    fps: int = 24,
):
    """
    Save video frames to file.

    Args:
        frames: Video frames as numpy array (T, H, W, C) or list
        output_path: Output file path
        fps: Frames per second
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("Please install imageio: pip install imageio[ffmpeg]")

    # Handle different input formats
    if isinstance(frames, list):
        if isinstance(frames[0], Image.Image):
            frames = [np.array(f) for f in frames]
        frames = np.stack(frames, axis=0)

    # Ensure uint8
    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)

    # Save
    imageio.mimwrite(output_path, frames, fps=fps)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    "Wan22PipelineOutput",
    "Wan22BasePipeline",
    "Wan22TextToVideoPipeline",
    "Wan22ImageToVideoPipeline",
    "save_video",
    "SIZE_CONFIGS",
]
