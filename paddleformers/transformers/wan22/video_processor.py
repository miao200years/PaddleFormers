# coding=utf-8
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Wan Team and The HuggingFace Inc. team. All rights reserved.
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
"""Video processor class for Wan2.2 - Training Support."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import paddle

    HAS_PADDLE = True
except ImportError:
    HAS_PADDLE = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False


@dataclass
class Wan22VideoMetadata:
    """Metadata for video processing"""

    total_num_frames: int
    fps: Optional[float] = None
    frames_indices: Optional[np.ndarray] = None
    height: Optional[int] = None
    width: Optional[int] = None


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 4,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 832 * 480 * 81,
) -> Tuple[int, int, int]:
    """
    Smart resize for Wan2.2 video dimensions.

    Args:
        num_frames: Number of video frames
        height: Input height
        width: Input width
        temporal_factor: Temporal patch size factor (Wan uses 4)
        factor: Spatial factor (patch_size * vae_stride = 2 * 16 = 32)
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels

    Returns:
        Tuple of (resized_height, resized_width, adjusted_frames)
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")

    # Adjust frames to be 4n+1 (Wan2.2 requirement)
    t_bar = ((num_frames - 1) // temporal_factor) * temporal_factor + 1
    t_bar = max(5, t_bar)  # Minimum 5 frames

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    total_pixels = t_bar * h_bar * w_bar

    if total_pixels > max_pixels:
        # Scale down
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif total_pixels < min_pixels:
        # Scale up
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar, t_bar


class Wan22VideoProcessor:
    """
    Video processor for Wan2.2 with training support.

    This processor handles:
    - Video frame sampling (fps-based or count-based)
    - Smart resize maintaining aspect ratio
    - Normalization and rescaling
    - Patch-based processing for transformer input
    - Grid THW (temporal, height, width) calculation
    - Training data preparation

    Based on Qwen3VL VideoProcessor design patterns.

    Example:
        ```python
        processor = Wan22VideoProcessor()

        # Process video for training
        outputs = processor(
            videos=video_frames,  # List of PIL Images or numpy arrays
            return_tensors="pd",
            return_metadata=True,
        )

        # Access processed data
        pixel_values = outputs["pixel_values_videos"]
        grid_thw = outputs["video_grid_thw"]
        ```
    """

    # Default parameters matching Wan2.2 architecture
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True

    # Wan2.2 specific parameters
    patch_size = 2  # VAE spatial patch
    temporal_patch_size = 4  # Temporal compression (4 frames -> 1 latent frame)
    vae_stride = 16  # VAE spatial stride
    merge_size = 2  # Spatial merge for attention

    fps = 24  # Default video fps
    min_frames = 5  # Minimum frames (4n+1 = 5)
    max_frames = 81  # Maximum frames (4n+1 = 81)
    do_sample_frames = True

    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(
        self,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_resize: bool = True,
        do_rescale: bool = True,
        do_normalize: bool = True,
        patch_size: int = 2,
        temporal_patch_size: int = 4,
        vae_stride: int = 16,
        merge_size: int = 2,
        fps: int = 24,
        min_frames: int = 5,
        max_frames: int = 81,
        **kwargs,
    ):
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.vae_stride = vae_stride
        self.merge_size = merge_size
        self.fps = fps
        self.min_frames = min_frames
        self.max_frames = max_frames

    def sample_frames(
        self,
        metadata: Wan22VideoMetadata,
        num_frames: Optional[int] = None,
        fps: Optional[Union[int, float]] = None,
    ) -> np.ndarray:
        """
        Sample frame indices from video.

        Args:
            metadata: Video metadata containing total frames and fps
            num_frames: Target number of frames (if None, calculated from fps)
            fps: Target fps for sampling (if None, uses default)

        Returns:
            Array of frame indices to sample
        """
        if fps is not None and num_frames is not None:
            raise ValueError("`num_frames` and `fps` are mutually exclusive")

        total_num_frames = metadata.total_num_frames
        video_fps = metadata.fps or 24

        if num_frames is None:
            if fps is not None:
                num_frames = int(total_num_frames / video_fps * fps)
            else:
                num_frames = int(total_num_frames / video_fps * self.fps)

        # Adjust to 4n+1
        num_frames = ((num_frames - 1) // 4) * 4 + 1
        num_frames = min(max(num_frames, self.min_frames), self.max_frames, total_num_frames)

        indices = np.linspace(0, total_num_frames - 1, num_frames).round().astype(int)
        return indices

    def resize(
        self,
        video: np.ndarray,
        size: Tuple[int, int],
        interpolation: str = "bilinear",
    ) -> np.ndarray:
        """
        Resize video frames.

        Args:
            video: Video array (T, H, W, C) or (T, C, H, W)
            size: Target (height, width)
            interpolation: Interpolation method

        Returns:
            Resized video array
        """
        if HAS_PIL:
            # Use PIL for resizing
            is_chw = video.shape[-1] != 3 and video.shape[1] == 3
            if is_chw:
                video = video.transpose(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)

            resized_frames = []
            for frame in video:
                if frame.dtype != np.uint8:
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                img = Image.fromarray(frame)
                img = img.resize((size[1], size[0]), Image.BILINEAR)  # PIL uses (W, H)
                resized_frames.append(np.array(img))

            resized = np.stack(resized_frames, axis=0)
            if is_chw:
                resized = resized.transpose(0, 3, 1, 2)  # (T, H, W, C) -> (T, C, H, W)
            return resized.astype(np.float32) / 255.0
        else:
            # Simple resize using numpy (nearest neighbor)
            T, H, W, C = video.shape
            new_H, new_W = size
            indices_h = (np.arange(new_H) * H / new_H).astype(int)
            indices_w = (np.arange(new_W) * W / new_W).astype(int)
            return video[:, indices_h][:, :, indices_w]

    def rescale(self, video: np.ndarray, scale: float = 1.0 / 255.0) -> np.ndarray:
        """Rescale video values"""
        return video.astype(np.float32) * scale

    def normalize(
        self,
        video: np.ndarray,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Normalize video to [-1, 1] range.

        Args:
            video: Video array (T, C, H, W) or (T, H, W, C)
            mean: Channel means
            std: Channel stds

        Returns:
            Normalized video
        """
        mean = mean or self.image_mean
        std = std or self.image_std

        # Determine channel dimension
        if video.shape[-1] == 3:  # (T, H, W, C)
            mean = np.array(mean).reshape(1, 1, 1, 3)
            std = np.array(std).reshape(1, 1, 1, 3)
        else:  # (T, C, H, W)
            mean = np.array(mean).reshape(1, 3, 1, 1)
            std = np.array(std).reshape(1, 3, 1, 1)

        return (video - mean) / std

    def compute_grid_thw(
        self,
        num_frames: int,
        height: int,
        width: int,
    ) -> Tuple[int, int, int]:
        """
        Compute grid dimensions after VAE encoding.

        For Wan2.2:
        - Temporal: frames compressed by temporal_patch_size (4)
        - Spatial: height/width compressed by vae_stride (16)

        Args:
            num_frames: Number of frames
            height: Frame height
            width: Frame width

        Returns:
            Tuple of (grid_t, grid_h, grid_w)
        """
        # Adjust frames to 4n+1
        adjusted_frames = ((num_frames - 1) // self.temporal_patch_size) * self.temporal_patch_size + 1

        # After VAE encoding
        grid_t = (adjusted_frames - 1) // self.temporal_patch_size + 1  # Temporal latent frames
        grid_h = height // self.vae_stride
        grid_w = width // self.vae_stride

        return (grid_t, grid_h, grid_w)

    def _preprocess_single_video(
        self,
        video: Union[List, np.ndarray],
        target_size: Optional[Tuple[int, int]] = None,
        target_frames: Optional[int] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int, int], Wan22VideoMetadata]:
        """
        Preprocess a single video.

        Args:
            video: List of PIL Images or numpy array (T, H, W, C)
            target_size: Target (height, width)
            target_frames: Target number of frames

        Returns:
            Tuple of (processed_video, grid_thw, metadata)
        """
        # Convert to numpy if needed
        if isinstance(video, list):
            if HAS_PIL and isinstance(video[0], Image.Image):
                frames = [np.array(img.convert("RGB")) for img in video]
                video = np.stack(frames, axis=0)
            else:
                video = np.stack(video, axis=0)

        T, H, W, C = video.shape
        metadata = Wan22VideoMetadata(
            total_num_frames=T,
            height=H,
            width=W,
            fps=self.fps,
        )

        # Sample frames if needed
        if target_frames is not None and target_frames != T:
            indices = self.sample_frames(metadata, num_frames=target_frames)
            video = video[indices]
            T = len(indices)
            metadata.frames_indices = indices
        else:
            metadata.frames_indices = np.arange(T)

        # Calculate target size
        if target_size is None:
            factor = self.patch_size * self.vae_stride  # 32
            h_bar, w_bar, _ = smart_resize(T, H, W, self.temporal_patch_size, factor)
            target_size = (h_bar, w_bar)

        # Resize
        if self.do_resize and (H != target_size[0] or W != target_size[1]):
            video = self.resize(video, target_size)

        # Rescale to [0, 1]
        if self.do_rescale:
            if video.max() > 1.0:
                video = self.rescale(video)

        # Normalize to [-1, 1]
        if self.do_normalize:
            video = self.normalize(video)

        # Compute grid
        grid_thw = self.compute_grid_thw(T, target_size[0], target_size[1])

        # Convert to (T, C, H, W) format for model
        if video.shape[-1] == 3:
            video = video.transpose(0, 3, 1, 2)

        return video, grid_thw, metadata

    def __call__(
        self,
        videos: Union[List, np.ndarray, "paddle.Tensor"],
        target_size: Optional[Tuple[int, int]] = None,
        target_frames: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_metadata: bool = False,
        **kwargs,
    ) -> Dict:
        """
        Process videos for Wan2.2 model.

        Args:
            videos: Single video or list of videos
                - List of PIL Images
                - numpy array (T, H, W, C) or (B, T, H, W, C)
                - paddle.Tensor
            target_size: Target (height, width)
            target_frames: Target number of frames (adjusted to 4n+1)
            return_tensors: Output format ("pd" for paddle, "np" for numpy)
            return_metadata: Whether to return video metadata

        Returns:
            Dictionary with:
                - pixel_values_videos: Processed video tensor
                - video_grid_thw: Grid dimensions for each video
                - video_metadata: (optional) Video metadata list
        """
        # Handle single video vs batch
        if isinstance(videos, np.ndarray):
            if videos.ndim == 4:  # Single video (T, H, W, C)
                videos = [videos]
            elif videos.ndim == 5:  # Batch (B, T, H, W, C)
                videos = [videos[i] for i in range(videos.shape[0])]
        elif not isinstance(videos, list):
            videos = [videos]

        # Check if videos is a list of frames (single video)
        if len(videos) > 0 and HAS_PIL and isinstance(videos[0], Image.Image):
            videos = [videos]  # Wrap single video

        processed_videos = []
        grid_thws = []
        metadatas = []

        for video in videos:
            processed, grid_thw, metadata = self._preprocess_single_video(video, target_size, target_frames)
            processed_videos.append(processed)
            grid_thws.append(grid_thw)
            metadatas.append(metadata)

        # Stack videos
        pixel_values = np.stack(processed_videos, axis=0)  # (B, T, C, H, W)
        grid_thw_array = np.array(grid_thws)  # (B, 3)

        # Convert to tensors if requested
        if return_tensors == "pd" and HAS_PADDLE:
            pixel_values = paddle.to_tensor(pixel_values)
            grid_thw_array = paddle.to_tensor(grid_thw_array)

        result = {
            "pixel_values_videos": pixel_values,
            "video_grid_thw": grid_thw_array,
        }

        if return_metadata:
            result["video_metadata"] = metadatas

        return result

    def get_number_of_video_patches(
        self,
        num_frames: int,
        height: int,
        width: int,
    ) -> int:
        """
        Calculate number of video patches after processing.

        Args:
            num_frames: Number of frames
            height: Frame height
            width: Frame width

        Returns:
            Total number of patches
        """
        grid_t, grid_h, grid_w = self.compute_grid_thw(num_frames, height, width)
        return grid_t * grid_h * grid_w


# Training-specific utilities
class Wan22TrainingProcessor:
    """
    Training-specific processor for Wan2.2.

    Combines video processing with:
    - Noise scheduling preparation
    - Latent space processing
    - Loss computation helpers
    """

    def __init__(
        self,
        video_processor: Optional[Wan22VideoProcessor] = None,
        noise_scheduler_config: Optional[Dict] = None,
    ):
        self.video_processor = video_processor or Wan22VideoProcessor()
        self.noise_scheduler_config = noise_scheduler_config or {
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "flow_shift": 5.0,  # Wan2.2 specific
        }

    def prepare_training_batch(
        self,
        videos: List,
        prompts: List[str],
        tokenizer=None,
        max_text_length: int = 512,
    ) -> Dict:
        """
        Prepare a complete training batch.

        Args:
            videos: List of videos (each is list of frames or numpy array)
            prompts: List of text prompts
            tokenizer: Text tokenizer (T5)
            max_text_length: Maximum text sequence length

        Returns:
            Dictionary with all training inputs
        """
        # Process videos
        video_outputs = self.video_processor(
            videos,
            return_tensors="pd" if HAS_PADDLE else "np",
            return_metadata=True,
        )

        # Process text
        text_outputs = {}
        if tokenizer is not None:
            encoded = tokenizer(
                prompts,
                padding="max_length",
                max_length=max_text_length,
                truncation=True,
                return_tensors="pd" if HAS_PADDLE else "np",
            )
            text_outputs = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            }

        return {
            **video_outputs,
            **text_outputs,
            "prompts": prompts,
        }

    def compute_snr_weights(
        self,
        timesteps: np.ndarray,
        snr_gamma: float = 5.0,
    ) -> np.ndarray:
        """
        Compute SNR-based loss weights for training.

        Args:
            timesteps: Sampled timesteps
            snr_gamma: SNR gamma value

        Returns:
            Loss weights array
        """
        # Simple SNR weight computation
        # For more advanced, use actual noise scheduler
        t_normalized = timesteps / self.noise_scheduler_config["num_train_timesteps"]
        snr = (1 - t_normalized) / t_normalized.clip(min=1e-8)
        weights = snr / (snr + snr_gamma)
        return weights


__all__ = [
    "Wan22VideoProcessor",
    "Wan22VideoMetadata",
    "Wan22TrainingProcessor",
    "smart_resize",
]
