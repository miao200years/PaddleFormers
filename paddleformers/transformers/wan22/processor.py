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
"""Processor class for Wan2.2 Video Generation.

This module provides ProcessorMixin-compatible processors for Wan2.2 models.
The Wan22Processor class implements the PaddleFormers ProcessorMixin interface
while using diffusers as the underlying backend.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .configuration import Wan22Config

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


@dataclass
class Wan22ProcessorOutput:
    """Output of Wan22Processor"""

    # Text inputs
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    prompt_tokens: Optional[np.ndarray] = None
    negative_prompt_tokens: Optional[np.ndarray] = None

    # Image inputs (for I2V)
    pixel_values: Optional[np.ndarray] = None  # (C, H, W) or (B, C, H, W)
    image_size: Optional[Tuple[int, int]] = None  # (width, height)

    # Generation parameters
    height: Optional[int] = None
    width: Optional[int] = None
    num_frames: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class Wan22ImageProcessor:
    """
    Image processor for Wan2.2 I2V (Image-to-Video) tasks.

    Handles:
    - Image resizing while maintaining aspect ratio
    - Normalization to [-1, 1]
    - Converting to tensor format
    """

    def __init__(
        self,
        do_resize: bool = True,
        do_normalize: bool = True,
        do_convert_rgb: bool = True,
        image_mean: List[float] = None,
        image_std: List[float] = None,
        resample: str = "lanczos",
    ):
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb
        self.image_mean = image_mean or [0.5, 0.5, 0.5]
        self.image_std = image_std or [0.5, 0.5, 0.5]
        self.resample = resample

        # PIL resampling method
        self._resample_map = {
            "nearest": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }

    @staticmethod
    def best_output_size(
        w: int,
        h: int,
        dw: int = 32,
        dh: int = 32,
        expected_area: int = 704 * 544,
    ) -> Tuple[int, int]:
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

    def resize(
        self,
        image: Image.Image,
        size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Resize image to target size"""
        if size is None:
            # Auto-calculate based on aspect ratio
            w, h = image.size
            size = self.best_output_size(w, h)

        resample = self._resample_map.get(self.resample, Image.LANCZOS)
        return image.resize(size, resample)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [-1, 1]"""
        # Assume image is in [0, 1] range
        mean = np.array(self.image_mean).reshape(3, 1, 1)
        std = np.array(self.image_std).reshape(3, 1, 1)
        return (image - mean) / std

    def __call__(
        self,
        images: Union[Image.Image, List[Image.Image], np.ndarray],
        size: Optional[Tuple[int, int]] = None,
        return_tensors: str = "np",
    ) -> Dict[str, np.ndarray]:
        """
        Process images for Wan2.2.

        Args:
            images: Input image(s)
            size: Target size (width, height). Auto-calculated if None.
            return_tensors: Output format ("np" for numpy)

        Returns:
            Dictionary with processed pixel values
        """
        # Handle single image
        if isinstance(images, Image.Image):
            images = [images]
        elif isinstance(images, np.ndarray) and images.ndim == 3:
            images = [Image.fromarray((images * 255).astype(np.uint8) if images.max() <= 1 else images)]

        processed = []
        image_sizes = []

        for img in images:
            # Convert to RGB
            if self.do_convert_rgb and img.mode != "RGB":
                img = img.convert("RGB")

            # Resize
            if self.do_resize:
                img = self.resize(img, size)

            image_sizes.append(img.size)

            # Convert to numpy (H, W, C) -> (C, H, W)
            img_array = np.array(img).astype(np.float32) / 255.0
            img_array = img_array.transpose(2, 0, 1)

            # Normalize
            if self.do_normalize:
                img_array = self.normalize(img_array)

            processed.append(img_array)

        # Stack into batch
        pixel_values = np.stack(processed, axis=0)

        return {
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        }


class Wan22Processor:
    """
    Processor for Wan2.2 video generation models.

    This processor implements the PaddleFormers ProcessorMixin interface
    for compatibility with the unified processor API.

    It handles:
    - Text prompt processing (tokenization)
    - Image preprocessing for I2V tasks
    - Generation parameter validation

    Example:
        ```python
        from paddleformers.transformers.wan22 import Wan22Processor

        processor = Wan22Processor()

        # T2V: Text-to-Video
        inputs = processor(
            prompt="A cat walking on grass",
            size="832*480",
            num_frames=25,
        )

        # I2V: Image-to-Video
        from PIL import Image
        image = Image.open("input.jpg")
        inputs = processor(
            prompt="The scene comes alive",
            images=image,
            num_frames=25,
        )
        ```

    Attributes:
        attributes: List of sub-processor attribute names (ProcessorMixin interface)
        image_processor_class: Class name of the image processor
        tokenizer_class: Class name of the tokenizer
        valid_processor_kwargs: Valid keyword arguments for processing
    """

    # ProcessorMixin interface attributes
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Wan22ImageProcessor"
    tokenizer_class = "T5Tokenizer"
    _auto_class = "AutoProcessor"

    # Valid kwargs for processing
    valid_processor_kwargs = [
        "prompt",
        "negative_prompt",
        "images",
        "size",
        "num_frames",
        "return_tensors",
    ]

    def __init__(
        self,
        image_processor: Wan22ImageProcessor = None,
        tokenizer=None,
        config: Wan22Config = None,
        **kwargs,
    ):
        """
        Initialize Wan22Processor.

        Args:
            image_processor: Image processor instance
            tokenizer: Tokenizer instance (optional, for text encoding)
            config: Wan22Config instance
            **kwargs: Additional arguments
        """
        self.image_processor = image_processor or Wan22ImageProcessor()
        self.tokenizer = tokenizer
        self.config = config or Wan22Config()

        # Default generation parameters
        self.default_size = "832*480"
        self.default_num_frames = 81
        self.default_fps = 24

        # Store any extra kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _parse_size(self, size: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
        """Parse size string or tuple to (width, height)"""
        if isinstance(size, str):
            if size in SIZE_CONFIGS:
                return SIZE_CONFIGS[size]
            elif "*" in size:
                w, h = size.split("*")
                return (int(w), int(h))
            else:
                raise ValueError(f"Invalid size format: {size}. Expected 'WxH' or key in SIZE_CONFIGS")
        return size

    def _validate_num_frames(self, num_frames: int) -> int:
        """Validate and adjust num_frames to be 4n+1"""
        # Wan2.2 requires num_frames to be 4n+1
        remainder = (num_frames - 1) % 4
        if remainder != 0:
            num_frames = ((num_frames - 1) // 4) * 4 + 1
        return max(5, num_frames)  # Minimum 5 frames

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        images: Union[Image.Image, List[Image.Image], np.ndarray] = None,
        size: Union[str, Tuple[int, int]] = None,
        num_frames: int = None,
        **kwargs,
    ) -> Wan22ProcessorOutput:
        """
        Process inputs for Wan2.2 video generation.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt for CFG
            images: Input image(s) for I2V task
            size: Output size as "WxH" string or (W, H) tuple
            num_frames: Number of frames to generate

        Returns:
            Wan22ProcessorOutput with processed inputs
        """
        output = Wan22ProcessorOutput()

        # Process prompt
        if prompt is not None:
            output.prompt = prompt if isinstance(prompt, str) else prompt[0]

            # Tokenize if tokenizer available
            if self.tokenizer is not None:
                tokens = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.config.text_len,
                    truncation=True,
                    return_tensors="np",
                )
                output.prompt_tokens = tokens["input_ids"]

        # Process negative prompt
        if negative_prompt is None:
            negative_prompt = self.config.sample_neg_prompt
        output.negative_prompt = negative_prompt

        if self.tokenizer is not None and negative_prompt:
            neg_tokens = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.config.text_len,
                truncation=True,
                return_tensors="np",
            )
            output.negative_prompt_tokens = neg_tokens["input_ids"]

        # Determine size
        if size is None:
            if images is not None:
                # Auto-calculate from image aspect ratio
                if isinstance(images, Image.Image):
                    img_w, img_h = images.size
                elif isinstance(images, list):
                    img_w, img_h = images[0].size
                else:
                    img_h, img_w = images.shape[:2]
                width, height = self.image_processor.best_output_size(img_w, img_h)
            else:
                width, height = self._parse_size(self.default_size)
        else:
            width, height = self._parse_size(size)

        output.width = width
        output.height = height

        # Process num_frames
        if num_frames is None:
            num_frames = self.default_num_frames
        output.num_frames = self._validate_num_frames(num_frames)

        # Process images (I2V)
        if images is not None:
            image_outputs = self.image_processor(
                images,
                size=(width, height),
            )
            output.pixel_values = image_outputs["pixel_values"]
            output.image_size = image_outputs["image_sizes"][0]

        return output

    def decode_video(
        self,
        video: np.ndarray,
        output_type: str = "pil",
    ) -> Union[List[Image.Image], np.ndarray]:
        """
        Post-process generated video.

        Args:
            video: Video tensor (T, H, W, C) or (B, T, H, W, C)
            output_type: "pil" for PIL Images, "np" for numpy

        Returns:
            Processed video frames
        """
        # Handle batch dimension
        if video.ndim == 5:
            video = video[0]  # Take first batch

        # Denormalize if needed (from [-1, 1] to [0, 1])
        if video.min() < 0:
            video = (video + 1) / 2

        # Clip to valid range
        video = np.clip(video, 0, 1)

        if output_type == "pil":
            frames = []
            for frame in video:
                frame_uint8 = (frame * 255).astype(np.uint8)
                frames.append(Image.fromarray(frame_uint8))
            return frames
        else:
            return video

    # ==================== ProcessorMixin Interface Methods ====================

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs,
    ) -> "Wan22Processor":
        """
        Load processor from pretrained model path.

        This method implements the ProcessorMixin interface for loading
        processors from local directories or model hubs.

        Args:
            pretrained_model_name_or_path: Path to model directory or model ID
            **kwargs: Additional arguments

        Returns:
            Wan22Processor instance
        """
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)

        # Try to load processor config
        processor_config = {}
        config_file = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        processor_file = os.path.join(pretrained_model_name_or_path, "processor_config.json")

        for cfg_path in [processor_file, config_file]:
            if os.path.isfile(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    processor_config = json.load(f)
                break

        # Extract image processor config
        image_processor_config = processor_config.get("image_processor", {})
        image_processor = Wan22ImageProcessor(**image_processor_config)

        # Try to load Wan22Config
        config = None
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.isfile(config_path):
            config = Wan22Config.from_pretrained(pretrained_model_name_or_path)

        # Create processor
        processor = cls(
            image_processor=image_processor,
            tokenizer=kwargs.pop("tokenizer", None),
            config=config,
            **kwargs,
        )

        # Apply config values
        processor.default_size = processor_config.get("default_size", "832*480")
        processor.default_num_frames = processor_config.get("default_num_frames", 81)

        return processor

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ) -> List[str]:
        """
        Save processor to directory.

        This method implements the ProcessorMixin interface for saving
        processors to local directories.

        Args:
            save_directory: Directory to save processor files
            push_to_hub: Whether to push to hub (not implemented for Wan22)
            **kwargs: Additional arguments

        Returns:
            List of saved file paths
        """
        os.makedirs(save_directory, exist_ok=True)
        saved_files = []

        # Save processor config
        processor_config = self.to_dict()
        processor_file = os.path.join(save_directory, "processor_config.json")
        with open(processor_file, "w", encoding="utf-8") as f:
            json.dump(processor_config, f, indent=2)
        saved_files.append(processor_file)

        # Also save as preprocessor_config.json for compatibility
        preprocessor_file = os.path.join(save_directory, "preprocessor_config.json")
        with open(preprocessor_file, "w", encoding="utf-8") as f:
            json.dump(processor_config, f, indent=2)
        saved_files.append(preprocessor_file)

        return saved_files

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert processor to dictionary.

        Returns:
            Dictionary representation of processor configuration
        """
        output = {
            "processor_class": self.__class__.__name__,
            "image_processor": {
                "do_resize": self.image_processor.do_resize,
                "do_normalize": self.image_processor.do_normalize,
                "do_convert_rgb": self.image_processor.do_convert_rgb,
                "image_mean": self.image_processor.image_mean,
                "image_std": self.image_processor.image_std,
                "resample": self.image_processor.resample,
            },
            "default_size": self.default_size,
            "default_num_frames": self.default_num_frames,
            "default_fps": self.default_fps,
        }

        # Add config info if available
        if self.config is not None:
            output["model_type"] = getattr(self.config, "model_type", "wan22")

        return output

    def to_json_string(self) -> str:
        """
        Convert processor to JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save processor to JSON file.

        Args:
            json_file_path: Path to JSON file
        """
        with open(json_file_path, "w", encoding="utf-8") as f:
            f.write(self.to_json_string())

    def __repr__(self) -> str:
        """String representation of processor"""
        return (
            f"{self.__class__.__name__}(\n"
            f"  image_processor={self.image_processor.__class__.__name__},\n"
            f"  tokenizer={self.tokenizer.__class__.__name__ if self.tokenizer else None},\n"
            f"  default_size={self.default_size},\n"
            f"  default_num_frames={self.default_num_frames}\n"
            f")"
        )

    # ==================== Training Support Methods ====================

    def prepare_for_training(
        self,
        videos: Union[List, np.ndarray] = None,
        prompts: Union[str, List[str]] = None,
        negative_prompts: Union[str, List[str]] = None,
        images: Union[Image.Image, List[Image.Image]] = None,
        target_size: Tuple[int, int] = None,
        target_frames: int = None,
        return_tensors: str = "np",
        **kwargs,
    ) -> Dict:
        """
        Prepare inputs for training Wan2.2 model.

        This method combines video/image processing with text tokenization
        to create complete training batches.

        Args:
            videos: Video inputs (list of frame lists or numpy arrays)
            prompts: Text prompts for training
            negative_prompts: Negative prompts (optional)
            images: Image inputs for I2V training
            target_size: Target video size (height, width)
            target_frames: Target number of frames (adjusted to 4n+1)
            return_tensors: Output format ("np", "pd" for paddle)

        Returns:
            Dictionary with training-ready inputs:
                - pixel_values: Processed video/image pixels
                - input_ids: Tokenized prompt IDs
                - attention_mask: Token attention masks
                - video_grid_thw: Grid dimensions
                - negative_input_ids: (optional) Negative prompt IDs
        """
        from .video_processor import Wan22VideoProcessor

        result = {}

        # Process videos if provided
        if videos is not None:
            video_processor = Wan22VideoProcessor()
            video_outputs = video_processor(
                videos,
                target_size=target_size,
                target_frames=target_frames,
                return_tensors=return_tensors,
                return_metadata=True,
            )
            result.update(video_outputs)

        # Process images for I2V
        elif images is not None:
            image_outputs = self.image_processor(images, size=target_size)
            result["pixel_values"] = image_outputs["pixel_values"]
            result["image_sizes"] = image_outputs["image_sizes"]

        # Process text prompts
        if prompts is not None:
            if isinstance(prompts, str):
                prompts = [prompts]

            if self.tokenizer is not None:
                text_outputs = self.tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=self.config.text_len,
                    truncation=True,
                    return_tensors=return_tensors,
                )
                result["input_ids"] = text_outputs["input_ids"]
                result["attention_mask"] = text_outputs.get("attention_mask")

            result["prompts"] = prompts

        # Process negative prompts
        if negative_prompts is not None:
            if isinstance(negative_prompts, str):
                negative_prompts = [negative_prompts]

            if self.tokenizer is not None:
                neg_outputs = self.tokenizer(
                    negative_prompts,
                    padding="max_length",
                    max_length=self.config.text_len,
                    truncation=True,
                    return_tensors=return_tensors,
                )
                result["negative_input_ids"] = neg_outputs["input_ids"]
                result["negative_attention_mask"] = neg_outputs.get("attention_mask")

        return result

    def compute_loss_weights(
        self,
        timesteps: np.ndarray,
        snr_gamma: float = 5.0,
        min_snr_gamma: bool = True,
    ) -> np.ndarray:
        """
        Compute loss weights based on SNR (Signal-to-Noise Ratio).

        Used for training with min-SNR weighting strategy.

        Args:
            timesteps: Sampled timesteps array
            snr_gamma: SNR gamma value for weighting
            min_snr_gamma: Whether to use min-SNR strategy

        Returns:
            Loss weights array
        """
        num_timesteps = 1000  # Default Wan2.2 timesteps

        # Compute SNR at each timestep
        t_normalized = timesteps.astype(np.float32) / num_timesteps
        # Simple linear beta schedule approximation
        snr = (1 - t_normalized) / np.clip(t_normalized, 1e-8, 1.0)

        if min_snr_gamma:
            weights = np.minimum(snr, snr_gamma) / snr_gamma
        else:
            weights = snr / (snr + snr_gamma)

        return weights

    def get_training_collate_fn(self):
        """
        Get collate function for DataLoader.

        Returns:
            Collate function that batches training samples
        """

        def collate_fn(batch):
            """Collate batch of training samples"""
            # Assuming batch is list of dicts
            result = {}

            # Get keys from first sample
            keys = batch[0].keys()

            for key in keys:
                values = [sample[key] for sample in batch]

                if isinstance(values[0], np.ndarray):
                    result[key] = np.stack(values, axis=0)
                elif isinstance(values[0], list):
                    result[key] = values  # Keep as list
                else:
                    result[key] = values

            return result

        return collate_fn

    @property
    def model_input_names(self) -> List[str]:
        """Return list of model input names for training"""
        return [
            "pixel_values",
            "pixel_values_videos",
            "input_ids",
            "attention_mask",
            "video_grid_thw",
            "timesteps",
            "noise",
        ]


__all__ = [
    "Wan22Processor",
    "Wan22ImageProcessor",
    "Wan22ProcessorOutput",
    "SIZE_CONFIGS",
]
