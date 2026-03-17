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
"""Tests for Wan2.2 Video Generation Models

This module tests the Wan2.2 model components as integrated into PaddleFormers.
Tests use diffusers as the backend - no external wan module paths required.
"""
from __future__ import annotations

import os
import tempfile
import unittest
import warnings

import numpy as np

warnings.filterwarnings("ignore")


class Wan22ConfigTest(unittest.TestCase):
    """Tests for Wan22Config"""

    def test_config_initialization(self):
        """Test basic config initialization"""
        from paddleformers.transformers.wan22 import Wan22Config

        config = Wan22Config()

        self.assertEqual(config.task, "ti2v-5B")
        self.assertEqual(config.text_len, 512)
        self.assertEqual(config.frame_num, 121)
        self.assertEqual(config.sample_fps, 24)
        self.assertEqual(config.sample_steps, 50)
        self.assertEqual(config.sample_shift, 5.0)
        self.assertEqual(config.sample_guide_scale, 5.0)

    def test_config_custom_values(self):
        """Test config with custom values"""
        from paddleformers.transformers.wan22 import Wan22Config

        config = Wan22Config(
            task="t2v-A14B",
            frame_num=81,
            sample_steps=30,
        )

        self.assertEqual(config.task, "t2v-A14B")
        self.assertEqual(config.frame_num, 81)
        self.assertEqual(config.sample_steps, 30)

    def test_config_serialization(self):
        """Test config save and load"""
        from paddleformers.transformers.wan22 import Wan22Config

        config = Wan22Config(frame_num=49)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            loaded_config = Wan22Config.from_pretrained(tmpdir)

            self.assertEqual(loaded_config.frame_num, 49)
            self.assertEqual(loaded_config.model_type, "wan22")

    def test_vae_config(self):
        """Test VAE sub-config"""
        from paddleformers.transformers.wan22 import Wan22Config, Wan22VAEConfig

        config = Wan22Config(vae_config={"z_dim": 48, "c_dim": 160})

        self.assertIsInstance(config.vae_config, Wan22VAEConfig)
        self.assertEqual(config.vae_config.z_dim, 48)
        self.assertEqual(config.vae_config.c_dim, 160)

    def test_dit_config(self):
        """Test DiT sub-config"""
        from paddleformers.transformers.wan22 import Wan22Config, Wan22DiTConfig

        config = Wan22Config(dit_config={"dim": 3072, "num_heads": 24})

        self.assertIsInstance(config.dit_config, Wan22DiTConfig)
        self.assertEqual(config.dit_config.dim, 3072)
        self.assertEqual(config.dit_config.num_heads, 24)

    def test_config_to_dict(self):
        """Test config to_dict method"""
        from paddleformers.transformers.wan22 import Wan22Config

        config = Wan22Config(frame_num=49, sample_steps=30)
        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["frame_num"], 49)
        self.assertEqual(config_dict["sample_steps"], 30)
        self.assertEqual(config_dict["model_type"], "wan22")


class Wan22ModelClassTest(unittest.TestCase):
    """Tests for Wan22 model class structure (no checkpoint required)"""

    def test_model_classes_importable(self):
        """Test that all model classes can be imported"""
        from paddleformers.transformers.wan22 import (
            Wan22ForImageToVideo,
            Wan22ForTextToVideo,
            Wan22PretrainedModel,
        )

        # Verify inheritance
        self.assertTrue(issubclass(Wan22ForTextToVideo, Wan22PretrainedModel))
        self.assertTrue(issubclass(Wan22ForImageToVideo, Wan22PretrainedModel))

    def test_model_class_attributes(self):
        """Test model class attributes"""
        from paddleformers.transformers.wan22 import Wan22Config, Wan22ForTextToVideo

        # Check class attributes
        self.assertEqual(Wan22ForTextToVideo.config_class, Wan22Config)
        self.assertEqual(Wan22ForTextToVideo.base_model_prefix, "wan22")

    def test_model_size_configs(self):
        """Test model size configurations"""
        from paddleformers.transformers.wan22 import Wan22ForTextToVideo

        # Check size configs exist
        self.assertIn("1280*704", Wan22ForTextToVideo.SIZE_CONFIGS)
        self.assertIn("832*480", Wan22ForTextToVideo.SIZE_CONFIGS)
        self.assertIn("480*832", Wan22ForTextToVideo.SIZE_CONFIGS)

        # Check size values
        self.assertEqual(Wan22ForTextToVideo.SIZE_CONFIGS["1280*704"], (1280, 704))
        self.assertEqual(Wan22ForTextToVideo.SIZE_CONFIGS["832*480"], (832, 480))

    def test_model_init_with_config(self):
        """Test model initialization with config"""
        from paddleformers.transformers.wan22 import (
            Wan22Config,
            Wan22ForImageToVideo,
            Wan22ForTextToVideo,
        )

        config = Wan22Config(frame_num=49)

        t2v_model = Wan22ForTextToVideo(config)
        self.assertEqual(t2v_model.config.frame_num, 49)

        i2v_model = Wan22ForImageToVideo(config)
        self.assertEqual(i2v_model.config.frame_num, 49)

    def test_best_output_size(self):
        """Test _best_output_size static method"""
        from paddleformers.transformers.wan22 import Wan22PretrainedModel

        # Test aspect ratio calculation
        w, h = Wan22PretrainedModel._best_output_size(
            1920, 1080, 32, 32, 1280 * 704  # 16:9 input  # divisors  # expected area
        )

        # Result should be divisible by 32
        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)

        # Area should not exceed expected
        self.assertLessEqual(w * h, 1280 * 704)


class Wan22VideoOutputTest(unittest.TestCase):
    """Tests for Wan22VideoOutput dataclass"""

    def test_video_output_creation(self):
        """Test VideoOutput dataclass"""
        from paddleformers.transformers.wan22.modeling import Wan22VideoOutput

        video = np.zeros((49, 480, 832, 3), dtype=np.uint8)

        output = Wan22VideoOutput(
            video=video,
            prompt="test prompt",
            size=(832, 480),
            frame_num=49,
            seed=42,
        )

        self.assertEqual(output.prompt, "test prompt")
        self.assertEqual(output.size, (832, 480))
        self.assertEqual(output.frame_num, 49)
        self.assertEqual(output.seed, 42)
        self.assertEqual(output.video.shape, (49, 480, 832, 3))


class Wan22SaveVideoTest(unittest.TestCase):
    """Tests for video saving functionality"""

    def test_save_video_numpy(self):
        """Test save_video with numpy array"""
        from paddleformers.transformers.wan22 import Wan22PretrainedModel

        # Create dummy video (T, H, W, C)
        video = np.random.randint(0, 255, (5, 64, 64, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_video.mp4")
            Wan22PretrainedModel.save_video(video, output_path, fps=8)

            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_save_video_transposed(self):
        """Test save_video with (T, C, H, W) format"""
        from paddleformers.transformers.wan22 import Wan22PretrainedModel

        # Create video in (T, C, H, W) format
        video = np.random.randint(0, 255, (5, 3, 64, 64), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_video.mp4")
            Wan22PretrainedModel.save_video(video, output_path, fps=8)

            self.assertTrue(os.path.exists(output_path))

    def test_save_video_float(self):
        """Test save_video with float array [0, 1]"""
        from paddleformers.transformers.wan22 import Wan22PretrainedModel

        # Create float video [0, 1]
        video = np.random.rand(5, 64, 64, 3).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_video.mp4")
            Wan22PretrainedModel.save_video(video, output_path, fps=8)

            self.assertTrue(os.path.exists(output_path))


class Wan22AutoIntegrationTest(unittest.TestCase):
    """Tests for Auto module integration"""

    def test_auto_config_mapping(self):
        """Test that wan22 is in AutoConfig mapping"""
        from paddleformers.transformers.auto.configuration import CONFIG_MAPPING_NAMES

        self.assertIn("wan22", CONFIG_MAPPING_NAMES)
        self.assertEqual(CONFIG_MAPPING_NAMES["wan22"], "Wan22Config")

    def test_auto_processor_mapping(self):
        """Test that wan22 is in AutoProcessor mapping"""
        try:
            from paddleformers.transformers.auto.processing import (
                PROCESSOR_MAPPING_NAMES,
            )

            self.assertIn("wan22", PROCESSOR_MAPPING_NAMES)
            self.assertEqual(PROCESSOR_MAPPING_NAMES["wan22"], "Wan22Processor")
        except ImportError as e:
            # Skip if transformers version doesn't have required exports
            self.skipTest(f"Skipping due to transformers compatibility: {e}")

    def test_auto_video_processor_mapping(self):
        """Test that wan22 is in AutoVideoProcessor mapping"""
        try:
            from paddleformers.transformers.auto.video_processing import (
                VIDEO_PROCESSOR_MAPPING_NAMES,
            )

            self.assertIn("wan22", VIDEO_PROCESSOR_MAPPING_NAMES)
            self.assertEqual(VIDEO_PROCESSOR_MAPPING_NAMES["wan22"], "Wan22VideoProcessor")
        except (ImportError, ModuleNotFoundError) as e:
            # Skip if transformers version doesn't have video_processing_auto module
            self.skipTest(f"Skipping due to transformers compatibility: {e}")

    def test_auto_model_mapping(self):
        """Test that Wan22 is in AutoModel mapping"""
        from paddleformers.transformers.auto.modeling import MAPPING_NAMES

        self.assertIn("Wan22", dict(MAPPING_NAMES).keys())


class Wan22IntegrationTest(unittest.TestCase):
    """
    Integration tests for Wan2.2 models using diffusers backend.

    These tests require:
    - diffusers library with Wan support
    - GPU with sufficient VRAM (16GB+)
    - HuggingFace model access or local checkpoint

    Run manually with: pytest -k Integration -v
    """

    @unittest.skip("Requires GPU and diffusers - run manually")
    def test_t2v_generation_diffusers(self):
        """Test T2V generation with diffusers backend"""
        from paddleformers.transformers.wan22 import Wan22ForTextToVideo

        # Load from HuggingFace
        model = Wan22ForTextToVideo.from_pretrained("Wan-AI/Wan2.2-T2V-5B-Diffusers")

        output = model.generate(
            prompt="A cat walks on the grass, realistic style",
            size="832*480",
            frame_num=49,
            sampling_steps=30,
            seed=42,
        )

        self.assertIsNotNone(output.video)
        self.assertEqual(output.frame_num, 49)

    @unittest.skip("Requires GPU and diffusers - run manually")
    def test_i2v_generation_diffusers(self):
        """Test I2V generation with diffusers backend"""
        from PIL import Image

        from paddleformers.transformers.wan22 import Wan22ForImageToVideo

        # Load from HuggingFace
        model = Wan22ForImageToVideo.from_pretrained("Wan-AI/Wan2.2-I2V-5B-480P-Diffusers")

        # Create dummy image
        dummy_image = Image.new("RGB", (832, 480), color="blue")

        output = model.generate(
            image=dummy_image,
            prompt="The scene comes alive with gentle movement",
            size="832*480",
            frame_num=49,
            sampling_steps=30,
            seed=42,
        )

        self.assertIsNotNone(output.video)
