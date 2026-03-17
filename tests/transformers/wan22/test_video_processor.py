# coding=utf-8
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2025 The Wan Team. All rights reserved.
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
"""Tests for Wan22VideoProcessor - Training Support"""
from __future__ import annotations

import shutil
import tempfile
import unittest

import numpy as np

# Avoid transformers conflict
# sys.modules.pop('transformers', None)

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


class Wan22VideoProcessorTester:
    """
    Helper class for testing Wan22VideoProcessor.
    Defines configuration and expected behaviors.
    """

    def __init__(
        self,
        parent,
        batch_size=2,
        num_frames=17,
        num_channels=3,
        height=240,
        width=320,
        do_resize=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        patch_size=2,
        temporal_patch_size=4,
        vae_stride=16,
        merge_size=2,
        fps=24,
        min_frames=5,
        max_frames=81,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.do_resize = do_resize
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.vae_stride = vae_stride
        self.merge_size = merge_size
        self.fps = fps
        self.min_frames = min_frames
        self.max_frames = max_frames

    def prepare_video_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "patch_size": self.patch_size,
            "temporal_patch_size": self.temporal_patch_size,
            "vae_stride": self.vae_stride,
            "merge_size": self.merge_size,
            "fps": self.fps,
            "min_frames": self.min_frames,
            "max_frames": self.max_frames,
        }

    def prepare_video_inputs(self, num_videos=None, num_frames=None):
        """Create random video inputs for testing"""
        num_videos = num_videos or self.batch_size
        num_frames = num_frames or self.num_frames

        videos = []
        for _ in range(num_videos):
            video = np.random.rand(num_frames, self.height, self.width, 3).astype(np.float32)
            videos.append(video)

        return videos

    def prepare_pil_video_inputs(self, num_videos=None, num_frames=None):
        """Create PIL Image video inputs for testing"""
        if not HAS_PIL:
            return None

        num_videos = num_videos or self.batch_size
        num_frames = num_frames or self.num_frames

        videos = []
        for _ in range(num_videos):
            frames = []
            for _ in range(num_frames):
                img = Image.fromarray(np.random.randint(0, 255, (self.height, self.width, 3), dtype=np.uint8))
                frames.append(img)
            videos.append(frames)

        return videos


class Wan22VideoProcessorTest(unittest.TestCase):
    """Tests for Wan22VideoProcessor class"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        self.video_processor_tester = Wan22VideoProcessorTester(self)

    def get_video_processor(self, **kwargs):
        """Get video processor instance"""
        from paddleformers.transformers.wan22.video_processor import Wan22VideoProcessor

        return Wan22VideoProcessor(**kwargs)

    # ==================== Basic Tests ====================

    def test_video_processor_creation(self):
        """Test basic video processor instantiation"""
        processor = self.get_video_processor()

        self.assertIsNotNone(processor)
        self.assertEqual(processor.patch_size, 2)
        self.assertEqual(processor.temporal_patch_size, 4)
        self.assertEqual(processor.vae_stride, 16)
        self.assertEqual(processor.merge_size, 2)
        self.assertEqual(processor.fps, 24)
        self.assertEqual(processor.min_frames, 5)
        self.assertEqual(processor.max_frames, 81)

    def test_video_processor_with_custom_params(self):
        """Test video processor with custom parameters"""
        processor = self.get_video_processor(
            patch_size=4,
            temporal_patch_size=2,
            fps=30,
        )

        self.assertEqual(processor.patch_size, 4)
        self.assertEqual(processor.temporal_patch_size, 2)
        self.assertEqual(processor.fps, 30)

    # ==================== smart_resize Tests ====================

    def test_smart_resize_basic(self):
        """Test smart_resize function"""
        from paddleformers.transformers.wan22.video_processor import smart_resize

        h, w, t = smart_resize(49, 480, 832, temporal_factor=4, factor=32)

        # Check divisibility
        self.assertEqual(h % 32, 0)
        self.assertEqual(w % 32, 0)
        # Check frame adjustment to 4n+1
        self.assertEqual((t - 1) % 4, 0)

    def test_smart_resize_various_inputs(self):
        """Test smart_resize with various input sizes"""
        from paddleformers.transformers.wan22.video_processor import smart_resize

        test_cases = [
            (49, 480, 832),  # Standard
            (81, 720, 1280),  # Larger
            (17, 240, 320),  # Smaller
            (25, 1080, 1920),  # HD
        ]

        for frames, height, width in test_cases:
            h, w, t = smart_resize(frames, height, width, temporal_factor=4, factor=32)

            self.assertEqual(h % 32, 0, f"Height {h} not divisible by 32")
            self.assertEqual(w % 32, 0, f"Width {w} not divisible by 32")
            self.assertEqual((t - 1) % 4, 0, f"Frames {t} not 4n+1")

    def test_smart_resize_min_values(self):
        """Test smart_resize with minimum values"""
        from paddleformers.transformers.wan22.video_processor import smart_resize

        with self.assertRaises(ValueError):
            smart_resize(10, 16, 16, temporal_factor=4, factor=32)  # Too small

    # ==================== compute_grid_thw Tests ====================

    def test_compute_grid_thw_basic(self):
        """Test compute_grid_thw method"""
        processor = self.get_video_processor()

        grid_t, grid_h, grid_w = processor.compute_grid_thw(49, 480, 832)

        # Expected: t=(49-1)/4+1=13, h=480/16=30, w=832/16=52
        self.assertEqual(grid_t, 13)
        self.assertEqual(grid_h, 30)
        self.assertEqual(grid_w, 52)

    def test_compute_grid_thw_various_sizes(self):
        """Test compute_grid_thw with various sizes"""
        processor = self.get_video_processor()

        test_cases = [
            (49, 480, 832, (13, 30, 52)),
            (81, 720, 1280, (21, 45, 80)),
            (17, 256, 512, (5, 16, 32)),
        ]

        for frames, height, width, expected in test_cases:
            grid = processor.compute_grid_thw(frames, height, width)
            self.assertEqual(grid, expected, f"Failed for ({frames}, {height}, {width})")

    # ==================== Video Processing Tests ====================

    def test_process_single_video_numpy(self):
        """Test processing a single numpy video"""
        processor = self.get_video_processor()

        # Create single video (T, H, W, C)
        video = np.random.rand(17, 240, 320, 3).astype(np.float32)

        result = processor(video, return_metadata=True)

        self.assertIn("pixel_values_videos", result)
        self.assertIn("video_grid_thw", result)
        self.assertIn("video_metadata", result)

        # Check shapes
        self.assertEqual(len(result["pixel_values_videos"].shape), 5)  # (B, T, C, H, W)
        self.assertEqual(result["pixel_values_videos"].shape[0], 1)  # batch=1

    def test_process_batch_videos(self):
        """Test processing a batch of videos"""
        processor = self.get_video_processor()
        tester = self.video_processor_tester

        videos = tester.prepare_video_inputs(num_videos=3, num_frames=17)

        result = processor(videos, return_metadata=True)

        self.assertEqual(result["pixel_values_videos"].shape[0], 3)  # batch=3
        self.assertEqual(len(result["video_metadata"]), 3)

    @unittest.skipIf(not HAS_PIL, "PIL not available")
    def test_process_pil_video(self):
        """Test processing PIL Image video"""
        processor = self.get_video_processor()
        tester = self.video_processor_tester

        videos = tester.prepare_pil_video_inputs(num_videos=1, num_frames=9)

        result = processor(videos, return_metadata=True)

        self.assertIn("pixel_values_videos", result)
        self.assertEqual(result["pixel_values_videos"].shape[0], 1)

    def test_process_with_target_frames(self):
        """Test processing with target_frames parameter"""
        processor = self.get_video_processor()

        # Create video with more frames than target
        video = np.random.rand(50, 240, 320, 3).astype(np.float32)

        result = processor(video, target_frames=17, return_metadata=True)

        # Check metadata contains sampled indices
        metadata = result["video_metadata"][0]
        self.assertEqual(len(metadata.frames_indices), 17)

    def test_process_with_target_size(self):
        """Test processing with target_size parameter"""
        processor = self.get_video_processor()

        video = np.random.rand(17, 240, 320, 3).astype(np.float32)

        result = processor(video, target_size=(480, 832), return_metadata=True)

        # Check output dimensions match target
        pixel_values = result["pixel_values_videos"]
        # Shape: (B, T, C, H, W)
        self.assertEqual(pixel_values.shape[3], 480)  # height
        self.assertEqual(pixel_values.shape[4], 832)  # width

    # ==================== Metadata Tests ====================

    def test_video_metadata(self):
        """Test Wan22VideoMetadata class"""
        from paddleformers.transformers.wan22.video_processor import Wan22VideoMetadata

        metadata = Wan22VideoMetadata(
            total_num_frames=49,
            fps=24.0,
            height=480,
            width=832,
        )

        self.assertEqual(metadata.total_num_frames, 49)
        self.assertEqual(metadata.fps, 24.0)
        self.assertEqual(metadata.height, 480)
        self.assertEqual(metadata.width, 832)

    def test_sample_frames(self):
        """Test frame sampling method"""
        processor = self.get_video_processor()
        from paddleformers.transformers.wan22.video_processor import Wan22VideoMetadata

        metadata = Wan22VideoMetadata(total_num_frames=100, fps=30)

        indices = processor.sample_frames(metadata, num_frames=25)

        self.assertEqual(len(indices), 25)
        self.assertTrue(indices[0] >= 0)
        self.assertTrue(indices[-1] <= 99)

    # ==================== Normalization Tests ====================

    def test_normalize(self):
        """Test normalization method"""
        processor = self.get_video_processor()

        # Create video in [0, 1] range
        video = np.random.rand(5, 3, 100, 100).astype(np.float32)  # (T, C, H, W)

        normalized = processor.normalize(video)

        # After normalization with mean=0.5, std=0.5, range should be [-1, 1]
        self.assertTrue(normalized.min() >= -2.0)  # Allow some tolerance
        self.assertTrue(normalized.max() <= 2.0)

    # ==================== Return Tensors Tests ====================

    @unittest.skipIf(not HAS_PADDLE, "Paddle not available")
    def test_return_paddle_tensors(self):
        """Test returning paddle tensors"""
        processor = self.get_video_processor()

        video = np.random.rand(17, 240, 320, 3).astype(np.float32)

        result = processor(video, return_tensors="pd")

        self.assertIsInstance(result["pixel_values_videos"], paddle.Tensor)
        self.assertIsInstance(result["video_grid_thw"], paddle.Tensor)

    def test_return_numpy_arrays(self):
        """Test returning numpy arrays (default)"""
        processor = self.get_video_processor()

        video = np.random.rand(17, 240, 320, 3).astype(np.float32)

        result = processor(video, return_tensors="np")

        self.assertIsInstance(result["pixel_values_videos"], np.ndarray)
        self.assertIsInstance(result["video_grid_thw"], np.ndarray)

    # ==================== get_number_of_video_patches Tests ====================

    def test_get_number_of_video_patches(self):
        """Test get_number_of_video_patches method"""
        processor = self.get_video_processor()

        num_patches = processor.get_number_of_video_patches(49, 480, 832)

        # Expected: 13 * 30 * 52 = 20280
        expected = 13 * 30 * 52
        self.assertEqual(num_patches, expected)


class Wan22TrainingProcessorTest(unittest.TestCase):
    """Tests for Wan22TrainingProcessor class"""

    def get_training_processor(self, **kwargs):
        """Get training processor instance"""
        from paddleformers.transformers.wan22.video_processor import (
            Wan22TrainingProcessor,
        )

        return Wan22TrainingProcessor(**kwargs)

    def test_training_processor_creation(self):
        """Test training processor instantiation"""
        processor = self.get_training_processor()

        self.assertIsNotNone(processor)
        self.assertIsNotNone(processor.video_processor)
        self.assertIn("flow_shift", processor.noise_scheduler_config)
        self.assertEqual(processor.noise_scheduler_config["flow_shift"], 5.0)

    def test_compute_snr_weights(self):
        """Test SNR weight computation"""
        processor = self.get_training_processor()

        timesteps = np.array([100, 300, 500, 700, 900])

        weights = processor.compute_snr_weights(timesteps, snr_gamma=5.0)

        self.assertEqual(len(weights), 5)
        # Weights should decrease as timesteps increase
        self.assertTrue(weights[0] > weights[-1])

    def test_custom_noise_scheduler_config(self):
        """Test with custom noise scheduler config"""
        custom_config = {
            "num_train_timesteps": 500,
            "beta_start": 0.0001,
            "beta_end": 0.02,
            "flow_shift": 3.0,
        }

        processor = self.get_training_processor(noise_scheduler_config=custom_config)

        self.assertEqual(processor.noise_scheduler_config["num_train_timesteps"], 500)
        self.assertEqual(processor.noise_scheduler_config["flow_shift"], 3.0)


class Wan22ProcessorTrainingMethodsTest(unittest.TestCase):
    """Tests for training methods in Wan22Processor"""

    def get_processor(self, **kwargs):
        """Get processor instance"""
        from paddleformers.transformers.wan22.processor import Wan22Processor

        return Wan22Processor(**kwargs)

    def test_compute_loss_weights(self):
        """Test compute_loss_weights method"""
        processor = self.get_processor()

        timesteps = np.array([100, 300, 500, 700, 900])

        weights = processor.compute_loss_weights(timesteps, snr_gamma=5.0)

        self.assertEqual(len(weights), 5)
        self.assertTrue(all(w > 0 for w in weights))

    def test_compute_loss_weights_min_snr(self):
        """Test compute_loss_weights with min_snr_gamma=True"""
        processor = self.get_processor()

        timesteps = np.array([100, 500, 900])

        weights_min = processor.compute_loss_weights(timesteps, snr_gamma=5.0, min_snr_gamma=True)
        weights_no_min = processor.compute_loss_weights(timesteps, snr_gamma=5.0, min_snr_gamma=False)

        # Both should be valid
        self.assertEqual(len(weights_min), 3)
        self.assertEqual(len(weights_no_min), 3)

    def test_get_training_collate_fn(self):
        """Test get_training_collate_fn method"""
        processor = self.get_processor()

        collate_fn = processor.get_training_collate_fn()

        self.assertTrue(callable(collate_fn))

        # Test with sample batch
        batch = [
            {"pixel_values": np.random.rand(3, 64, 64), "prompt": "test1"},
            {"pixel_values": np.random.rand(3, 64, 64), "prompt": "test2"},
        ]

        collated = collate_fn(batch)

        self.assertIn("pixel_values", collated)
        self.assertIn("prompt", collated)
        self.assertEqual(collated["pixel_values"].shape[0], 2)

    def test_model_input_names(self):
        """Test model_input_names property"""
        processor = self.get_processor()

        input_names = processor.model_input_names

        self.assertIsInstance(input_names, list)
        self.assertIn("pixel_values", input_names)
        self.assertIn("input_ids", input_names)
        self.assertIn("timesteps", input_names)
