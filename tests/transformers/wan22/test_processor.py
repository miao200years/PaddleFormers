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
"""Tests for Wan22Processor"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np

# Avoid transformers conflict
# sys.modules.pop('transformers', None)


class Wan22ProcessorTest(unittest.TestCase):
    """Tests for Wan22Processor class"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        """Clean up"""
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def get_processor(self, **kwargs):
        """Get processor instance"""
        from paddleformers.transformers.wan22 import Wan22Processor

        return Wan22Processor(**kwargs)

    def get_image_processor(self, **kwargs):
        """Get image processor instance"""
        from paddleformers.transformers.wan22 import Wan22ImageProcessor

        return Wan22ImageProcessor(**kwargs)

    def test_processor_creation(self):
        """Test basic processor instantiation"""
        processor = self.get_processor()

        self.assertIsNotNone(processor)
        self.assertIsNotNone(processor.image_processor)
        self.assertEqual(processor.default_size, "832*480")
        self.assertEqual(processor.default_num_frames, 81)

    def test_processor_with_text_only(self):
        """Test processor with text-only input (T2V)"""
        processor = self.get_processor()

        output = processor(
            prompt="A cat walking on grass",
            size="832*480",
            num_frames=25,
        )

        self.assertEqual(output.prompt, "A cat walking on grass")
        self.assertEqual(output.width, 832)
        self.assertEqual(output.height, 480)
        self.assertEqual(output.num_frames, 25)
        self.assertIsNone(output.pixel_values)

    def test_processor_with_image(self):
        """Test processor with image input (I2V)"""
        from PIL import Image

        processor = self.get_processor()

        # Create test image
        test_img = Image.fromarray(np.random.randint(0, 255, (480, 832, 3), dtype=np.uint8))

        output = processor(
            prompt="The scene comes alive",
            images=test_img,
            num_frames=25,
        )

        self.assertEqual(output.prompt, "The scene comes alive")
        self.assertIsNotNone(output.pixel_values)
        self.assertEqual(output.pixel_values.shape[0], 1)  # batch
        self.assertEqual(output.pixel_values.shape[1], 3)  # channels
        self.assertEqual(output.num_frames, 25)

    def test_size_parsing(self):
        """Test size string parsing"""
        processor = self.get_processor()

        # Test string format
        w, h = processor._parse_size("1280*720")
        self.assertEqual((w, h), (1280, 720))

        # Test tuple format
        w, h = processor._parse_size((960, 540))
        self.assertEqual((w, h), (960, 540))

        # Test SIZE_CONFIGS key
        w, h = processor._parse_size("480*832")
        self.assertEqual((w, h), (480, 832))

    def test_num_frames_validation(self):
        """Test num_frames validation (must be 4n+1)"""
        processor = self.get_processor()

        # Already valid
        self.assertEqual(processor._validate_num_frames(25), 25)
        self.assertEqual(processor._validate_num_frames(49), 49)
        self.assertEqual(processor._validate_num_frames(81), 81)

        # Needs adjustment
        self.assertEqual(processor._validate_num_frames(24), 21)  # (24-1)//4*4+1 = 21
        self.assertEqual(processor._validate_num_frames(26), 25)  # (26-1)//4*4+1 = 25

        # Minimum value
        self.assertEqual(processor._validate_num_frames(3), 5)

    def test_negative_prompt_default(self):
        """Test default negative prompt"""
        from paddleformers.transformers.wan22 import Wan22Config

        config = Wan22Config()
        processor = self.get_processor(config=config)

        output = processor(prompt="test")

        self.assertIsNotNone(output.negative_prompt)
        self.assertEqual(output.negative_prompt, config.sample_neg_prompt)

    def test_output_to_dict(self):
        """Test Wan22ProcessorOutput.to_dict()"""
        processor = self.get_processor()

        output = processor(
            prompt="test",
            size="832*480",
            num_frames=25,
        )

        d = output.to_dict()

        self.assertIn("prompt", d)
        self.assertIn("width", d)
        self.assertIn("height", d)
        self.assertIn("num_frames", d)
        self.assertNotIn("pixel_values", d)  # None values excluded

    def test_save_pretrained(self):
        """Test processor save_pretrained"""
        processor = self.get_processor()

        save_path = os.path.join(self.tmpdir, "processor")
        processor.save_pretrained(save_path)

        # Check config file exists
        config_path = os.path.join(save_path, "preprocessor_config.json")
        self.assertTrue(os.path.exists(config_path))

        # Load and verify
        import json

        with open(config_path) as f:
            config = json.load(f)

        self.assertEqual(config["processor_class"], "Wan22Processor")
        self.assertIn("image_processor", config)


class Wan22ImageProcessorTest(unittest.TestCase):
    """Tests for Wan22ImageProcessor class"""

    def get_image_processor(self, **kwargs):
        """Get image processor instance"""
        from paddleformers.transformers.wan22 import Wan22ImageProcessor

        return Wan22ImageProcessor(**kwargs)

    def test_image_processor_creation(self):
        """Test image processor instantiation"""
        ip = self.get_image_processor()

        self.assertTrue(ip.do_resize)
        self.assertTrue(ip.do_normalize)
        self.assertEqual(ip.image_mean, [0.5, 0.5, 0.5])
        self.assertEqual(ip.image_std, [0.5, 0.5, 0.5])

    def test_best_output_size_landscape(self):
        """Test best_output_size for landscape images"""
        from paddleformers.transformers.wan22 import Wan22ImageProcessor

        # 16:9 aspect ratio
        w, h = Wan22ImageProcessor.best_output_size(1920, 1080)

        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)
        # Should maintain roughly 16:9 ratio
        ratio = w / h
        self.assertAlmostEqual(ratio, 16 / 9, delta=0.2)

    def test_best_output_size_portrait(self):
        """Test best_output_size for portrait images"""
        from paddleformers.transformers.wan22 import Wan22ImageProcessor

        # 9:16 aspect ratio
        w, h = Wan22ImageProcessor.best_output_size(1080, 1920)

        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)
        # Should maintain roughly 9:16 ratio
        ratio = w / h
        self.assertAlmostEqual(ratio, 9 / 16, delta=0.2)

    def test_best_output_size_square(self):
        """Test best_output_size for square images"""
        from paddleformers.transformers.wan22 import Wan22ImageProcessor

        w, h = Wan22ImageProcessor.best_output_size(1000, 1000)

        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)
        self.assertEqual(w, h)  # Should remain square

    def test_process_single_image(self):
        """Test processing a single image"""
        from PIL import Image

        ip = self.get_image_processor()

        # Create test image
        img = Image.fromarray(np.random.randint(0, 255, (480, 832, 3), dtype=np.uint8))

        result = ip(img)

        self.assertIn("pixel_values", result)
        self.assertIn("image_sizes", result)

        # Check shape: (B, C, H, W)
        pv = result["pixel_values"]
        self.assertEqual(pv.shape[0], 1)  # batch
        self.assertEqual(pv.shape[1], 3)  # channels

    def test_process_multiple_images(self):
        """Test processing multiple images"""
        from PIL import Image

        ip = self.get_image_processor()

        # Create test images
        imgs = [
            Image.fromarray(np.random.randint(0, 255, (480, 832, 3), dtype=np.uint8)),
            Image.fromarray(np.random.randint(0, 255, (480, 832, 3), dtype=np.uint8)),
        ]

        result = ip(imgs)

        self.assertEqual(result["pixel_values"].shape[0], 2)  # batch of 2

    def test_normalize_range(self):
        """Test that normalization produces correct range"""
        from PIL import Image

        ip = self.get_image_processor()

        # Create uniform image
        img = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))

        result = ip(img)
        pv = result["pixel_values"]

        # With mean=0.5, std=0.5, value 128/255≈0.5 should map to ~0
        self.assertTrue(np.abs(pv.mean()) < 0.1)

    def test_resize_to_specific_size(self):
        """Test resizing to specific size"""
        from PIL import Image

        ip = self.get_image_processor()

        img = Image.fromarray(np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8))

        result = ip(img, size=(640, 480))

        # Output size should match requested
        self.assertEqual(result["image_sizes"][0], (640, 480))
        self.assertEqual(result["pixel_values"].shape[2], 480)  # height
        self.assertEqual(result["pixel_values"].shape[3], 640)  # width


class Wan22ProcessorOutputTest(unittest.TestCase):
    """Tests for Wan22ProcessorOutput dataclass"""

    def test_output_creation(self):
        """Test output dataclass creation"""
        from paddleformers.transformers.wan22 import Wan22ProcessorOutput

        output = Wan22ProcessorOutput(
            prompt="test prompt",
            width=832,
            height=480,
            num_frames=25,
        )

        self.assertEqual(output.prompt, "test prompt")
        self.assertEqual(output.width, 832)
        self.assertEqual(output.height, 480)
        self.assertEqual(output.num_frames, 25)

    def test_output_defaults(self):
        """Test default values"""
        from paddleformers.transformers.wan22 import Wan22ProcessorOutput

        output = Wan22ProcessorOutput()

        self.assertIsNone(output.prompt)
        self.assertIsNone(output.pixel_values)
        self.assertIsNone(output.width)

    def test_output_to_dict_excludes_none(self):
        """Test to_dict excludes None values"""
        from paddleformers.transformers.wan22 import Wan22ProcessorOutput

        output = Wan22ProcessorOutput(
            prompt="test",
            width=832,
        )

        d = output.to_dict()

        self.assertIn("prompt", d)
        self.assertIn("width", d)
        self.assertNotIn("height", d)
        self.assertNotIn("pixel_values", d)


class Wan22SizeConfigsTest(unittest.TestCase):
    """Tests for SIZE_CONFIGS"""

    def test_size_configs_exist(self):
        """Test SIZE_CONFIGS is available"""
        from paddleformers.transformers.wan22 import SIZE_CONFIGS

        self.assertIsInstance(SIZE_CONFIGS, dict)
        self.assertGreater(len(SIZE_CONFIGS), 0)

    def test_size_configs_values(self):
        """Test SIZE_CONFIGS values are valid"""
        from paddleformers.transformers.wan22 import SIZE_CONFIGS

        expected_keys = ["1280*720", "720*1280", "832*480", "480*832"]

        for key in expected_keys:
            self.assertIn(key, SIZE_CONFIGS)
            w, h = SIZE_CONFIGS[key]
            self.assertIsInstance(w, int)
            self.assertIsInstance(h, int)
            self.assertGreater(w, 0)
            self.assertGreater(h, 0)

    def test_size_configs_divisibility(self):
        """Test all sizes are divisible by 32"""
        from paddleformers.transformers.wan22 import SIZE_CONFIGS

        for key, (w, h) in SIZE_CONFIGS.items():
            # Most video models require dimensions divisible by 16 or 32
            self.assertEqual(w % 16, 0, f"{key}: width {w} not divisible by 16")
            self.assertEqual(h % 16, 0, f"{key}: height {h} not divisible by 16")


def run_processor_tests():
    """Run all processor tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(Wan22ProcessorTest))
    suite.addTests(loader.loadTestsFromTestCase(Wan22ImageProcessorTest))
    suite.addTests(loader.loadTestsFromTestCase(Wan22ProcessorOutputTest))
    suite.addTests(loader.loadTestsFromTestCase(Wan22SizeConfigsTest))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)
