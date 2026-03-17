# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""Tests for Wan22 Configuration classes"""
from __future__ import annotations

import os
import tempfile
import unittest

# Avoid transformers conflict
# sys.modules.pop('transformers', None)


class Wan22ConfigTest(unittest.TestCase):
    """Tests for Wan22Config"""

    def test_default_config(self):
        """Test default configuration values"""
        from paddleformers.transformers.wan22 import Wan22Config

        config = Wan22Config()

        self.assertEqual(config.model_type, "wan22")
        self.assertEqual(config.task, "ti2v-5B")
        self.assertEqual(config.text_len, 512)
        self.assertEqual(config.frame_num, 121)
        self.assertEqual(config.sample_fps, 24)
        self.assertEqual(config.sample_steps, 50)
        self.assertEqual(config.sample_shift, 5.0)
        self.assertEqual(config.sample_guide_scale, 5.0)

    def test_custom_config(self):
        """Test configuration with custom values"""
        from paddleformers.transformers.wan22 import Wan22Config

        config = Wan22Config(
            task="t2v-A14B",
            frame_num=81,
            sample_steps=30,
            sample_shift=3.0,
        )

        self.assertEqual(config.task, "t2v-A14B")
        self.assertEqual(config.frame_num, 81)
        self.assertEqual(config.sample_steps, 30)
        self.assertEqual(config.sample_shift, 3.0)

    def test_dit_config_defaults(self):
        """Test DiT configuration defaults"""
        from paddleformers.transformers.wan22 import Wan22Config

        config = Wan22Config()

        self.assertEqual(config.dit_num_layers, 30)
        self.assertEqual(config.dit_num_attention_heads, 24)
        self.assertEqual(config.dit_attention_head_dim, 128)

    def test_save_and_load(self):
        """Test config serialization"""
        from paddleformers.transformers.wan22 import Wan22Config

        config = Wan22Config(frame_num=49, sample_steps=25)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.save_pretrained(tmpdir)
            config_file = os.path.join(tmpdir, "config.json")
            self.assertTrue(os.path.exists(config_file))

            loaded = Wan22Config.from_pretrained(tmpdir)
            self.assertEqual(loaded.frame_num, 49)
            self.assertEqual(loaded.sample_steps, 25)

    def test_to_dict(self):
        """Test config to_dict method"""
        from paddleformers.transformers.wan22 import Wan22Config

        config = Wan22Config(frame_num=49)
        d = config.to_dict()

        self.assertIsInstance(d, dict)
        self.assertEqual(d["frame_num"], 49)
        self.assertEqual(d["model_type"], "wan22")


class Wan22VAEConfigTest(unittest.TestCase):
    """Tests for Wan22VAEConfig"""

    def test_default_values(self):
        """Test default VAE configuration"""
        from paddleformers.transformers.wan22 import Wan22VAEConfig

        config = Wan22VAEConfig()

        self.assertEqual(config.z_dim, 48)
        self.assertEqual(config.c_dim, 160)

    def test_custom_values(self):
        """Test custom VAE configuration"""
        from paddleformers.transformers.wan22 import Wan22VAEConfig

        config = Wan22VAEConfig(z_dim=32, c_dim=128)

        self.assertEqual(config.z_dim, 32)
        self.assertEqual(config.c_dim, 128)


class Wan22DiTConfigTest(unittest.TestCase):
    """Tests for Wan22DiTConfig"""

    def test_default_values(self):
        """Test default DiT configuration"""
        from paddleformers.transformers.wan22 import Wan22DiTConfig

        config = Wan22DiTConfig()

        self.assertEqual(config.dim, 3072)
        self.assertEqual(config.num_heads, 24)
        self.assertEqual(config.num_layers, 30)

    def test_custom_values(self):
        """Test custom DiT configuration"""
        from paddleformers.transformers.wan22 import Wan22DiTConfig

        config = Wan22DiTConfig(dim=2048, num_heads=16, num_layers=20)

        self.assertEqual(config.dim, 2048)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.num_layers, 20)


class Wan22ConfigSubConfigTest(unittest.TestCase):
    """Tests for sub-configs in Wan22Config"""

    def test_vae_sub_config_dict(self):
        """Test VAE sub-config from dict"""
        from paddleformers.transformers.wan22 import Wan22Config, Wan22VAEConfig

        config = Wan22Config(vae_config={"z_dim": 32, "c_dim": 128})

        self.assertIsInstance(config.vae_config, Wan22VAEConfig)
        self.assertEqual(config.vae_config.z_dim, 32)

    def test_dit_sub_config_dict(self):
        """Test DiT sub-config from dict"""
        from paddleformers.transformers.wan22 import Wan22Config, Wan22DiTConfig

        config = Wan22Config(dit_config={"dim": 2048, "num_heads": 16})

        self.assertIsInstance(config.dit_config, Wan22DiTConfig)
        self.assertEqual(config.dit_config.dim, 2048)


def run_configuration_tests():
    """Run all configuration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(Wan22ConfigTest))
    suite.addTests(loader.loadTestsFromTestCase(Wan22VAEConfigTest))
    suite.addTests(loader.loadTestsFromTestCase(Wan22DiTConfigTest))
    suite.addTests(loader.loadTestsFromTestCase(Wan22ConfigSubConfigTest))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)
