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
"""
Test suite for Wan2.2 PaddlePaddle Native Implementation

This module tests all components of the native PaddlePaddle implementation:
- Configuration
- DiT Transformer components
- VAE (Encoder/Decoder)
- Schedulers
- Pipelines
"""
import unittest

import paddle

# Remove transformers to avoid conflicts
# sys.modules.pop('transformers', None)


class TestWan22Configuration(unittest.TestCase):
    """Test Wan22 configuration classes"""

    def test_config_creation(self):
        """Test basic config creation"""
        from paddleformers.transformers.wan22.configuration import Wan22Config

        config = Wan22Config()
        self.assertEqual(config.dit_num_layers, 30)
        self.assertEqual(config.dit_attention_head_dim, 128)
        self.assertEqual(config.dit_num_attention_heads, 24)

    def test_vae_config(self):
        """Test VAE config"""
        from paddleformers.transformers.wan22.configuration import Wan22VAEConfig

        vae_config = Wan22VAEConfig()
        self.assertEqual(vae_config.z_dim, 48)
        self.assertEqual(vae_config.c_dim, 160)


class TestWan22ModelingPaddle(unittest.TestCase):
    """Test native PaddlePaddle model components"""

    @classmethod
    def setUpClass(cls):
        paddle.device.set_device("cpu")

    def test_rmsnorm(self):
        """Test RMSNorm layer"""
        from paddleformers.transformers.wan22.modeling_paddle import Wan22RMSNorm

        norm = Wan22RMSNorm(256)
        x = paddle.randn([2, 10, 256])
        out = norm(x)

        self.assertEqual(out.shape, [2, 10, 256])

    def test_attention(self):
        """Test multi-head attention"""
        from paddleformers.transformers.wan22.modeling_paddle import Wan22Attention

        attn = Wan22Attention(dim=256, heads=4, dim_head=64)
        x = paddle.randn([2, 16, 256])
        out = attn(x)

        self.assertEqual(out.shape, [2, 16, 256])

    def test_feedforward(self):
        """Test feed-forward network"""
        from paddleformers.transformers.wan22.modeling_paddle import Wan22FeedForward

        ff = Wan22FeedForward(dim=256, hidden_dim=1024)
        x = paddle.randn([2, 16, 256])
        out = ff(x)

        self.assertEqual(out.shape, [2, 16, 256])

    def test_timesteps(self):
        """Test timestep embeddings"""
        from paddleformers.transformers.wan22.modeling_paddle import Wan22Timesteps

        ts = Wan22Timesteps(256)
        t = paddle.to_tensor([100, 500])
        emb = ts(t)

        self.assertEqual(emb.shape, [2, 256])

    def test_rotary_embeddings(self):
        """Test rotary position embeddings"""
        from paddleformers.transformers.wan22.modeling_paddle import (
            get_1d_rotary_pos_embed,
        )

        freqs_cos, freqs_sin = get_1d_rotary_pos_embed(64, 16, use_real=True)

        # Shape is [seq_len, dim] where dim = input_dim (tile happens internally)
        self.assertEqual(freqs_cos.shape, [16, 64])
        self.assertEqual(freqs_sin.shape, [16, 64])


class TestWan22VAE(unittest.TestCase):
    """Test VAE components"""

    @classmethod
    def setUpClass(cls):
        paddle.device.set_device("cpu")

    def test_causal_conv3d(self):
        """Test causal 3D convolution"""
        from paddleformers.transformers.wan22.vae import WanCausalConv3d

        conv = WanCausalConv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        x = paddle.randn([1, 3, 5, 16, 16])
        out = conv(x)

        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 64)

    def test_vae_model_creation(self):
        """Test VAE model instantiation"""
        from paddleformers.transformers.wan22.vae import Wan22VAEModel

        vae = Wan22VAEModel(
            in_channels=3,
            latent_channels=16,
            base_channels=32,
            channel_multipliers=(1, 2),
            temporal_downsample=(False,),
        )

        params = sum(p.numel() for p in vae.parameters())
        self.assertGreater(params, 0)


class TestWan22Scheduler(unittest.TestCase):
    """Test scheduler components"""

    def test_flow_match_scheduler(self):
        """Test Flow Matching scheduler"""
        from paddleformers.transformers.wan22.scheduler import Wan22FlowMatchScheduler

        scheduler = Wan22FlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        scheduler.set_timesteps(50)

        self.assertEqual(len(scheduler.timesteps), 50)
        self.assertGreater(scheduler.timesteps[0], scheduler.timesteps[-1])

    def test_unipc_scheduler(self):
        """Test UniPC scheduler"""
        from paddleformers.transformers.wan22.scheduler import Wan22UniPCScheduler

        scheduler = Wan22UniPCScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(50)

        self.assertEqual(len(scheduler.timesteps), 50)


class TestWan22Pipeline(unittest.TestCase):
    """Test pipeline components"""

    def test_size_configs(self):
        """Test size configurations"""
        from paddleformers.transformers.wan22.pipeline import SIZE_CONFIGS

        self.assertIn("1280*720", SIZE_CONFIGS)
        self.assertEqual(SIZE_CONFIGS["1280*720"], (1280, 720))

    def test_best_output_size(self):
        """Test aspect ratio calculation"""
        from paddleformers.transformers.wan22.pipeline import _best_output_size

        # Test 16:9 aspect ratio
        w, h = _best_output_size(1920, 1080)
        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)

        # Test 9:16 aspect ratio
        w, h = _best_output_size(1080, 1920)
        self.assertEqual(w % 32, 0)
        self.assertEqual(h % 32, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests"""

    @classmethod
    def setUpClass(cls):
        paddle.device.set_device("cpu")


def run_all_tests():
    """Run all tests and return summary"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestWan22Configuration))
    suite.addTests(loader.loadTestsFromTestCase(TestWan22ModelingPaddle))
    suite.addTests(loader.loadTestsFromTestCase(TestWan22VAE))
    suite.addTests(loader.loadTestsFromTestCase(TestWan22Scheduler))
    suite.addTests(loader.loadTestsFromTestCase(TestWan22Pipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result
