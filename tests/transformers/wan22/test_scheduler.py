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
"""Tests for Wan22 Scheduler classes"""
from __future__ import annotations

import unittest

import paddle

# Avoid transformers conflict
# sys.modules.pop('transformers', None)


class Wan22FlowMatchSchedulerTest(unittest.TestCase):
    """Tests for Wan22FlowMatchScheduler"""

    @classmethod
    def setUpClass(cls):
        paddle.device.set_device("cpu")

    def get_scheduler(self, **kwargs):
        """Get scheduler instance"""
        from paddleformers.transformers.wan22 import Wan22FlowMatchScheduler

        return Wan22FlowMatchScheduler(**kwargs)

    def test_scheduler_creation(self):
        """Test scheduler instantiation"""
        scheduler = self.get_scheduler()

        self.assertEqual(scheduler.num_train_timesteps, 1000)
        self.assertEqual(scheduler.shift, 5.0)

    def test_scheduler_with_custom_params(self):
        """Test scheduler with custom parameters"""
        scheduler = self.get_scheduler(
            num_train_timesteps=500,
            shift=3.0,
        )

        self.assertEqual(scheduler.num_train_timesteps, 500)
        self.assertEqual(scheduler.shift, 3.0)

    def test_set_timesteps(self):
        """Test set_timesteps method"""
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(50)

        self.assertEqual(len(scheduler.timesteps), 50)
        # Timesteps should be decreasing
        for i in range(len(scheduler.timesteps) - 1):
            self.assertGreater(scheduler.timesteps[i], scheduler.timesteps[i + 1])

    def test_set_timesteps_different_values(self):
        """Test set_timesteps with different values"""
        scheduler = self.get_scheduler()

        for num_steps in [10, 25, 50, 100]:
            scheduler.set_timesteps(num_steps)
            self.assertEqual(len(scheduler.timesteps), num_steps)

    def test_timesteps_range(self):
        """Test timesteps are within valid range"""
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(50)

        for t in scheduler.timesteps:
            t_val = float(t) if hasattr(t, "numpy") else t
            self.assertGreaterEqual(t_val, 0)
            self.assertLessEqual(t_val, scheduler.num_train_timesteps)

    def test_step_function(self):
        """Test scheduler step function"""
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(50)

        # Create dummy model output and sample
        model_output = paddle.randn([1, 4, 8, 8])
        sample = paddle.randn([1, 4, 8, 8])

        timestep = scheduler.timesteps[0]

        output = scheduler.step(model_output, timestep, sample)

        # Check output has prev_sample attribute
        self.assertTrue(hasattr(output, "prev_sample"))
        self.assertEqual(list(output.prev_sample.shape), list(sample.shape))

    def test_step_output_type(self):
        """Test scheduler step returns SchedulerOutput"""
        from paddleformers.transformers.wan22 import SchedulerOutput

        scheduler = self.get_scheduler()
        scheduler.set_timesteps(50)

        model_output = paddle.randn([1, 4, 8, 8])
        sample = paddle.randn([1, 4, 8, 8])

        output = scheduler.step(model_output, scheduler.timesteps[0], sample)

        self.assertIsInstance(output, (dict, SchedulerOutput))


class Wan22UniPCSchedulerTest(unittest.TestCase):
    """Tests for Wan22UniPCScheduler"""

    @classmethod
    def setUpClass(cls):
        paddle.device.set_device("cpu")

    def get_scheduler(self, **kwargs):
        """Get scheduler instance"""
        from paddleformers.transformers.wan22 import Wan22UniPCScheduler

        return Wan22UniPCScheduler(**kwargs)

    def test_scheduler_creation(self):
        """Test scheduler instantiation"""
        scheduler = self.get_scheduler()

        self.assertEqual(scheduler.num_train_timesteps, 1000)

    def test_set_timesteps(self):
        """Test set_timesteps method"""
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(50)

        self.assertEqual(len(scheduler.timesteps), 50)

    def test_beta_schedule_linear(self):
        """Test linear beta schedule"""
        scheduler = self.get_scheduler(beta_schedule="linear")

        self.assertIsNotNone(scheduler.betas)

    def test_beta_schedule_scaled_linear(self):
        """Test scaled_linear beta schedule"""
        scheduler = self.get_scheduler(beta_schedule="scaled_linear")

        self.assertIsNotNone(scheduler.betas)

    def test_step_function(self):
        """Test scheduler step function"""
        scheduler = self.get_scheduler()
        scheduler.set_timesteps(50)

        model_output = paddle.randn([1, 4, 8, 8])
        sample = paddle.randn([1, 4, 8, 8])

        timestep = scheduler.timesteps[0]

        output = scheduler.step(model_output, timestep, sample)

        # Check output has prev_sample attribute
        self.assertTrue(hasattr(output, "prev_sample"))


class SchedulerOutputTest(unittest.TestCase):
    """Tests for SchedulerOutput"""

    def test_scheduler_output_creation(self):
        """Test SchedulerOutput creation"""
        from paddleformers.transformers.wan22 import SchedulerOutput

        prev_sample = paddle.randn([1, 4, 8, 8])
        output = SchedulerOutput(prev_sample=prev_sample)

        self.assertIsNotNone(output.prev_sample)
        self.assertEqual(output.prev_sample.shape, [1, 4, 8, 8])

    def test_scheduler_output_dict_access(self):
        """Test SchedulerOutput attribute access"""
        from paddleformers.transformers.wan22 import SchedulerOutput

        prev_sample = paddle.randn([1, 4, 8, 8])
        output = SchedulerOutput(prev_sample=prev_sample)

        # SchedulerOutput is a dataclass with prev_sample attribute
        self.assertTrue(hasattr(output, "prev_sample"))
        self.assertEqual(list(output.prev_sample.shape), [1, 4, 8, 8])


class SchedulerComparisonTest(unittest.TestCase):
    """Tests comparing different schedulers"""

    @classmethod
    def setUpClass(cls):
        paddle.device.set_device("cpu")

    def test_both_schedulers_same_steps(self):
        """Test both schedulers produce same number of steps"""
        from paddleformers.transformers.wan22 import (
            Wan22FlowMatchScheduler,
            Wan22UniPCScheduler,
        )

        flow_scheduler = Wan22FlowMatchScheduler()
        unipc_scheduler = Wan22UniPCScheduler()

        flow_scheduler.set_timesteps(50)
        unipc_scheduler.set_timesteps(50)

        self.assertEqual(len(flow_scheduler.timesteps), len(unipc_scheduler.timesteps))

    def test_schedulers_different_output_shapes(self):
        """Test both schedulers preserve output shape"""
        from paddleformers.transformers.wan22 import (
            Wan22FlowMatchScheduler,
            Wan22UniPCScheduler,
        )

        sample_shape = [2, 16, 32, 32]
        sample = paddle.randn(sample_shape)
        model_output = paddle.randn(sample_shape)

        # FlowMatch
        flow_scheduler = Wan22FlowMatchScheduler()
        flow_scheduler.set_timesteps(50)
        flow_output = flow_scheduler.step(model_output, flow_scheduler.timesteps[0], sample)

        # UniPC
        unipc_scheduler = Wan22UniPCScheduler()
        unipc_scheduler.set_timesteps(50)
        unipc_output = unipc_scheduler.step(model_output, unipc_scheduler.timesteps[0], sample)

        # Use attribute access instead of dict access
        self.assertEqual(list(flow_output.prev_sample.shape), sample_shape)
        self.assertEqual(list(unipc_output.prev_sample.shape), sample_shape)


def run_scheduler_tests():
    """Run all scheduler tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(Wan22FlowMatchSchedulerTest))
    suite.addTests(loader.loadTestsFromTestCase(Wan22UniPCSchedulerTest))
    suite.addTests(loader.loadTestsFromTestCase(SchedulerOutputTest))
    suite.addTests(loader.loadTestsFromTestCase(SchedulerComparisonTest))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)
