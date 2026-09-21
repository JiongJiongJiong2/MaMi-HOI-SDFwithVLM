import importlib.util
import unittest

import numpy as np

from manip.world_model.dwm.config import object_configs_for_split
from manip.world_model.dwm.schema import (
    CONTACT_MODE_LABELS,
    STATE_DIM,
)


HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(HAS_MUJOCO, "mujoco is not installed")
class DWMEnvironmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from manip.world_model.dwm.env import DWMSimEnv

        cls.env_class = DWMSimEnv
        cls.config = object_configs_for_split("train")[0]

    def test_model_and_state_contract(self):
        env = self.env_class(self.config)
        self.assertEqual(env.model.nu, 51)
        state = env.reset(reset_id=0)
        self.assertEqual(state.vector.shape, (STATE_DIM,))
        self.assertEqual(state.contact_forces.shape, (16, 3))
        self.assertTrue(np.isfinite(state.vector).all())

    def test_reset_is_deterministic(self):
        env = self.env_class(self.config)
        first = env.reset(reset_id=3)
        second = env.reset(reset_id=3)
        np.testing.assert_allclose(first.vector, second.vector, atol=1e-7)

    def test_hold_chunk_is_finite_and_advances(self):
        env = self.env_class(self.config)
        initial = env.reset(reset_id=4)
        targets = env.position_targets()
        states = env.rollout(np.repeat(targets[None], 4, axis=0))
        final = states[-1]
        self.assertTrue(np.isfinite(final.vector).all())
        self.assertIn(env.contact_mode(), CONTACT_MODE_LABELS)
        self.assertGreater(
            float(np.linalg.norm(final.vector - initial.vector)),
            0.0,
        )

    def test_checkpoint_restore_reproduces_branch(self):
        env = self.env_class(self.config)
        env.reset(reset_id=7)
        targets = env.position_targets()
        probe = np.repeat(targets[None], 2, axis=0)
        probe[0, 0] += 0.010
        env.rollout(probe)
        checkpoint = env.checkpoint()
        action = np.repeat(targets[None], 3, axis=0)
        action[:, 1] += 0.008

        first_states = env.rollout(action)
        first_mode = env.contact_mode_index
        env.restore(checkpoint)
        second_states = env.rollout(action)
        second_mode = env.contact_mode_index

        for first, second in zip(first_states, second_states):
            np.testing.assert_array_equal(first.vector, second.vector)
        self.assertEqual(first_mode, second_mode)


if __name__ == "__main__":
    unittest.main()
