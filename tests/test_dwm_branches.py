import importlib.util
import unittest

import numpy as np

from manip.world_model.dwm.branches import (
    BRANCH_NAMES,
    HORIZON,
    make_action_branches,
    make_probe_sequences,
    make_target_action,
)
from manip.world_model.dwm.config import ACTION_DIM, object_configs_for_split


HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(HAS_MUJOCO, "mujoco is not installed")
class DWMActionBranchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from manip.world_model.dwm.env import DWMSimEnv

        cls.env = DWMSimEnv(object_configs_for_split("train")[0])
        cls.env.reset(reset_id=1)

    def test_branch_names_shapes_and_determinism(self):
        first = make_action_branches(self.env)
        second = make_action_branches(self.env)
        self.assertEqual(tuple(first), BRANCH_NAMES)
        for name in BRANCH_NAMES:
            self.assertEqual(first[name].shape, (HORIZON, ACTION_DIM))
            np.testing.assert_allclose(first[name], second[name])

    def test_push_changes_only_requested_wrist_channel(self):
        actions = make_action_branches(self.env)
        difference = actions["push_x+"] - actions["hold"]
        self.assertAlmostEqual(difference[0, 0], 0.010)
        np.testing.assert_allclose(difference[:, 1:], 0.0)

    def test_actions_contain_no_object_state(self):
        for action in make_action_branches(self.env).values():
            self.assertEqual(action.shape[-1], ACTION_DIM)
            self.assertTrue(np.isfinite(action).all())

    def test_probe_sequences_are_nested_and_distinct(self):
        actions = make_action_branches(self.env)
        fixed, random = make_probe_sequences(actions, seed=19)
        self.assertEqual(fixed.shape, (4, ACTION_DIM))
        self.assertEqual(random.shape, (4, ACTION_DIM))
        self.assertFalse(np.array_equal(fixed, random))
        fixed_again, random_again = make_probe_sequences(
            actions,
            seed=19,
        )
        np.testing.assert_array_equal(fixed, fixed_again)
        np.testing.assert_array_equal(random, random_again)

    def test_target_action_is_valid_and_outside_candidates(self):
        target = make_target_action(self.env, seed=20260922)
        self.assertEqual(target.shape, (HORIZON, ACTION_DIM))
        self.assertTrue(np.isfinite(target).all())
        for candidate in make_action_branches(self.env).values():
            self.assertGreater(
                float(np.max(np.abs(target - candidate))),
                1e-7,
            )


if __name__ == "__main__":
    unittest.main()
