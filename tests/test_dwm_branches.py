import importlib.util
import unittest

import numpy as np

from manip.world_model.dwm.branches import (
    BRANCH_NAMES,
    HORIZON,
    make_action_branches,
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


if __name__ == "__main__":
    unittest.main()
