import unittest

import torch

from manip.world_model.dwm.model import DWMTransitionModel
from manip.world_model.dwm.schema import STATE_DIM


class DWMTransitionModelTest(unittest.TestCase):
    def test_output_shapes(self):
        model = DWMTransitionModel(hidden_size=32, max_horizon=4)
        state = torch.randn(3, STATE_DIM)
        actions = torch.randn(3, 4, 51)
        output = model(state, actions)
        self.assertEqual(output["object_delta"].shape, (3, 4, 9))
        self.assertEqual(
            output["contact_mode_logits"].shape,
            (3, 4, 4),
        )
        self.assertEqual(
            output["contact_impulse_normalized"].shape,
            (3, 4, 3),
        )

    def test_action_changes_prediction(self):
        torch.manual_seed(3)
        model = DWMTransitionModel(hidden_size=32, max_horizon=4)
        state = torch.randn(2, STATE_DIM)
        actions = torch.randn(2, 4, 51)
        first = model(state, actions)["object_delta"]
        second = model(state, torch.zeros_like(actions))["object_delta"]
        self.assertFalse(torch.allclose(first, second))


if __name__ == "__main__":
    unittest.main()
