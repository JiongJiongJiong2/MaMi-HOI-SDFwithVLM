import unittest

import torch

from manip.world_model.consistency import (
    ForwardConsequenceModel,
    InverseActionModel,
    build_consequences,
    build_hand_actions,
    decode_forward_consequences,
)


class ForwardInverseConsistencyTest(unittest.TestCase):
    def test_shapes(self):
        forward = ForwardConsequenceModel(max_horizon=4)
        inverse = InverseActionModel(max_horizon=4)
        state_history = torch.randn(3, 4, 34)
        action_history = torch.randn(3, 4, 15)
        future_actions = torch.randn(3, 4, 15)
        future_states = torch.randn(3, 4, 34)
        hand_actions = build_hand_actions(future_actions)
        consequences = build_consequences(
            future_actions,
            future_states,
        )
        forward_output = forward(
            state_history,
            action_history,
            hand_actions,
        )
        inverse_output = inverse(
            state_history,
            action_history,
            consequences,
        )
        self.assertEqual(forward_output.shape, (3, 4, 11))
        self.assertEqual(inverse_output.shape, (3, 4, 6))
        self.assertEqual(
            decode_forward_consequences(forward_output).shape,
            (3, 4, 11),
        )

    def test_future_object_action_is_not_a_hand_action(self):
        future_actions = torch.zeros(2, 3, 15)
        future_actions[..., :6] = 1.0
        future_actions[..., 6:] = 2.0
        hand_actions = build_hand_actions(future_actions)
        self.assertTrue(torch.equal(hand_actions, torch.ones(2, 3, 6)))


if __name__ == "__main__":
    unittest.main()

