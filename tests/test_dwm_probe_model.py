import importlib.util
import unittest

import torch

from manip.world_model.dwm.probe_model import DWMProbeRankingModel


HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "torch is not installed")
class DWMProbeRankingModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9)
        self.model = DWMProbeRankingModel(
            hidden_size=32,
            response_dim=8,
        )
        self.initial = torch.randn(2, 168)
        self.probe_action = torch.randn(2, 4, 51)
        self.probe_state = torch.randn(2, 5, 168)
        self.probe_mask = torch.tensor([
            [True, False, False, False],
            [True, True, False, False],
        ])
        self.post_state = torch.randn(2, 168)
        self.candidate = torch.randn(2, 5, 8, 51)
        self.target = torch.randn(2, 3)

    def test_output_shapes(self):
        output = self.model(
            self.initial,
            self.probe_action,
            self.probe_state,
            self.probe_mask,
            self.post_state,
            self.candidate,
            self.target,
        )
        self.assertEqual(output["candidate_score"].shape, (2, 5))
        self.assertEqual(output["predicted_object_delta"].shape, (2, 5, 9))
        self.assertEqual(output["response_mu"].shape, (2, 8))
        self.assertEqual(output["response_logvar"].shape, (2, 8))

    def test_padding_does_not_change_response_context(self):
        first_action = self.probe_action.clone()
        first_state = self.probe_state.clone()
        second_action = first_action.clone()
        second_state = first_state.clone()
        for row in range(self.probe_mask.shape[0]):
            invalid = ~self.probe_mask[row]
            second_action[row, invalid] = torch.randn_like(
                second_action[row, invalid]
            )
            second_state[row, 1:][invalid] = torch.randn_like(
                second_state[row, 1:][invalid]
            )
        first = self.model(
            self.initial,
            first_action,
            first_state,
            self.probe_mask,
            self.post_state,
            self.candidate,
            self.target,
        )
        second = self.model(
            self.initial,
            second_action,
            second_state,
            self.probe_mask,
            self.post_state,
            self.candidate,
            self.target,
        )
        torch.testing.assert_close(
            first["response_mu"],
            second["response_mu"],
        )

    def test_candidate_scores_are_permutation_equivariant(self):
        permutation = torch.tensor([3, 0, 4, 1, 2])
        first = self.model(
            self.initial,
            self.probe_action,
            self.probe_state,
            self.probe_mask,
            self.post_state,
            self.candidate,
            self.target,
        )
        second = self.model(
            self.initial,
            self.probe_action,
            self.probe_state,
            self.probe_mask,
            self.post_state,
            self.candidate[:, permutation],
            self.target,
        )
        torch.testing.assert_close(
            second["candidate_score"],
            first["candidate_score"][:, permutation],
        )

    def test_geometry_residual_starts_at_geometry_baseline(self):
        model = DWMProbeRankingModel(
            hidden_size=32,
            response_dim=8,
            use_geometry_baseline=True,
        )
        geometry = torch.randn(2, 5)
        output = model(
            self.initial,
            self.probe_action,
            self.probe_state,
            self.probe_mask,
            self.post_state,
            self.candidate,
            self.target,
            geometry_logit=geometry,
            geometry_features=torch.zeros(2, 5, 7),
        )
        torch.testing.assert_close(
            output["candidate_score"],
            geometry,
        )


if __name__ == "__main__":
    unittest.main()
