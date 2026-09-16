"""Focused tests for the optional HOI-Dyn response module."""

import copy
import unittest

import torch

from manip.world_model.hoidyn.adapter import reduce_hoidyn_regulate_loss
from manip.world_model.hoidyn.model import Dynamics


class _FakeDataset:
    @staticmethod
    def de_normalize_obj_pos_min_max(value):
        return value


def _config():
    return {
        "history_win": 1,
        "future_win": 1,
        "feat_dim": 32,
        "head": 4,
        "depth": 1,
        "rot_loss_method": "theta",
    }


class HoidynIntegrationTests(unittest.TestCase):
    def test_reduce_regulate_loss(self):
        regulate = {
            "pc_loss": torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
        }
        per_sample, stats = reduce_hoidyn_regulate_loss(regulate)
        self.assertEqual(tuple(per_sample.shape), (2,))
        self.assertTrue(torch.isfinite(per_sample).all())
        self.assertEqual(stats["loss_type"], "pc")
        self.assertTrue(torch.isfinite(torch.tensor(stats["total"])))

    def test_frozen_dynamics_forward_and_backward(self):
        torch.manual_seed(1)
        model = Dynamics(_config())
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        batch_size = 2
        seq_len = 8
        gt_x0 = torch.randn(batch_size, seq_len, 220)
        pred_x0 = (gt_x0 + 0.01 * torch.randn_like(gt_x0)).requires_grad_(True)
        data_dict = {
            "input_obj_bps": torch.randn(batch_size, 1024, 3),
            "reference_obj_rot_mat": torch.eye(3).reshape(1, 1, 3, 3).repeat(
                batch_size, 1, 1, 1
            ),
            "rest_pose_obj_pts": torch.randn(batch_size, 100, 3),
        }
        padding_mask = torch.ones(batch_size, 1, seq_len)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }

        output = model.compute_dynamics_loss_for_generated_hoi(
            {
                "gt_x0": gt_x0,
                "pred_x0": pred_x0,
                "padding_mask": padding_mask,
            },
            max_step=2,
            data_dict=copy.deepcopy(data_dict),
            dataset=_FakeDataset(),
            loss_type="pc",
            dyn_loss_type="res",
            contact_source="gt",
        )
        regulate = output["regulate_dynamics"]["pc_loss"]
        self.assertTrue(torch.isfinite(regulate).all())

        regulate.mean().backward()
        self.assertIsNotNone(pred_x0.grad)
        self.assertTrue(torch.isfinite(pred_x0.grad).all())
        for name, parameter in model.named_parameters():
            self.assertIsNone(parameter.grad)
            self.assertTrue(torch.equal(before[name], parameter.detach()))


if __name__ == "__main__":
    unittest.main()
