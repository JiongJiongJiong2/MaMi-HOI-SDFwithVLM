import math
import unittest

import torch

from manip.model.sdf_utils import (
    sample_object_sdf_at_points,
    sdf_contact_losses,
    sdf_trajectory_ranking_loss,
    world_to_object_points,
)


def make_sphere_sdf(resolution=33, radius=0.35, extent=2.0):
    """Create a normalized SDF grid in [D(z), H(y), W(x)] order."""
    axis = torch.linspace(-extent / 2.0, extent / 2.0, resolution)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    world_sdf = torch.sqrt(xx.square() + yy.square() + zz.square()) - radius
    normalized_sdf = world_sdf / (extent / 2.0)
    return normalized_sdf[None, None]


class DynamicSDFTests(unittest.TestCase):
    def setUp(self):
        self.grid = make_sphere_sdf()
        self.centroid = torch.zeros(1, 3)
        self.extents = torch.full((1, 3), 2.0)

    def test_signed_distance_sign_and_scale(self):
        points = torch.tensor([[[0.0, 0.0, 0.0], [0.60, 0.0, 0.0]]])
        distances = sample_object_sdf_at_points(self.grid, points, self.centroid, self.extents)
        self.assertLess(distances[0, 0, 0].item(), -0.30)
        self.assertAlmostEqual(distances[0, 1, 0].item(), 0.25, places=2)

    def test_query_points_are_differentiable(self):
        points = torch.tensor([[[0.60, 0.0, 0.0]]], requires_grad=True)
        distances = sample_object_sdf_at_points(self.grid, points, self.centroid, self.extents)
        distances.sum().backward()
        self.assertTrue(torch.isfinite(points.grad).all())
        self.assertGreater(points.grad.abs().sum().item(), 0.0)

    def test_world_to_object_transform(self):
        # A +90-degree Z rotation maps local [1, 0, 0] to world [0, 1, 0].
        rotation = torch.tensor([[[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]])
        translation = torch.tensor([[[1.0, 2.0, 3.0]]])
        points_world = torch.tensor([[[[1.0, 3.0, 3.0]]]])
        points_object = world_to_object_points(points_world, rotation, translation)
        expected = torch.tensor([[[[1.0, 0.0, 0.0]]]])
        self.assertTrue(torch.allclose(points_object, expected, atol=1e-6))

    def test_contact_and_ranking_losses(self):
        valid = torch.ones(1, 3, 1)
        contact = torch.ones(1, 3, 2)
        gt = torch.zeros(1, 3, 2)
        pred_good = torch.full((1, 3, 2), 0.005)
        pred_bad = torch.full((1, 3, 2), -0.03)

        penetration, attraction = sdf_contact_losses(pred_good, contact, valid)
        self.assertEqual(penetration.item(), 0.0)
        self.assertGreater(attraction.item(), 0.0)

        loss_good = sdf_trajectory_ranking_loss(pred_good, gt, contact, valid)
        loss_bad = sdf_trajectory_ranking_loss(pred_bad, gt, contact, valid)
        self.assertLess(loss_good.item(), loss_bad.item())

    def test_empty_contact_mask_has_zero_ranking_loss(self):
        values = torch.zeros(1, 3, 2)
        zero_contact = torch.zeros(1, 3, 2)
        valid = torch.ones(1, 3, 1)
        loss = sdf_trajectory_ranking_loss(values, values, zero_contact, valid)
        self.assertTrue(math.isclose(loss.item(), 0.0, abs_tol=1e-8))


if __name__ == "__main__":
    unittest.main()
