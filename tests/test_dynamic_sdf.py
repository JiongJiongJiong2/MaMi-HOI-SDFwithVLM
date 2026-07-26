import math
import inspect
import unittest
from pathlib import Path

import torch

from manip.model.sdf_utils import (
    build_dynamic_sdf_prediction_query,
    object_sdf_in_bounds_mask,
    object_to_world_points,
    project_to_so3,
    rotation_validity_statistics,
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
        self.assertAlmostEqual(distances[0, 0, 0].item(), -0.35, places=5)
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

    def test_world_object_round_trip(self):
        rotation = torch.tensor([[
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        ]])
        translation = torch.tensor([[[1.0, 2.0, 3.0], [-0.2, 0.4, 1.3]]])
        points_object = torch.tensor([[
            [[1.0, 0.0, 0.0], [0.2, -0.4, 0.6]],
            [[-0.3, 0.5, 0.9], [0.0, 0.0, 0.0]],
        ]])
        points_world = object_to_world_points(
            points_object, rotation, translation
        )
        recovered = world_to_object_points(
            points_world, rotation, translation
        )
        self.assertTrue(torch.allclose(recovered, points_object, atol=1e-6))

    def test_world_to_object_is_rigid_invariant(self):
        object_rotation = torch.eye(3).reshape(1, 1, 3, 3)
        object_com = torch.tensor([[[0.3, -0.2, 0.7]]])
        point_world = torch.tensor([[[[0.8, 0.1, 1.0]]]])
        canonical = world_to_object_points(
            point_world, object_rotation, object_com
        )
        global_rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        global_translation = torch.tensor([1.2, -0.8, 0.4])
        transformed_point = (
            torch.matmul(point_world, global_rotation.T)
            + global_translation
        )
        transformed_com = (
            torch.matmul(object_com, global_rotation.T)
            + global_translation
        )
        transformed_object_rotation = torch.matmul(
            global_rotation.reshape(1, 1, 3, 3),
            object_rotation,
        )
        transformed_canonical = world_to_object_points(
            transformed_point,
            transformed_object_rotation,
            transformed_com,
        )
        self.assertTrue(
            torch.allclose(transformed_canonical, canonical, atol=1e-6)
        )

    def test_so3_projection_is_proper_and_differentiable(self):
        raw = torch.tensor(
            [[
                [1.0, 0.12, -0.04],
                [-0.08, 0.95, 0.15],
                [0.02, -0.10, -0.90],
            ]],
            requires_grad=True,
        )
        projected = project_to_so3(raw)
        orthogonality_error, determinant = rotation_validity_statistics(
            projected
        )
        self.assertLess(orthogonality_error.max().item(), 1e-5)
        self.assertTrue(
            torch.allclose(determinant, torch.ones_like(determinant), atol=1e-5)
        )
        projected.square().sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_so3_projection_rejects_collapsed_rows(self):
        with self.assertRaises(ValueError):
            project_to_so3(torch.zeros(1, 3, 3))

    def test_oob_mask_and_sampler_mask_agree(self):
        points = torch.tensor([[
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 1.0],
            [1.001, 0.0, 0.0],
        ]])
        expected = torch.tensor([[True, True, False]])
        direct = object_sdf_in_bounds_mask(
            points, self.centroid, self.extents
        )
        _, sampled = sample_object_sdf_at_points(
            self.grid,
            points,
            self.centroid,
            self.extents,
            return_valid_mask=True,
        )
        self.assertTrue(torch.equal(direct, expected))
        self.assertTrue(torch.equal(sampled, expected))

    def test_invalid_query_does_not_contribute_to_loss(self):
        values = torch.tensor([[[0.2], [-10.0]]])
        contact = torch.ones_like(values)
        valid = torch.tensor([[[1.0], [0.0]]])
        penetration, attraction = sdf_contact_losses(
            values, contact, valid
        )
        self.assertEqual(penetration.item(), 0.0)
        self.assertAlmostEqual(attraction.item(), 0.2, places=6)

    def test_prediction_query_changes_with_predicted_geometry(self):
        palm = torch.tensor([[[[0.8, 0.1, 0.0]]]])
        rotation = torch.eye(3).reshape(1, 1, 3, 3)
        com = torch.zeros(1, 1, 3)
        base, _ = build_dynamic_sdf_prediction_query(
            palm, rotation, com
        )
        shifted_palm, _ = build_dynamic_sdf_prediction_query(
            palm + torch.tensor([0.2, 0.0, 0.0]), rotation, com
        )
        shifted_object, _ = build_dynamic_sdf_prediction_query(
            palm, rotation, com + torch.tensor([[[0.2, 0.0, 0.0]]])
        )
        rotated, _ = build_dynamic_sdf_prediction_query(
            palm,
            torch.tensor([[[[
                0.0, -1.0, 0.0,
            ], [
                1.0, 0.0, 0.0,
            ], [
                0.0, 0.0, 1.0,
            ]]]]),
            com,
        )
        self.assertFalse(torch.allclose(base, shifted_palm))
        self.assertFalse(torch.allclose(base, shifted_object))
        self.assertFalse(torch.allclose(base, rotated))

    def test_prediction_query_has_no_gt_future_input(self):
        parameter_names = tuple(
            inspect.signature(
                build_dynamic_sdf_prediction_query
            ).parameters
        )
        self.assertEqual(
            parameter_names,
            (
                'predicted_palm_world',
                'predicted_object_rotation',
                'predicted_object_com',
            ),
        )
        palm = torch.tensor([[[[0.3, -0.1, 0.2]]]])
        rotation = torch.eye(3).reshape(1, 1, 3, 3)
        com = torch.zeros(1, 1, 3)
        first, _ = build_dynamic_sdf_prediction_query(palm, rotation, com)
        # These represent two incompatible GT futures. Neither is accepted by
        # the U1 prediction-query API, so changing them cannot change output.
        gt_future_a = torch.zeros(1, 10, 24, 3)
        gt_future_b = torch.randn(1, 10, 24, 3)
        self.assertFalse(torch.equal(gt_future_a, gt_future_b))
        second, _ = build_dynamic_sdf_prediction_query(palm, rotation, com)
        self.assertTrue(torch.equal(first, second))

    def test_u1_entrypoints_do_not_enable_gt_query_or_u6(self):
        repository_root = Path(__file__).resolve().parents[1]
        train_script = (
            repository_root / 'scripts' / 'train_dynamic_sdf.sh'
        ).read_text(encoding='utf-8')
        evaluation_script = (
            repository_root / 'scripts' / 'evaluate_u0_u1.sh'
        ).read_text(encoding='utf-8')
        for source in (train_script, evaluation_script):
            self.assertNotIn('--use_local_sdf', source)
            self.assertNotIn('--use_sdf_contrastive', source)
        self.assertIn('--use_dynamic_sdf', train_script)
        self.assertIn(
            'val_hand_query_points = None',
            (
                repository_root
                / 'train'
                / 'trainer_control_GAPA_chois.py'
            ).read_text(encoding='utf-8'),
        )

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
