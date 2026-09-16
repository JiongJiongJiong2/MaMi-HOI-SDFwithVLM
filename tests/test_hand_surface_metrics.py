import unittest

import torch

from t2m_eval.hand_surface_metrics import (
    HAND_CONTACT_METRIC_VERSION,
    HAND_SURFACE_METRIC_VERSION,
    _frame_min_distances,
    aggregate_hand_contact_metrics_v3,
    aggregate_hand_surface_metrics_v2,
    compute_hand_contact_metrics_v3,
    compute_gt_proxy_contacts,
    compute_hand_surface_metrics_v2,
)
from manip.model.sdf_utils import point_to_triangle_unsigned_distance


def _build_inputs():
    object_verts = torch.zeros(2, 3, 3)
    object_verts[:, 0] = torch.tensor([0.0, 0.0, 0.0])
    object_verts[:, 1] = torch.tensor([1.0, 0.0, 0.0])
    object_verts[:, 2] = torch.tensor([0.0, 1.0, 0.0])

    gt_verts = torch.zeros(2, 2, 3)
    gt_verts[0, :, 2] = torch.tensor([0.04, 0.06])
    gt_verts[1, :, 2] = torch.tensor([0.03, 0.07])

    pred_verts = torch.zeros_like(gt_verts)
    pred_verts[0, :, 2] = torch.tensor([0.03, 0.05])
    pred_verts[1, :, 2] = torch.tensor([0.02, 0.04])

    gt_jpos = torch.zeros(2, 24, 3)
    gt_jpos[0, 22, 2] = 0.01
    gt_jpos[0, 23, 2] = 0.02
    gt_jpos[1, 22, 2] = 0.08
    gt_jpos[1, 23, 2] = 0.03

    pred_jpos = torch.zeros_like(gt_jpos)
    pred_jpos[0, 22, 2] = 0.02
    pred_jpos[0, 23, 2] = 0.03
    pred_jpos[1, 22, 2] = 0.09
    pred_jpos[1, 23, 2] = 0.04

    return {
        "gt_human_verts": gt_verts,
        "pred_human_verts": pred_verts,
        "gt_human_jpos": gt_jpos,
        "pred_human_jpos": pred_jpos,
        "gt_obj_verts": object_verts,
        "pred_obj_verts": object_verts.clone(),
        "object_faces": torch.tensor([[0, 1, 2]]),
    }


class HandSurfaceMetricTests(unittest.TestCase):
    def test_batched_frame_distances_match_scalar_reference(self):
        torch.manual_seed(7)
        points = torch.randn(3, 5, 3)
        object_verts = torch.randn(3, 8, 3)
        object_faces = torch.tensor(
            [
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
                [5, 6, 7],
            ]
        )
        frame_mask = torch.tensor([True, False, True])

        expected = []
        for frame in frame_mask.nonzero().flatten().tolist():
            distances, _ = point_to_triangle_unsigned_distance(
                points[frame],
                object_verts[frame][object_faces],
                point_chunk=2,
                triangle_chunk=3,
            )
            expected.append(distances.min())
        expected = torch.stack(expected)

        actual = _frame_min_distances(
            points,
            object_verts,
            object_faces,
            frame_mask.nonzero().flatten(),
            frame_chunk=1,
            point_chunk=2,
            triangle_chunk=3,
        )
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-6, atol=1e-7))

    def test_contact_labels_match_proxy_threshold(self):
        inputs = _build_inputs()
        left, right = compute_gt_proxy_contacts(
            inputs["gt_human_jpos"],
            inputs["gt_obj_verts"],
        )
        self.assertTrue(torch.equal(left, torch.tensor([True, False])))
        self.assertTrue(torch.equal(right, torch.tensor([True, True])))

    def test_proxy_and_mesh_clearances_are_per_hand(self):
        inputs = _build_inputs()
        left, right = compute_gt_proxy_contacts(
            inputs["gt_human_jpos"],
            inputs["gt_obj_verts"],
        )
        result = compute_hand_surface_metrics_v2(
            **inputs,
            left_hand_vertex_idxs=[0],
            right_hand_vertex_idxs=[1],
            gt_left_contact=left,
            gt_right_contact=right,
        )

        self.assertEqual(result["version"], HAND_SURFACE_METRIC_VERSION)
        self.assertAlmostEqual(result["gt_proxy_left_mm"], 10.0, places=4)
        self.assertAlmostEqual(result["pred_proxy_left_mm"], 20.0, places=4)
        self.assertAlmostEqual(result["gt_proxy_right_mm"], 25.0, places=4)
        self.assertAlmostEqual(result["pred_proxy_right_mm"], 35.0, places=4)
        self.assertAlmostEqual(result["gt_hand_mesh_left_mm"], 40.0, places=4)
        self.assertAlmostEqual(result["pred_hand_mesh_left_mm"], 30.0, places=4)
        self.assertAlmostEqual(result["gt_hand_mesh_right_mm"], 65.0, places=4)
        self.assertAlmostEqual(result["pred_hand_mesh_right_mm"], 45.0, places=4)
        self.assertEqual(result["left_contact_frame_count"], 1)
        self.assertEqual(result["right_contact_frame_count"], 2)

    def test_empty_contact_is_null_not_zero(self):
        inputs = _build_inputs()
        result = compute_hand_surface_metrics_v2(
            **inputs,
            left_hand_vertex_idxs=[0],
            right_hand_vertex_idxs=[1],
            gt_left_contact=torch.tensor([False, False]),
            gt_right_contact=torch.tensor([False, False]),
        )
        self.assertIsNone(result["gt_proxy_left_mm"])
        self.assertIsNone(result["pred_hand_mesh_right_mm"])
        self.assertEqual(result["left_contact_frame_count"], 0)
        self.assertEqual(result["right_contact_frame_count"], 0)

    def test_aggregate_preserves_zero_and_ignores_null(self):
        records = [
            {
                "gt_proxy_left_mm": 0.0,
                "pred_proxy_left_mm": 2.0,
                "gt_proxy_right_mm": None,
                "pred_proxy_right_mm": 4.0,
                "gt_hand_mesh_left_mm": 0.0,
                "pred_hand_mesh_left_mm": 6.0,
                "gt_hand_mesh_right_mm": None,
                "pred_hand_mesh_right_mm": 8.0,
                "left_contact_frame_count": 2,
                "right_contact_frame_count": 0,
            },
            {
                "gt_proxy_left_mm": 10.0,
                "pred_proxy_left_mm": 12.0,
                "gt_proxy_right_mm": 14.0,
                "pred_proxy_right_mm": 16.0,
                "gt_hand_mesh_left_mm": 20.0,
                "pred_hand_mesh_left_mm": 22.0,
                "gt_hand_mesh_right_mm": 24.0,
                "pred_hand_mesh_right_mm": 26.0,
                "left_contact_frame_count": 3,
                "right_contact_frame_count": 4,
            },
        ]
        result = aggregate_hand_surface_metrics_v2(records)
        self.assertEqual(result["gt_proxy_left_mm"], 5.0)
        self.assertEqual(result["gt_proxy_left_mm_valid_sequence_count"], 2)
        self.assertEqual(result["gt_proxy_right_mm"], 14.0)
        self.assertEqual(result["gt_proxy_right_mm_valid_sequence_count"], 1)
        self.assertEqual(result["left_contact_frame_count"], 5)
        self.assertEqual(result["right_contact_frame_count"], 4)

    def test_v3_uses_saved_mask_and_reports_micro_counts(self):
        inputs = _build_inputs()
        result = compute_hand_contact_metrics_v3(
            **inputs,
            left_hand_vertex_idxs=[0],
            right_hand_vertex_idxs=[1],
            saved_left_contact=torch.tensor([True, False]),
            saved_right_contact=torch.tensor([False, True]),
        )

        self.assertEqual(result["version"], HAND_CONTACT_METRIC_VERSION)
        self.assertEqual(result["left_contact_frame_count"], 1)
        self.assertEqual(result["right_contact_frame_count"], 1)
        self.assertEqual(result["left_tp"], 1)
        self.assertEqual(result["left_tn"], 1)
        self.assertEqual(result["right_tp"], 1)
        self.assertEqual(result["right_fp"], 1)
        self.assertAlmostEqual(result["right_precision"], 0.5, places=6)
        self.assertAlmostEqual(result["right_recall"], 1.0, places=6)
        self.assertAlmostEqual(result["gt_proxy_left_mm"], 10.0, places=4)
        self.assertAlmostEqual(result["pred_proxy_left_mm"], 20.0, places=4)
        self.assertAlmostEqual(result["gt_hand_mesh_left_mm"], 40.0, places=4)
        self.assertAlmostEqual(result["pred_hand_mesh_left_mm"], 30.0, places=4)

    def test_v3_empty_gt_contact_uses_null_clearance_and_empty_denominators(self):
        inputs = _build_inputs()
        result = compute_hand_contact_metrics_v3(
            **inputs,
            left_hand_vertex_idxs=[0],
            right_hand_vertex_idxs=[1],
            saved_left_contact=torch.tensor([False, False]),
            saved_right_contact=torch.tensor([False, False]),
        )

        self.assertIsNone(result["gt_proxy_left_mm"])
        self.assertIsNone(result["pred_hand_mesh_left_mm"])
        self.assertEqual(result["left_tp"], 0)
        self.assertEqual(result["left_fp"], 1)
        self.assertEqual(result["left_precision"], 0.0)
        self.assertIsNone(result["left_recall"])
        self.assertEqual(result["left_f1"], 0.0)

    def test_v3_aggregate_separates_micro_and_macro(self):
        records = [
            {
                "sequence": "seq_a",
                "left_tp": 1,
                "left_fp": 0,
                "left_tn": 1,
                "left_fn": 0,
                "left_precision": 1.0,
                "left_recall": 1.0,
                "left_f1": 1.0,
                "gt_proxy_left_mm": 10.0,
                "pred_proxy_left_mm": 12.0,
                "gt_hand_mesh_left_mm": 14.0,
                "pred_hand_mesh_left_mm": 16.0,
                "right_tp": 0,
                "right_fp": 0,
                "right_tn": 1,
                "right_fn": 0,
                "right_precision": None,
                "right_recall": None,
                "right_f1": None,
            },
            {
                "sequence": "seq_b",
                "left_tp": 0,
                "left_fp": 1,
                "left_tn": 0,
                "left_fn": 0,
                "left_precision": 0.0,
                "left_recall": None,
                "left_f1": 0.0,
                "gt_proxy_left_mm": 20.0,
                "pred_proxy_left_mm": 22.0,
                "gt_hand_mesh_left_mm": 24.0,
                "pred_hand_mesh_left_mm": 26.0,
                "right_tp": 0,
                "right_fp": 0,
                "right_tn": 1,
                "right_fn": 0,
                "right_precision": None,
                "right_recall": None,
                "right_f1": None,
            },
        ]
        result = aggregate_hand_contact_metrics_v3(records)

        self.assertEqual(result["micro"]["left"]["tp"], 1)
        self.assertEqual(result["micro"]["left"]["fp"], 1)
        self.assertAlmostEqual(result["micro"]["left"]["precision"], 0.5)
        self.assertAlmostEqual(result["micro"]["left"]["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(result["macro"]["left"]["precision"], 0.5)
        self.assertAlmostEqual(result["macro"]["left"]["f1"], 0.5)
        self.assertIsNone(result["macro"]["right"]["f1"])
        self.assertAlmostEqual(
            result["clearance_macro_mm"]["left"]["gt_proxy_left_mm"],
            15.0,
        )


if __name__ == "__main__":
    unittest.main()
