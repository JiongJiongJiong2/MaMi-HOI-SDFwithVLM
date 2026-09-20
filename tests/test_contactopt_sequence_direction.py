import unittest

import numpy as np

from scripts.analyze_contactopt_sequence_direction import (
    aggregate_frames,
    direction_statistics,
    group_difference,
)


class ContactOptSequenceDirectionTest(unittest.TestCase):
    def test_normal_and_tangential_decomposition(self):
        objects = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        base = np.asarray(
            [
                [0.01, 0.0, 0.0],
                [1.01, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        candidate = np.asarray(
            [
                [0.015, 0.002, 0.0],
                [1.015, 0.003, 0.0],
            ],
            dtype=np.float64,
        )
        result = direction_statistics(
            base,
            candidate,
            objects,
            near_threshold_m=0.02,
        )
        self.assertAlmostEqual(
            result["near_signed_normal_mm"],
            5.0,
            places=6,
        )
        self.assertAlmostEqual(
            result["near_outward_normal_mm"],
            5.0,
            places=6,
        )
        self.assertAlmostEqual(
            result["near_inward_normal_mm"],
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            result["near_tangential_mm"],
            2.5,
            places=6,
        )
        self.assertEqual(result["near_vertex_count"], 2)

    def test_far_vertices_are_excluded(self):
        objects = np.zeros((1, 3), dtype=np.float64)
        base = np.asarray([[0.03, 0.0, 0.0]], dtype=np.float64)
        candidate = np.asarray([[0.04, 0.0, 0.0]], dtype=np.float64)
        result = direction_statistics(
            base,
            candidate,
            objects,
            near_threshold_m=0.02,
        )
        self.assertIsNone(result["near_signed_normal_mm"])
        self.assertEqual(result["near_vertex_count"], 0)

    def test_object_vertex_count_can_differ_from_hand_vertex_count(self):
        objects = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        base = np.asarray([[0.01, 0.0, 0.0]], dtype=np.float64)
        candidate = np.asarray([[0.02, 0.0, 0.0]], dtype=np.float64)
        result = direction_statistics(
            base,
            candidate,
            objects,
            near_threshold_m=0.03,
        )
        self.assertAlmostEqual(
            result["near_signed_normal_mm"],
            10.0,
            places=6,
        )

    def test_aggregate_frames_ignores_missing_metric_values(self):
        frames = [
            {"metric": 1.0},
            {"metric": 3.0},
            {"metric": None},
        ]
        self.assertEqual(
            aggregate_frames(frames, metrics=("metric",))["metric"],
            2.0,
        )

    def test_group_difference_resamples_sequences(self):
        first = [
            {"sequence": "a", "metric": 3.0},
            {"sequence": "a", "metric": 5.0},
            {"sequence": "b", "metric": 4.0},
        ]
        second = [
            {"sequence": "c", "metric": 1.0},
            {"sequence": "d", "metric": 2.0},
        ]
        result = group_difference(
            first,
            second,
            "metric",
            bootstrap_samples=100,
            seed=1,
        )
        self.assertAlmostEqual(result["mean"], 2.5)
        self.assertEqual(len(result["ci95"]), 2)


if __name__ == "__main__":
    unittest.main()
