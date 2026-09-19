import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_contactopt_sequence_temporal import (
    ratio,
    summarize_window,
    threshold_pass,
    trajectory_stat,
)


class ContactOptSequenceTemporalTest(unittest.TestCase):
    def test_trajectory_stat_uses_frame_differences(self):
        values = np.asarray(
            [
                [[0.0, 0.0, 0.0]],
                [[0.001, 0.0, 0.0]],
                [[0.002, 0.0, 0.0]],
                [[0.003, 0.0, 0.0]],
            ]
        )
        stats = trajectory_stat(values)
        self.assertAlmostEqual(stats["speed_mm"], 1.0)
        self.assertAlmostEqual(stats["acceleration_mm"], 0.0)
        self.assertAlmostEqual(stats["jerk_mm"], 0.0)

    def test_ratio_and_threshold_handle_zero_baseline(self):
        self.assertIsNone(ratio(1.0, 0.0))
        self.assertTrue(threshold_pass(None, 2.0))
        self.assertTrue(threshold_pass(2.0, 2.0))
        self.assertFalse(threshold_pass(2.1, 2.0))

    def test_summarize_window_tracks_contact_and_temporal_gates(self):
        geometry = {
            "frames": np.asarray([0, 1, 2, 3]),
            "input_hand_vertices": np.asarray(
                [
                    [[0.0, 0.0, 0.0]],
                    [[0.001, 0.0, 0.0]],
                    [[0.002, 0.0, 0.0]],
                    [[0.003, 0.0, 0.0]],
                ]
            ),
            "refined_hand_vertices": np.asarray(
                [
                    [[0.0, 0.0, 0.0]],
                    [[0.001, 0.0, 0.0]],
                    [[0.002, 0.0, 0.0]],
                    [[0.003, 0.0, 0.0]],
                ]
            ),
        }
        case = {
            "eligible_frames": [0, 1, 2, 3],
            "rows": [
                {
                    "hand_contact_relative_change": 0.2,
                    "distance_relative_change": -0.1,
                    "wrist_root_drift_m": 0.0,
                    "object_vertex_drift_m": 0.0,
                },
                {
                    "hand_contact_relative_change": 0.1,
                    "distance_relative_change": -0.2,
                    "wrist_root_drift_m": 0.0,
                    "object_vertex_drift_m": 0.0,
                },
                {
                    "hand_contact_relative_change": 0.1,
                    "distance_relative_change": -0.1,
                    "wrist_root_drift_m": 0.0,
                    "object_vertex_drift_m": 0.0,
                },
                {
                    "hand_contact_relative_change": 0.1,
                    "distance_relative_change": -0.1,
                    "wrist_root_drift_m": 0.0,
                    "object_vertex_drift_m": 0.0,
                },
            ],
        }
        row = {
            "chunk_id": "chunk",
            "sequence": "seq",
            "split": "dev",
            "object_name": "object",
        }
        result = summarize_window(
            geometry,
            case,
            row,
            [0, 1, 2, 3],
            {
                "max_speed_ratio": 2.0,
                "max_acceleration_ratio": 2.0,
                "max_jerk_ratio": 3.0,
                "max_wrist_drift_m": 0.00001,
                "max_object_drift_m": 0.000001,
                "max_distance_relative_change": 0.1,
            },
        )
        self.assertTrue(result["combined_gate"]["passed"])
        self.assertEqual(result["contact_improved_frames"], 4)
        self.assertEqual(result["distance_improved_frames"], 4)


if __name__ == "__main__":
    unittest.main()
