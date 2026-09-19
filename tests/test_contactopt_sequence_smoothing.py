import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_contactopt_sequence_smoothing import (
    arm_metrics,
    contact_gate,
)


class ContactOptSequenceSmoothingTest(unittest.TestCase):
    def test_arm_metrics_passes_identical_trajectory(self):
        values = np.asarray(
            [
                [[0.0, 0.0, 0.0]],
                [[0.001, 0.0, 0.0]],
                [[0.002, 0.0, 0.0]],
                [[0.003, 0.0, 0.0]],
            ]
        )
        result = arm_metrics(
            values,
            values,
            {
                "max_speed_ratio": 2.0,
                "max_acceleration_ratio": 2.0,
                "max_jerk_ratio": 3.0,
            },
        )
        self.assertTrue(result["temporal_gate"]["passed"])

    def test_contact_gate_requires_frozen_coordinates(self):
        rows = [
            {
                "contact_relative_change": 0.1,
                "distance_relative_change": -0.1,
                "wrist_root_drift_m": 0.0,
                "object_vertex_drift_m": 0.0,
            }
            for _ in range(4)
        ]
        result = contact_gate(
            rows,
            {
                "max_wrist_drift_m": 0.00001,
                "max_object_drift_m": 0.000001,
                "max_distance_relative_change": 0.10,
            },
        )
        self.assertTrue(result["passed"])


if __name__ == "__main__":
    unittest.main()
