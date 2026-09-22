import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from optimize_contactopt_direction_projection import (
    clip_pose_delta,
    damped_projection_step,
    normal_displacement_components,
    optimization_weights,
    temporal_rollback_reasons,
)


class ContactOptDirectionProjectionTest(unittest.TestCase):
    def test_normal_displacement_signs_and_tangent(self):
        objects = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        base = np.asarray(
            [[0.01, 0.0, 0.0], [1.01, 0.0, 0.0]],
            dtype=np.float64,
        )
        candidate = np.asarray(
            [
                [0.015, 0.002, 0.0],
                [1.005, -0.003, 0.0],
            ],
            dtype=np.float64,
        )
        result = normal_displacement_components(
            base,
            candidate,
            objects,
            near_threshold_m=0.02,
        )
        self.assertAlmostEqual(
            float(result["outward_m"][0] * 1000.0),
            5.0,
        )
        self.assertAlmostEqual(
            float(result["outward_m"][1] * 1000.0),
            -5.0,
        )
        self.assertEqual(
            result["selected"].tolist(),
            [True, True],
        )

    def test_tangent_and_zero_displacement_are_not_outward(self):
        objects = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
        base = np.asarray([[0.01, 0.0, 0.0]], dtype=np.float64)
        candidate = np.asarray(
            [[0.01, 0.003, 0.0], [0.01, 0.0, 0.0]],
            dtype=np.float64,
        )
        result = normal_displacement_components(
            np.repeat(base, 2, axis=0),
            candidate,
            objects,
            near_threshold_m=0.02,
        )
        self.assertAlmostEqual(
            float(result["outward_m"][0]),
            0.0,
        )
        self.assertAlmostEqual(
            float(result["outward_m"][1]),
            0.0,
        )

    def test_damped_projection_reduces_residual_and_respects_step_cap(self):
        jacobian = np.asarray(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        residual = np.asarray([-0.002, -0.001], dtype=np.float64)
        step = damped_projection_step(
            jacobian,
            residual,
            damping=0.0,
            max_step=1.0,
        )
        self.assertLess(
            np.linalg.norm(jacobian @ step - residual),
            1e-12,
        )
        capped = damped_projection_step(
            jacobian,
            residual,
            damping=0.0,
            max_step=0.001,
        )
        self.assertLessEqual(np.linalg.norm(capped), 0.001 + 1e-12)

    def test_pose_cap_and_rollback_reasons(self):
        delta = np.asarray(
            [[3.0, 4.0, 0.0], [0.1, 0.0, 0.0]],
            dtype=np.float64,
        )
        clipped = clip_pose_delta(delta, max_norm=1.0)
        self.assertAlmostEqual(np.linalg.norm(clipped[0]), 1.0)
        self.assertAlmostEqual(np.linalg.norm(clipped[1]), 0.1)
        self.assertEqual(
            temporal_rollback_reasons(True, False, 0.5, 1.0),
            ["temporal_gate"],
        )
        self.assertEqual(
            temporal_rollback_reasons(True, True, 1.5, 1.0),
            ["pose_budget"],
        )

    def test_arm_weights_keep_controls_on_same_temporal_budget(self):
        args = type(
            "Args",
            (),
            {
                "anchor_weight": 0.2,
                "pose_anchor_weight": 0.05,
                "acceleration_weight": 2.0,
                "jerk_weight": 2.0,
                "tail_weight": 8.0,
                "contact_weight": 10.0,
                "normal_weight": 20.0,
                "strong_contact_weight": 100.0,
            },
        )()
        direction = optimization_weights(
            "direction_constrained",
            args,
        )
        strong = optimization_weights("strong_contact", args)
        self.assertEqual(direction["acceleration_weight"], 2.0)
        self.assertEqual(direction["normal_weight"], 20.0)
        self.assertEqual(strong["acceleration_weight"], 2.0)
        self.assertEqual(strong["contact_weight"], 100.0)


if __name__ == "__main__":
    unittest.main()
