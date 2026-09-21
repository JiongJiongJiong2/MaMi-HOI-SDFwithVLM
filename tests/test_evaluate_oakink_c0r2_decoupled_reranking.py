import unittest

import numpy as np

try:
    from scripts.evaluate_oakink_c0r2_decoupled_reranking import (
        exact_mcnemar,
        joint_features,
        reference_valid,
        region_features,
    )
except ModuleNotFoundError:
    from evaluate_oakink_c0r2_decoupled_reranking import (  # noqa: E402
        exact_mcnemar,
        joint_features,
        reference_valid,
        region_features,
    )


class EvaluateOakInkC0R2DecoupledRerankingTest(unittest.TestCase):
    def test_reference_valid_requires_receiver_position(self):
        events = {
            "valid": np.asarray([[True] * 61, [True] * 61]),
        }
        events["valid"][1, 25] = False
        targets = {
            "target_offsets": np.asarray([0, 1, 2]),
        }
        positions = {0: 25, 1: 25}
        mask = reference_valid(events, targets, positions)
        np.testing.assert_array_equal(mask, [True, False])

    def test_region_features_clips_at_scale(self):
        region = np.asarray([[0.0, 0.0, 0.0]])
        candidate = np.asarray([[0.03, 0.0, 0.0]])
        result = region_features(candidate, region)
        self.assertEqual(result["region_clearance"], 1.0)

    def test_joint_features_counts_collision(self):
        joints = np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        candidate = np.asarray([
            [0.004, 0.0, 0.0],
            [0.020, 0.0, 0.0],
            [0.100, 0.0, 0.0],
        ])
        result = joint_features(candidate, joints)
        self.assertAlmostEqual(result["collision_fraction"], 2.0 / 3.0)
        self.assertGreater(result["joint_clearance"], 0.0)

    def test_exact_mcnemar_symmetric(self):
        self.assertAlmostEqual(exact_mcnemar(7, 1), 0.0703125)
        self.assertAlmostEqual(exact_mcnemar(1, 7), 0.0703125)


if __name__ == "__main__":
    unittest.main()
