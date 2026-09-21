import unittest

import numpy as np

from scripts.analyze_arctic_motion_feature_stability import (
    balanced_group_folds,
    expected_calibration_error,
    feature_indices,
    select_features,
)


class ArcticMotionFeatureStabilityTest(unittest.TestCase):
    def test_feature_groups(self):
        self.assertEqual(feature_indices("distance"), list(range(10)))
        self.assertEqual(feature_indices("motion_only"), list(range(10, 16)))
        self.assertEqual(
            feature_indices("all_minus_hand_speed"),
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15],
        )

    def test_select_features(self):
        features = np.arange(2 * 30 * 16).reshape(2, 30, 16)
        selected = select_features(features, "motion_only")
        self.assertEqual(selected.shape, (2, 30 * 6))

    def test_balanced_group_folds_are_disjoint(self):
        groups = [f"g{index}" for index in range(8)]
        participants = np.asarray(
            [group for group in groups for _ in range(3)]
        )
        labels = np.asarray(
            [1, 0, 0] * len(groups)
        )
        folds = balanced_group_folds(participants, labels, 4, 7)
        for fit_indices, score_indices in folds:
            self.assertFalse(set(fit_indices) & set(score_indices))
            self.assertEqual(len(fit_indices) + len(score_indices), len(labels))

    def test_fold_assignment_changes_with_seed(self):
        groups = [f"g{index}" for index in range(8)]
        participants = np.asarray(
            [group for group in groups for _ in range(3)]
        )
        labels = np.asarray(
            [1, 0, 0] * len(groups)
        )
        first = balanced_group_folds(participants, labels, 4, 1)
        second = balanced_group_folds(participants, labels, 4, 2)
        first_partition = {
            tuple(sorted(participants[indices].tolist()))
            for _, indices in first
        }
        second_partition = {
            tuple(sorted(participants[indices].tolist()))
            for _, indices in second
        }
        self.assertNotEqual(first_partition, second_partition)

    def test_expected_calibration_error(self):
        labels = np.asarray([0, 0, 1, 1])
        probabilities = np.asarray([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(
            expected_calibration_error(labels, probabilities),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
