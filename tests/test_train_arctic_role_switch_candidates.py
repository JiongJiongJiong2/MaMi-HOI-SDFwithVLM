import unittest

import numpy as np

from scripts.train_arctic_role_switch_candidates import (
    average_precision,
    best_f1_threshold,
    binary_f1,
    merge_split_scores,
)


class TrainArcticRoleSwitchCandidatesTest(unittest.TestCase):
    def test_metric_helpers(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(average_precision(labels, scores), 1.0)
        threshold = best_f1_threshold(labels, scores)
        self.assertEqual(threshold, 0.8)
        self.assertEqual(binary_f1(labels, scores >= threshold), 1.0)

    def test_merge_split_scores(self):
        merged = merge_split_scores(
            np.asarray(["train", "val", "train"]),
            np.asarray([0.1, 0.2]),
            np.asarray([0.9]),
        )
        np.testing.assert_allclose(merged, [0.1, 0.9, 0.2])

    def test_merge_split_scores_with_test(self):
        merged = merge_split_scores(
            np.asarray(["train", "val", "test"]),
            np.asarray([0.1]),
            np.asarray([0.2]),
            np.asarray([0.3]),
        )
        np.testing.assert_allclose(merged, [0.1, 0.2, 0.3])


if __name__ == "__main__":
    unittest.main()
