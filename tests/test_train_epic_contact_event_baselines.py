import unittest

import numpy as np

from scripts.train_epic_contact_event_baselines import (
    average_precision,
    best_f1_threshold,
    binary_f1,
    roc_auc,
)


class EpicContactEventBaselineTest(unittest.TestCase):
    def test_auc_and_ap(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(roc_auc(labels, scores), 1.0)
        self.assertEqual(average_precision(labels, scores), 1.0)

    def test_f1_threshold(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.4, 0.6, 0.9])
        threshold = best_f1_threshold(labels, scores)
        self.assertEqual(threshold, 0.6)
        self.assertEqual(binary_f1(labels, scores >= threshold), 1.0)

    def test_constant_scores_return_prevalence_ap(self):
        labels = np.asarray([0, 0, 0, 1])
        scores = np.zeros(4)
        self.assertAlmostEqual(average_precision(labels, scores), 0.25)
        self.assertEqual(roc_auc(labels, scores), 0.5)


if __name__ == "__main__":
    unittest.main()
