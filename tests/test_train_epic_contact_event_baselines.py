import unittest

import numpy as np

from scripts.train_epic_contact_event_baselines import (
    average_precision,
    best_f1_threshold,
    binary_f1,
    build_samples,
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

    def test_contact_threshold_parameter_changes_labels(self):
        frames = [
            {
                "split": "train",
                "video_id": "P01_video",
                "frame": frame,
                "left": {
                    "valid": True,
                    "clip_id": "clip",
                    "min_distance_m": 0.0012,
                    "object_name": "cup",
                },
                "right": {
                    "valid": False,
                    "clip_id": None,
                    "min_distance_m": None,
                    "object_name": None,
                },
            }
            for frame in range(5)
        ]
        strict = build_samples(frames, ["cup"], 0.001)
        relaxed = build_samples(frames, ["cup"], 0.0015)
        self.assertEqual(strict[0]["current_contact"], 0)
        self.assertEqual(relaxed[0]["current_contact"], 1)


if __name__ == "__main__":
    unittest.main()
