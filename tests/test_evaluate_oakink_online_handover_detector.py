import unittest
import tempfile

import numpy as np

from scripts.evaluate_oakink_online_handover_detector import (
    match_events,
    peak_frames,
    write_scored_predictions,
)


class OakInkOnlineHandoverDetectorTest(unittest.TestCase):
    def test_peak_frames_and_nms(self):
        scores = np.asarray([0, 0.8, 0.9, 0, 0.7, 0.6])
        peaks = peak_frames(scores, 0.5, nms_frames=1)
        self.assertEqual([frame for frame, _ in peaks], [3, 5])

    def test_match_events(self):
        truth = {"s1": [20], "s2": [30]}
        predictions = {"s1": [25, 200], "s2": [31]}
        result = match_events(truth, predictions, tolerance=15)
        self.assertEqual(result["true_positive"], 2)
        self.assertEqual(result["false_positive"], 1)
        self.assertAlmostEqual(result["precision"], 2 / 3)

    def test_write_scored_predictions(self):
        scored = {
            "sequences": ["s1", "s2"],
            "splits": np.asarray(["train", "val"]),
            "sources": np.asarray(["handover", "nonhandover"]),
            "scores": [
                np.asarray([0.1, 0.2]),
                np.asarray([0.3]),
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/scores.npz"
            write_scored_predictions(path, scored)
            arrays = np.load(path, allow_pickle=False)
            try:
                self.assertEqual(arrays["scores"].shape, (2, 2))
                np.testing.assert_array_equal(arrays["lengths"], [2, 1])
            finally:
                arrays.close()


if __name__ == "__main__":
    unittest.main()
