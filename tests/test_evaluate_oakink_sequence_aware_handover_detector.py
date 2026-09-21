import unittest

import numpy as np

from scripts.evaluate_oakink_sequence_aware_handover_detector import (
    primary_predictions,
)


class OakInkSequenceAwareDetectorTest(unittest.TestCase):
    def test_primary_predictions(self):
        arrays = {
            "sequences": np.asarray(["s1", "s2", "s3"]),
            "splits": np.asarray(["val", "val", "val"]),
            "sources": np.asarray(["handover", "handover", "nonhandover"]),
            "lengths": np.asarray([5, 3, 4]),
            "scores": np.asarray([
                [0.1, 0.8, 0.9, 0.2, 0.0],
                [0.2, 0.3, 0.4, 0.0, 0.0],
                [0.9, 0.9, 0.9, 0.9, 0.0],
            ]),
        }
        predictions, emitted = primary_predictions(arrays, 0.5)
        self.assertEqual(predictions, {"s1": [3]})
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["frame"], 3)


if __name__ == "__main__":
    unittest.main()
