import unittest

import numpy as np

from scripts.evaluate_oakink_cross_intent_specificity import (
    specificity_summary,
)


class OakInkCrossIntentSpecificityTest(unittest.TestCase):
    def test_specificity_summary(self):
        labels = np.asarray([1, 1, 0, 0])
        scores = np.asarray([0.9, 0.8, 0.2, 0.1])
        result = specificity_summary(
            "test",
            labels,
            scores,
            np.asarray([0.1, 0.2]),
            0.5,
        )
        self.assertEqual(result["handover_positive_recall"], 1.0)
        self.assertEqual(result["cross_intent_false_positive_rate"], 0.0)
        self.assertEqual(result["combined"]["positive_count"], 2)


if __name__ == "__main__":
    unittest.main()
