import unittest

import numpy as np

from scripts.train_epic_contact_event_boosting import (
    gate_result,
    train_boosting,
)


def evaluation(release_auprc=0.35, onset_auprc=0.70, next_auprc=0.93):
    return {
        "test": {
            "learned_logistic": {
                "next_contact": {"auprc": 0.93, "f1": 0.90},
                "onset": {"auprc": 0.70, "f1": 0.72},
                "release": {"auprc": 0.30, "f1": 0.38},
            },
            "boosting": {
                "next_contact": {"auprc": next_auprc, "f1": 0.90},
                "onset": {"auprc": onset_auprc, "f1": 0.72},
                "release": {"auprc": release_auprc, "f1": 0.38},
            },
        },
    }


def bootstrap(lower):
    return {
        "release": {
            "boosting_minus_learned_logistic": {
                "auprc": {"ci95": [lower, 0.10]},
            },
        },
    }


class EpicContactEventBoostingTest(unittest.TestCase):
    def test_gate_passes_on_improved_release(self):
        result = gate_result(evaluation(), bootstrap(0.01))
        self.assertTrue(result["overall"])

    def test_gate_rejects_nonpositive_release_interval(self):
        result = gate_result(evaluation(), bootstrap(0.0))
        self.assertFalse(result["overall"])
        self.assertFalse(result["release_auprc_ci_lower_positive"])

    def test_gate_rejects_large_onset_regression(self):
        result = gate_result(evaluation(onset_auprc=0.68), bootstrap(0.01))
        self.assertFalse(result["overall"])
        self.assertFalse(result["onset_auprc_noninferior"])

    def test_boosting_fits_separable_data(self):
        negative = np.column_stack((
            np.linspace(0.0, 0.2, 40),
            np.zeros(40),
        ))
        positive = np.column_stack((
            np.linspace(0.8, 1.0, 40),
            np.ones(40),
        ))
        train_x = np.vstack((negative, positive))
        train_y = np.asarray([0] * 40 + [1] * 40)
        model = train_boosting(train_x, train_y, 7)
        scores = model.predict_proba(train_x)[:, 1]
        self.assertGreater(scores[40:].mean(), scores[:40].mean())


if __name__ == "__main__":
    unittest.main()
