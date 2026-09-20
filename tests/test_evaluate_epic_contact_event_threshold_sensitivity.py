import unittest

from scripts.evaluate_epic_contact_event_threshold_sensitivity import (
    best_trivial_auprc,
    sensitivity_gate,
)


def curve(release_auprc=0.30, onset_auprc=0.60):
    return [
        {
            "threshold_m": threshold_m,
            "evaluation": {
                "test": {
                    "copy_current": {
                        "onset": {"auprc": 0.10},
                        "release": {"auprc": 0.09},
                    },
                    "distance_linear": {
                        "onset": {"auprc": 0.06},
                        "release": {"auprc": 0.08},
                    },
                    "learned_logistic": {
                        "onset": {"auprc": onset_auprc},
                        "release": {"auprc": release_auprc},
                    },
                },
            },
        }
        for threshold_m in (0.0005, 0.001, 0.0015)
    ]


class ThresholdSensitivityTest(unittest.TestCase):
    def test_best_trivial_uses_maximum(self):
        evaluation = curve()[0]["evaluation"]["test"]
        self.assertAlmostEqual(
            best_trivial_auprc(evaluation, "release"),
            0.09,
        )

    def test_gate_passes_when_learned_beats_trivial(self):
        self.assertTrue(sensitivity_gate(curve())["overall"])

    def test_gate_fails_when_neighbor_release_collapses(self):
        rows = curve()
        rows[0]["evaluation"]["test"]["learned_logistic"]["release"][
            "auprc"
        ] = 0.05
        self.assertFalse(sensitivity_gate(rows)["overall"])


if __name__ == "__main__":
    unittest.main()
