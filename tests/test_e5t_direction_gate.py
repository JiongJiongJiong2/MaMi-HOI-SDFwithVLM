import math
import unittest

import numpy as np

from scripts.analyze_e5t_direction_gate import (
    cluster_bootstrap_count_difference,
    choose_e5,
    fit_threshold,
    selection_stats,
)


def row(feature, split="train", sequence="s", pass_e5=True, pass_e5t=True):
    return {
        "split": split,
        "sequence": sequence,
        "near_outward_fraction": feature,
        "e5_contact_pass": pass_e5,
        "e5t_contact_pass": pass_e5t,
        "e5_combined_pass": pass_e5,
        "e5t_combined_pass": pass_e5t,
    }


class E5TDirectionGateTest(unittest.TestCase):
    def test_fit_chooses_train_threshold(self):
        rows = [
            row(0.1, sequence="a"),
            row(0.2, sequence="b"),
            row(0.9, pass_e5t=False, sequence="c"),
            row(0.8, pass_e5t=False, sequence="d"),
            *[
                row(0.05 * index, sequence=f"n{index}")
                for index in range(1, 10)
            ],
        ]
        fitted = fit_threshold(rows, "near_outward_fraction")
        self.assertGreater(fitted["threshold"], 0.2)
        self.assertLess(fitted["threshold"], 0.9)
        self.assertEqual(
            fitted["train_stats"]["combined_pass_count"],
            12,
        )

    def test_invalid_risk_keeps_e5t(self):
        item = row(None)
        self.assertFalse(
            choose_e5(item, "near_outward_fraction", math.inf)
        )

    def test_non_contact_failure_never_selects_e5(self):
        item = row(1.0, pass_e5=False, pass_e5t=True)
        self.assertFalse(
            choose_e5(item, "near_outward_fraction", 0.5)
        )

    def test_selection_stats_uses_candidate_when_not_selected(self):
        rows = [
            row(0.1, pass_e5=True, pass_e5t=True),
            row(0.9, pass_e5=True, pass_e5t=False),
        ]
        stats = selection_stats(
            rows,
            "near_outward_fraction",
            0.5,
        )
        self.assertEqual(stats["selected_e5_count"], 1)
        self.assertEqual(stats["contact_pass_count"], 2)
        self.assertEqual(stats["combined_pass_count"], 2)

    def test_cluster_bootstrap_preserves_cluster_counts(self):
        selected = np.asarray([True, True, False, False])
        baseline = np.asarray([False, False, False, False])
        groups = np.asarray(["a", "a", "b", "b"])
        result = cluster_bootstrap_count_difference(
            selected,
            baseline,
            groups,
            samples=100,
            seed=1,
        )
        self.assertEqual(result["mean"], 2)
        self.assertEqual(result["ci95"], [0, 4])


if __name__ == "__main__":
    unittest.main()
