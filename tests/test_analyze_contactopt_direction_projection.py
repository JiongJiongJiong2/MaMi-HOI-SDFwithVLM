import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_contactopt_direction_projection import (
    acceptance_gate,
    cluster_bootstrap_count_difference,
    summarize_result,
)


def result_row(
    chunk,
    sequence,
    split,
    combined,
    contact,
    temporal,
):
    return {
        "chunk_id": chunk,
        "sequence": sequence,
        "split": split,
        "object_name": "object",
        "frames": [0, 1, 2, 3, 4, 5, 6, 7],
        "optimized_combined_pass": combined,
        "optimized_contact_gate": {
            "passed": contact,
            "mean_distance_relative_change": -0.1,
        },
        "optimized_temporal_gate": {"passed": temporal},
        "optimized_acceleration_ratio": 0.9,
        "optimized_jerk_ratio": 0.8,
    }


class AnalyzeContactOptDirectionProjectionTest(unittest.TestCase):
    def test_cluster_bootstrap_count_difference(self):
        result = cluster_bootstrap_count_difference(
            [1, 1, 0, 1],
            [0, 1, 0, 0],
            ["a", "a", "b", "c"],
            samples=100,
            seed=1,
        )
        self.assertEqual(result["mean"], 2)
        self.assertEqual(len(result["ci95"]), 2)

    def test_acceptance_gate_records_mechanism_and_engineering_cases(self):
        keys = [
            *[("train", f"train_{index}") for index in range(12)],
            *[("dev", f"dev_{index}") for index in range(8)],
        ]
        arm_rows = {}
        for arm_index, arm in enumerate((
            "ordinary_smoothing",
            "contact_projection",
            "unconstrained_continuation",
            "strong_contact",
            "direction_constrained",
        )):
            arm_rows[arm] = {}
            for index, (split, sequence) in enumerate(keys):
                direction_bonus = arm == "direction_constrained"
                arm_rows[arm][(
                    f"{split}_{sequence}",
                    (0, 1, 2, 3, 4, 5, 6, 7),
                )] = result_row(
                    f"{split}_{sequence}",
                    sequence,
                    split,
                    combined=direction_bonus,
                    contact=True,
                    temporal=True,
                )
        e5_rows = {}
        e5t_rows = {}
        for index, (split, sequence) in enumerate(keys):
            key = (
                f"{split}_{sequence}",
                (0, 1, 2, 3, 4, 5, 6, 7),
            )
            e5_rows[key] = result_row(
                f"{split}_{sequence}",
                sequence,
                split,
                combined=False,
                contact=True,
                temporal=False,
            )
            e5t_rows[key] = result_row(
                f"{split}_{sequence}",
                sequence,
                split,
                combined=False,
                contact=False,
                temporal=True,
            )
        _, summaries, differences = summarize_result(
            arm_rows,
            e5_rows,
            e5t_rows,
            bootstrap_samples=100,
            seed=1,
        )
        gate = acceptance_gate(summaries, differences)
        self.assertTrue(gate["checks"]["pass"])
        self.assertIn(
            gate["decision"],
            ("GO direction mechanism", "ENGINEERING_ONLY direction correction"),
        )

    def test_acceptance_gate_rejects_contact_regression(self):
        keys = [
            ("train", "a"),
            ("train", "b"),
            ("dev", "c"),
            ("dev", "d"),
        ]
        arm_rows = {}
        for arm in (
            "ordinary_smoothing",
            "contact_projection",
            "unconstrained_continuation",
            "strong_contact",
            "direction_constrained",
        ):
            arm_rows[arm] = {}
            for index, (split, sequence) in enumerate(keys):
                contact = not (
                    arm == "direction_constrained" and index == 0
                )
                arm_rows[arm][(
                    f"{split}_{sequence}",
                    (0, 1, 2, 3, 4, 5, 6, 7),
                )] = result_row(
                    f"{split}_{sequence}",
                    sequence,
                    split,
                    combined=True,
                    contact=contact,
                    temporal=True,
                )
        e5_rows = {}
        e5t_rows = {}
        for index, (split, sequence) in enumerate(keys):
            key = (
                f"{split}_{sequence}",
                (0, 1, 2, 3, 4, 5, 6, 7),
            )
            e5_rows[key] = result_row(
                f"{split}_{sequence}",
                sequence,
                split,
                combined=True,
                contact=True,
                temporal=True,
            )
            e5t_rows[key] = result_row(
                f"{split}_{sequence}",
                sequence,
                split,
                combined=False,
                contact=False,
                temporal=True,
            )
        _, summaries, differences = summarize_result(
            arm_rows,
            e5_rows,
            e5t_rows,
            bootstrap_samples=100,
            seed=1,
        )
        gate = acceptance_gate(summaries, differences)
        self.assertTrue(
            gate["checks"]["contact_not_below_e5t_train_and_dev"]
        )
        self.assertTrue(
            gate["checks"]["contact_regression_vs_e5_within_12"]
        )


if __name__ == "__main__":
    unittest.main()
