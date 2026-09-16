import unittest

import numpy as np

from manip.contact.event_schema import (
    ContactEventSchema,
    aggregate_sequence_hand_records,
    analyze_hand_sequence,
    interval_runs,
    match_intervals,
)


class ContactEventSchemaTests(unittest.TestCase):
    def test_interval_runs_and_minimum_length(self):
        mask = np.asarray([False, True, True, False, True, True, False])
        self.assertEqual(interval_runs(mask), [(1, 2), (4, 5)])
        self.assertEqual(interval_runs(mask, 3), [])

    def test_match_intervals_is_one_to_one(self):
        matches = match_intervals(
            [(10, 20), (30, 40)],
            [(12, 22), (32, 43)],
            5,
        )
        self.assertEqual(matches, [0, 1])

    def test_hand_sequence_counts_and_timing(self):
        schema = ContactEventSchema()
        gt = np.asarray([0, 1, 1, 0, 0, 0])
        pred = np.asarray([0, 0, 1, 1, 0, 0])
        distance = np.where(
            pred,
            schema.contact_threshold_m - 0.001,
            schema.contact_threshold_m + 0.001,
        )
        result = analyze_hand_sequence(gt, pred, distance, schema)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["tn"], 3)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["matched_gt_interval_count"], 1)
        self.assertEqual(result["missed_gt_interval_count"], 0)
        self.assertEqual(result["unmatched_pred_interval_count"], 0)
        self.assertEqual(result["onset_errors_frames"], [1])
        self.assertEqual(result["release_delays_frames"], [1])
        self.assertEqual(result["continued_after_release_frames"], [1])

    def test_aggregate_micro_counts(self):
        records = [
            {
                "sequence": "a",
                "hand": "left",
                "tp": 2,
                "fp": 1,
                "tn": 3,
                "fn": 2,
                "f1": 4 / 7,
                "gt_interval_count": 1,
                "pred_interval_count": 2,
                "matched_gt_interval_count": 1,
                "missed_gt_interval_count": 0,
                "unmatched_pred_interval_count": 1,
                "continued_after_release_frames": [1],
                "mean_onset_error_frames": 1.0,
                "median_abs_onset_error_frames": 1.0,
                "mean_positive_release_delay_frames": 1.0,
            },
            {
                "sequence": "b",
                "hand": "left",
                "tp": 3,
                "fp": 0,
                "tn": 2,
                "fn": 1,
                "f1": 6 / 7,
                "gt_interval_count": 1,
                "pred_interval_count": 1,
                "matched_gt_interval_count": 1,
                "missed_gt_interval_count": 0,
                "unmatched_pred_interval_count": 0,
                "continued_after_release_frames": [],
                "mean_onset_error_frames": 0.0,
                "median_abs_onset_error_frames": 0.0,
                "mean_positive_release_delay_frames": None,
            },
        ]
        result = aggregate_sequence_hand_records(records)["left"]
        self.assertEqual(result["micro"]["tp"], 5)
        self.assertEqual(result["micro"]["fp"], 1)
        self.assertEqual(result["micro"]["tn"], 5)
        self.assertEqual(result["micro"]["fn"], 3)
        self.assertAlmostEqual(
            result["sequence_macro"]["f1"],
            (4 / 7 + 6 / 7) / 2,
        )


if __name__ == "__main__":
    unittest.main()
