import unittest

from scripts.audit_predicted_hand_surface_queries import (
    aggregate_sequence_rows,
    summarize_predictions,
)


class PredictedHandSurfaceAuditTests(unittest.TestCase):
    def test_summary_preserves_zero_clearance_and_counts_contact(self):
        predictions = [
            {
                "sequence": "seq_a",
                "hand": "left",
                "distance_m": 0.0,
                "gt_contact": True,
            },
            {
                "sequence": "seq_a",
                "hand": "left",
                "distance_m": 0.06,
                "gt_contact": True,
            },
            {
                "sequence": "seq_b",
                "hand": "left",
                "distance_m": 0.04,
                "gt_contact": False,
            },
            {
                "sequence": "seq_b",
                "hand": "right",
                "distance_m": 0.02,
                "gt_contact": True,
            },
        ]
        rows = summarize_predictions(predictions, 0.05)
        by_key = {
            (row["sequence"], row["hand"]): row
            for row in rows
        }
        self.assertEqual(
            by_key[("seq_a", "left")]["mean_clearance_on_gt_contact_mm"],
            30.0,
        )
        self.assertEqual(by_key[("seq_a", "left")]["fn"], 1)
        self.assertEqual(by_key[("seq_b", "left")]["fp"], 1)

        aggregate = aggregate_sequence_rows(rows)
        self.assertEqual(
            aggregate["left"]["mean_clearance_on_gt_contact_macro_mm"],
            30.0,
        )
        self.assertEqual(aggregate["left"]["tp"], 1)
        self.assertEqual(aggregate["left"]["fn"], 1)
        self.assertEqual(aggregate["left"]["fp"], 1)
        self.assertEqual(aggregate["right"]["tp"], 1)


if __name__ == "__main__":
    unittest.main()
