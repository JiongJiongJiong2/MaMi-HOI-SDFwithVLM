import unittest

import numpy as np

from scripts.audit_arctic_handover_gate import (
    binary_segments,
    close_short_false_gaps,
    contact_summary,
    pilot_gate,
    remove_short_true_runs,
    role_switch_candidates,
    stable_contact,
)


class ArcticHandoverGateTest(unittest.TestCase):
    def test_close_short_false_gaps(self):
        mask = np.asarray([0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
        result = close_short_false_gaps(mask, 2)
        np.testing.assert_array_equal(
            result,
            [0, 1, 1, 1, 1, 1, 1, 0],
        )

    def test_remove_short_true_runs(self):
        mask = np.asarray([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)
        result = remove_short_true_runs(mask, 3)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 1, 1, 1, 0])

    def test_stable_contact_and_segments(self):
        raw = np.asarray([0, 1, 0, 1, 1, 1, 0], dtype=bool)
        stable = stable_contact(raw, minimum_length=3, maximum_gap=0)
        self.assertEqual(binary_segments(stable), [(3, 6)])

    def test_role_switch_candidate(self):
        outgoing = [(0, 20)]
        receiving = [(25, 45)]
        candidates = role_switch_candidates(
            outgoing,
            receiving,
            15,
            15,
            "right",
            "left",
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["direction"], "right_to_left")
        self.assertEqual(candidates[0]["onset_minus_release"], 5)

    def test_contact_summary_counts_transfer(self):
        right = np.full(60, 0.02)
        left = np.full(60, 0.02)
        right[:20] = 0.001
        left[25:50] = 0.001
        summary = contact_summary(
            right,
            left,
            0.003,
            15,
            0,
            15,
        )
        self.assertEqual(summary["stable_segments_right"], 1)
        self.assertEqual(summary["stable_segments_left"], 1)
        self.assertEqual(len(summary["candidates"]), 1)

    def test_gate_requires_official_test_separation(self):
        train = {
            "candidate_count": 10,
            "candidate_participants": 3,
            "candidate_directions": {
                "left_to_right": 2,
                "right_to_left": 2,
            },
            "candidate_objects": ["a", "b", "c"],
            "bimanual_sequences": 1,
        }
        val = {
            "bimanual_sequences": 1,
            "candidate_count": 1,
        }
        gate = pilot_gate(train, val)
        self.assertTrue(gate["pilot_overall"])
        self.assertFalse(gate["official_test_available"])
        self.assertFalse(gate["final_benchmark_ready"])


if __name__ == "__main__":
    unittest.main()
