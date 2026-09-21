import unittest

import numpy as np

from scripts.analyze_arctic_role_switch_candidates import (
    classify_tier,
    enrich_candidate,
    split_summary,
)


class Args:
    minimum_contact_frames = 15
    tier_a_offset = 5
    tier_b_offset = 10
    tier_a_overlap = 10
    tier_b_overlap = 20
    outgoing_gap_frames = 15


class RoleSwitchCandidateTest(unittest.TestCase):
    def test_tier_a_classification(self):
        row = {
            "onset_minus_release": 3,
            "overlap_before_release": 2,
            "outgoing_gap": 20,
            "receiving_stable_after_release": 15,
        }
        self.assertEqual(classify_tier(row, Args), "tier_a")

    def test_tier_b_classification(self):
        row = {
            "onset_minus_release": -8,
            "overlap_before_release": 15,
            "outgoing_gap": 20,
            "receiving_stable_after_release": 15,
        }
        self.assertEqual(classify_tier(row, Args), "tier_b")

    def test_enrich_candidate(self):
        right = np.full(80, 0.02)
        left = np.full(80, 0.02)
        right[:20] = 0.001
        left[20:60] = 0.001
        trajectory = {
            "right": right,
            "left": left,
            "length": 80,
        }
        row = {
            "outgoing_hand": "right",
            "receiving_hand": "left",
            "outgoing_release": 20,
            "receiving_start": 20,
            "receiving_end": 60,
        }
        result = enrich_candidate(row, trajectory, 0.003, 15)
        self.assertEqual(result["outgoing_duration"], 20)
        self.assertEqual(result["receiving_duration"], 40)
        self.assertEqual(result["outgoing_gap"], 60)
        self.assertEqual(result["receiving_stable_after_release"], 15)

    def test_split_summary(self):
        rows = [
            {
                "split": "train",
                "sequence": "s01/a",
                "participant_id": "s01",
                "object_name": "box",
                "direction": "left_to_right",
            },
            {
                "split": "train",
                "sequence": "s02/b",
                "participant_id": "s02",
                "object_name": "box",
                "direction": "right_to_left",
            },
        ]
        result = split_summary(rows, "train")
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["candidate_participants"], 2)
        self.assertEqual(result["candidate_objects"], ["box"])


if __name__ == "__main__":
    unittest.main()
