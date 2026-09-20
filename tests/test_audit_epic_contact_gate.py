import unittest

import numpy as np

from scripts.audit_epic_contact_gate import (
    contact_run_sequence,
    contact_state,
    frame_gap_summary,
    summarize,
    summarize_keys,
)


class EpicContactGateTest(unittest.TestCase):
    def test_contact_state_uses_validity_and_threshold(self):
        contact, minimum = contact_state(
            np.asarray([0.002, 0.004]),
            np.asarray([1.0]),
            0.003,
        )
        self.assertTrue(contact)
        self.assertEqual(minimum, 0.002)
        contact, minimum = contact_state(
            np.asarray([0.002]),
            np.asarray([0.0]),
            0.003,
        )
        self.assertFalse(contact)
        self.assertIsNone(minimum)

    def test_frame_gap_summary(self):
        result = frame_gap_summary([1, 2, 4, 7])
        self.assertEqual(result["gap_1_count"], 1)
        self.assertEqual(result["gap_gt_1_count"], 2)
        self.assertEqual(result["max_gap"], 3)

    def test_contact_run_sequence_detects_direct_role_switch(self):
        rows = [
            {
                "frame": 0,
                "left_contact": True,
                "right_contact": False,
            },
            {
                "frame": 1,
                "left_contact": False,
                "right_contact": True,
            },
            {
                "frame": 2,
                "left_contact": True,
                "right_contact": True,
            },
        ]
        result = contact_run_sequence(rows)
        self.assertEqual(result["left_to_right_direct_count"], 1)
        self.assertEqual(
            [run["code"] for run in result["runs"]],
            ["L", "R", "B"],
        )

    def test_summary_counts_contact_transitions(self):
        data = {}
        for index, distance in enumerate((0.001, 0.001, 0.004)):
            data[
                "P01_01_0000000000_0000000002_left_cup"
                f"_frame_{index:010d}"
            ] = {
                "video_id": "P01_01",
                "frame_num": np.asarray([index]),
                "left_valid": np.asarray([1.0]),
                "right_valid": np.asarray([0.0]),
                "dist.lo": np.asarray([distance]),
                "dist.ro": np.asarray([1.0]),
                "obj_class": np.asarray([1]),
            }
        result = summarize(data)
        self.assertEqual(result["continuity"]["contact_start_count"], 1)
        self.assertEqual(result["continuity"]["contact_end_count"], 1)
        self.assertEqual(result["merged_frame_count"], 3)

    def test_key_summary_parses_clip_ids(self):
        keys = [
            "P01_01_0000000000_0000000002_left_cup_frame_0000000000",
            "P01_01_0000000000_0000000002_left_cup_frame_0000000001",
            "P01_01_0000000003_0000000005_right_bowl_frame_0000000003",
        ]
        result = summarize_keys(keys)
        self.assertEqual(result["key_count"], 3)
        self.assertEqual(result["video_count"], 1)
        self.assertEqual(result["clip_count"], 2)


if __name__ == "__main__":
    unittest.main()
