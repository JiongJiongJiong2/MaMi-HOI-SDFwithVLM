import unittest

import numpy as np

from scripts.build_oakink_handover_manifest import (
    parse_sample_name,
    parse_sequence,
    role_switch_event,
    split_for_subjects,
)


class OakInkHandoverManifestTest(unittest.TestCase):
    def test_parse_sequence(self):
        row = parse_sequence("A15015_0004_0008_0007")
        self.assertEqual(row["object_id"], "A15015")
        self.assertEqual(row["giver"], "0008")
        self.assertEqual(row["receiver"], "0007")

    def test_split_for_subjects(self):
        self.assertEqual(split_for_subjects("0001", "0003"), "test")
        self.assertIsNone(split_for_subjects("0001", "0006"))

    def test_parse_sample_name(self):
        row = parse_sample_name(
            "anno/hand_v/A15015_0004_0008_0007__"
            "2021-10-03-14-45-01__0__106__0.pkl"
        )
        self.assertEqual(row["sequence"], "A15015_0004_0008_0007")
        self.assertEqual(row["frame"], 106)
        self.assertEqual(row["subject_flag"], "0")

    def test_role_switch_event(self):
        giver = np.zeros(80, dtype=bool)
        receiver = np.zeros(80, dtype=bool)
        giver[:20] = True
        receiver[25:60] = True
        _, _, candidates = role_switch_event(
            giver,
            receiver,
            minimum_frames=15,
            window_frames=30,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["onset_minus_release"], 5)


if __name__ == "__main__":
    unittest.main()
