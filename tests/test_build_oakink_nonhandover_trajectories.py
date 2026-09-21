import unittest

from scripts.build_oakink_nonhandover_trajectories import (
    parse_nonhandover_sequence,
    split_for_subject,
)


class OakInkNonHandoverTrajectoriesTest(unittest.TestCase):
    def test_parse_nonhandover_sequence(self):
        row = parse_nonhandover_sequence("A01001_0001_0000")
        self.assertEqual(row["intent_id"], "0001")
        self.assertEqual(row["subject"], "0000")

    def test_reject_handover(self):
        self.assertIsNone(
            parse_nonhandover_sequence("A01001_0004_0001_0003")
        )

    def test_split_for_subject(self):
        self.assertEqual(split_for_subject("0001"), "test")
        self.assertEqual(split_for_subject("0005"), "train")


if __name__ == "__main__":
    unittest.main()
