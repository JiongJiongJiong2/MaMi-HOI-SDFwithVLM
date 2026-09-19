import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_contactopt_sequence_dataset import build_chunks, group_windows


class ContactOptSequenceDatasetRunnerTest(unittest.TestCase):
    def test_build_chunks_respects_frame_budget_and_overlap(self):
        windows = [
            {"frames": [0, 1, 2, 3, 4, 5, 6, 7]},
            {"frames": [4, 5, 6, 7, 8, 9, 10, 11]},
            {"frames": [20, 21, 22, 23, 24, 25, 26, 27]},
        ]
        chunks = build_chunks(windows, max_frames_per_chunk=16)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 2)
        self.assertEqual(len(chunks[1]), 1)

    def test_group_windows_rejects_test_without_opt_in(self):
        manifest = {
            "windows": [
                {
                    "split": "test",
                    "sequence": "seq_test",
                    "frames": [0],
                }
            ]
        }
        with self.assertRaises(ValueError):
            group_windows(manifest, ("test",), [], allow_test=False)

    def test_group_windows_filters_split_and_sequence(self):
        manifest = {
            "windows": [
                {"split": "train", "sequence": "a", "frames": [0]},
                {"split": "dev", "sequence": "b", "frames": [1]},
                {"split": "test", "sequence": "c", "frames": [2]},
            ]
        }
        grouped = group_windows(
            manifest,
            ("train", "dev"),
            ["b"],
            allow_test=False,
        )
        self.assertEqual(set(grouped), {"b"})


if __name__ == "__main__":
    unittest.main()
