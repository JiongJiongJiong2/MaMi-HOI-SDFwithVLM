import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_contactopt_sequence_dataset import (
    contiguous_runs,
    evenly_spaced_indices,
    iter_shifted_windows,
    split_sequences,
    stable_sequence_bucket,
)


class ContactOptSequenceDatasetTest(unittest.TestCase):
    def test_contiguous_runs_handles_edges_and_gaps(self):
        self.assertEqual(contiguous_runs([False, False]), [])
        self.assertEqual(
            contiguous_runs([True, True, False, True]),
            [(0, 1), (3, 3)],
        )

    def test_shifted_windows_cover_long_runs(self):
        self.assertEqual(
            list(iter_shifted_windows(10, 19, window=8, stride=4)),
            [(10, 17), (12, 19)],
        )
        self.assertEqual(
            list(iter_shifted_windows(0, 4, window=5, stride=2)),
            [(0, 4)],
        )
        self.assertEqual(
            list(iter_shifted_windows(0, 3, window=5, stride=2)),
            [],
        )

    def test_evenly_spaced_indices_are_deterministic_and_bounded(self):
        self.assertEqual(evenly_spaced_indices(3, 8), [0, 1, 2])
        self.assertEqual(evenly_spaced_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(evenly_spaced_indices(10, 1), [5])

    def test_sequence_split_respects_explicit_dev_and_is_disjoint(self):
        sequences = ["seq_a", "seq_b", "seq_c", "seq_d"]
        split = split_sequences(
            sequences,
            dev_sequences={"seq_b"},
            test_fraction=0.5,
            split_seed=7,
        )
        assigned = split["train"] + split["dev"] + split["test"]
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(len(assigned), len(sequences))
        self.assertEqual(split["dev"], ["seq_b"])
        self.assertTrue(set(split["train"]).isdisjoint(split["test"]))

    def test_stable_sequence_bucket_is_reproducible(self):
        first = stable_sequence_bucket("sub16_monitor_001", 123)
        second = stable_sequence_bucket("sub16_monitor_001", 123)
        different = stable_sequence_bucket("sub16_monitor_002", 123)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.0)
        self.assertLess(first, 1.0)
        self.assertNotEqual(first, different)


if __name__ == "__main__":
    unittest.main()
