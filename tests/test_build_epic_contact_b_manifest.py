import io
import pickle
import unittest

import numpy as np

from scripts.build_epic_contact_b_manifest import (
    ManifestAccumulator,
    StreamingTopLevelUnpickler,
    contact_state,
)


class EpicContactManifestTest(unittest.TestCase):
    def test_contact_state(self):
        contact, minimum = contact_state(
            np.asarray([0.002, 0.004]),
            np.asarray([1.0]),
            0.003,
        )
        self.assertTrue(contact)
        self.assertEqual(minimum, 0.002)

    def test_streaming_unpickler_processes_each_top_level_sample(self):
        data = {
            str(index): {
                "array": np.arange(4),
                "value": index,
            }
            for index in range(25)
        }
        stream = io.BytesIO(pickle.dumps(data, protocol=4))
        seen = []
        unpickler = StreamingTopLevelUnpickler(
            stream,
            lambda key, sample: seen.append((key, sample["value"])),
            trim_memo=True,
            memo_window=100,
        )
        result = unpickler.load()
        self.assertEqual(len(seen), 25)
        self.assertEqual(
            result,
            {str(index): None for index in range(25)},
        )

    def test_episode_observation_flags(self):
        accumulator = ManifestAccumulator(0.003, 1)
        rows = [
            {"frame": 0, "contact": False, "min_distance_m": 1.0},
            {"frame": 1, "contact": True, "min_distance_m": 0.001},
            {"frame": 2, "contact": True, "min_distance_m": 0.002},
            {"frame": 3, "contact": False, "min_distance_m": 1.0},
        ]
        result = accumulator._episodes_from_segment(
            "train",
            "P01_01_0000000000_0000000003",
            "left",
            "cup",
            rows,
        )
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["onset_observed"])
        self.assertTrue(result[0]["release_observed"])
        self.assertEqual(result[0]["contact_frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
