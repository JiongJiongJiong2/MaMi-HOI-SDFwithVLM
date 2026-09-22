import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_epic_contact_memory import (
    ACTION_TO_ID,
    causal_window,
    class_weights,
    macro_f1,
    reference_tracks,
)


class EpicContactMemoryTrainTest(unittest.TestCase):
    def test_causal_window_never_uses_future(self):
        values = np.arange(10, dtype=np.float32)[:, None]
        window = causal_window(values[:5], history=3)
        self.assertEqual(window.shape, (3, 1))
        np.testing.assert_array_equal(
            window[:, 0],
            np.asarray([2.0, 3.0, 4.0]),
        )
        one_dimensional = causal_window(
            np.asarray([0, 1, 2, 3]),
            history=2,
        )
        np.testing.assert_array_equal(
            one_dimensional,
            np.asarray([2, 3]),
        )

    def test_reference_tracks_hold_update_and_close(self):
        frames = 4
        hand_vertices = np.asarray([
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.03, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.03, 0.0, 0.0], [0.1, 0.0, 0.0]],
        ], dtype=np.float32)
        arrays = {
            "contact": np.asarray([True, True, True, False]),
            "min_distance_m": np.asarray(
                [0.0005, 0.0005, 0.003, 0.004],
                dtype=np.float32,
            ),
            "top_hand_indices": np.asarray(
                [[0, 1]] * frames,
                dtype=np.int64,
            ),
            "top_object_indices": np.asarray(
                [[0, 0]] * frames,
                dtype=np.int64,
            ),
            "top_distances_m": np.asarray(
                [
                    [0.0005, 0.002],
                    [0.0005, 0.002],
                    [0.003, 0.004],
                    [0.004, 0.004],
                ],
                dtype=np.float32,
            ),
            "hand_vertices": hand_vertices,
            "object_vertices": np.zeros((1, 3), dtype=np.float32),
            "object_rotation": np.repeat(
                np.eye(3, dtype=np.float32)[None],
                frames,
                axis=0,
            ),
            "object_translation": np.zeros(
                (frames, 3),
                dtype=np.float32,
            ),
            "object_diameter_m": np.asarray(0.1, dtype=np.float32),
        }
        sequences = reference_tracks(
            arrays,
            object_name="bowl",
            hand="left",
            hold_ratio=0.01,
            update_ratio=0.02,
            update_persistence=1,
            close_persistence=1,
            history=8,
        )
        self.assertEqual(len(sequences), 1)
        labels = sequences[0]["labels"]
        self.assertIn(ACTION_TO_ID["update"], labels)
        self.assertEqual(labels[-1], ACTION_TO_ID["close"])

    def test_class_weights_and_macro_f1(self):
        weights = class_weights(np.asarray([0, 0, 0, 1, 2, 3]))
        self.assertEqual(weights.shape, (4,))
        macro, per_class = macro_f1(
            np.asarray([0, 1, 2, 3]),
            np.asarray([0, 1, 2, 3]),
        )
        self.assertEqual(macro, 1.0)
        self.assertEqual(per_class, [1.0, 1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
