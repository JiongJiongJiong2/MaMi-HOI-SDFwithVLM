import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from build_epic_contact_memory_manifest import (
    normalize_object_vertices,
    split_contiguous,
    top_contact_pairs,
)


class EpicContactMemoryManifestTest(unittest.TestCase):
    def test_normalize_object_vertices_recovers_canonical_mesh(self):
        canonical = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        translation = np.asarray([0.1, 0.2, 0.3])
        world = canonical @ rotation.T + translation
        recovered = normalize_object_vertices(
            world,
            rotation,
            translation,
        )
        np.testing.assert_allclose(recovered, canonical, atol=1e-12)

    def test_top_contact_pairs_prefers_smallest_distances(self):
        distances = np.asarray([0.003, 0.001, 0.002])
        object_indices = np.asarray([10, 11, 12])
        hand_vertices = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )
        object_vertices = np.asarray(
            [[0.0, 0.0, 0.0]] * 13
        )
        pairs = top_contact_pairs(
            distances,
            object_indices,
            hand_vertices,
            object_vertices,
            top_k=2,
        )
        self.assertEqual(
            pairs["hand_indices"].tolist(),
            [1, 2],
        )
        self.assertEqual(
            pairs["object_indices"].tolist(),
            [11, 12],
        )
        np.testing.assert_allclose(
            pairs["distances_m"],
            [0.001, 0.002],
            atol=1e-7,
        )

    def test_split_contiguous_splits_on_large_gap(self):
        self.assertEqual(
            split_contiguous([1, 2, 4, 5, 9], 1),
            [[1, 2], [4, 5], [9]],
        )

    def test_top_contact_pairs_rejects_bad_object_index(self):
        with self.assertRaises(ValueError):
            top_contact_pairs(
                np.asarray([0.001]),
                np.asarray([3]),
                np.zeros((1, 3)),
                np.zeros((2, 3)),
                top_k=1,
            )


if __name__ == "__main__":
    unittest.main()
