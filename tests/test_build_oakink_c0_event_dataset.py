import unittest

import numpy as np

try:
    from scripts.build_oakink_c0_event_dataset import (
        contact_region,
        event_window_indices,
        object_diameter,
        object_vertex_distances,
    )
except ModuleNotFoundError:
    from build_oakink_c0_event_dataset import (  # noqa: E402
        contact_region,
        event_window_indices,
        object_diameter,
        object_vertex_distances,
    )


class BuildOakInkC0EventDatasetTest(unittest.TestCase):
    def test_event_window_preserves_relative_position(self):
        indices, valid = event_window_indices(
            release=40,
            length=80,
            history=30,
            future=30,
        )
        self.assertEqual(indices.shape, (61,))
        self.assertEqual(indices[30], 40)
        self.assertEqual(valid.sum(), 61)

    def test_event_window_marks_out_of_range(self):
        indices, valid = event_window_indices(
            release=10,
            length=25,
            history=30,
            future=30,
        )
        self.assertFalse(valid[0])
        self.assertTrue(valid[30])
        self.assertFalse(valid[-1])
        self.assertTrue(np.all(indices[valid] >= 0))
        self.assertTrue(np.all(indices[valid] < 25))

    def test_contact_region_is_object_vertex_indexed(self):
        object_vertices = np.asarray([
            [0.0, 0.0, 0.0],
            [0.004, 0.0, 0.0],
            [0.010, 0.0, 0.0],
        ])
        hand_vertices = np.asarray([[0.0, 0.0, 0.0]])
        indices, distances = contact_region(
            object_vertices,
            hand_vertices,
            threshold_m=0.005,
        )
        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_allclose(distances, [0.0, 0.004])

    def test_object_diameter_is_bbox_diagonal(self):
        vertices = np.asarray([
            [-1.0, -2.0, -3.0],
            [1.0, 2.0, 3.0],
        ])
        self.assertAlmostEqual(object_diameter(vertices), np.sqrt(56.0))

    def test_object_vertex_distances_are_per_object_vertex(self):
        object_vertices = np.asarray([
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        hand_vertices = np.asarray([[0.0, 1.0, 0.0]])
        np.testing.assert_allclose(
            object_vertex_distances(object_vertices, hand_vertices),
            [1.0, np.sqrt(10.0)],
        )


if __name__ == "__main__":
    unittest.main()
