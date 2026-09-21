import unittest

import numpy as np

try:
    from scripts.evaluate_oakink_c0_oracle_reranking import (
        clipped_distance,
        rank_of,
        receiver_reference_position,
        select_top,
        standardize,
        to_builtin,
        to_object_frame,
    )
except ModuleNotFoundError:
    from evaluate_oakink_c0_oracle_reranking import (  # noqa: E402
        clipped_distance,
        rank_of,
        receiver_reference_position,
        select_top,
        standardize,
        to_builtin,
        to_object_frame,
    )


class EvaluateOakInkC0OracleRerankingTest(unittest.TestCase):
    def test_standardize_zero_variance(self):
        np.testing.assert_allclose(standardize([2.0, 2.0]), [0.0, 0.0])

    def test_clipped_distance(self):
        self.assertEqual(clipped_distance(0.015, 0.030), 0.5)
        self.assertEqual(clipped_distance(-1.0, 0.030), 0.0)
        self.assertEqual(clipped_distance(1.0, 0.030), 1.0)

    def test_select_top_uses_stable_event_order_for_ties(self):
        selected = select_top(
            scores=np.asarray([1.0, 1.0]),
            candidate_event_indices=np.asarray([9, 3]),
        )
        self.assertEqual(selected, 3)

    def test_rank_of(self):
        rank = rank_of(
            candidate_event_indices=np.asarray([4, 7, 9]),
            scores=np.asarray([0.5, 0.9, 0.1]),
            target_event_index=4,
        )
        self.assertEqual(rank, 2)

    def test_to_object_frame_inverts_rigid_transform(self):
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        points = np.asarray([[1.5, 2.0, 3.0]])
        np.testing.assert_allclose(
            to_object_frame(points, transform),
            [[0.5, 0.0, 0.0]],
        )

    def test_receiver_reference_prefers_receiver_interval(self):
        events = {
            "release_indices": np.asarray([50]),
            "receiver_starts": np.asarray([45]),
            "receiver_ends": np.asarray([55]),
            "valid": np.ones((1, 61), dtype=bool),
            "receiver_distance_m": np.full((1, 61), 0.020),
        }
        self.assertEqual(receiver_reference_position(events, 0), 25)

    def test_to_builtin_converts_numpy_scalars(self):
        result = to_builtin({
            "flag": np.bool_(True),
            "count": np.int64(3),
            "values": [np.float32(1.5)],
        })
        self.assertIs(result["flag"], True)
        self.assertIs(result["count"], 3)
        self.assertEqual(result["values"], [1.5])


if __name__ == "__main__":
    unittest.main()
