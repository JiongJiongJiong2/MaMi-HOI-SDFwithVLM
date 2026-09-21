import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.generate_dwm_probe_dataset import (
    PROBE_LENGTHS,
    collect_split,
)


HAS_MUJOCO = importlib.util.find_spec("mujoco") is not None


@unittest.skipUnless(HAS_MUJOCO, "mujoco is not installed")
class DWMProbeDatasetTest(unittest.TestCase):
    def test_smoke_schema_and_rank(self):
        args = SimpleNamespace(
            seed=20260922,
            resets_per_object=1,
            object_limit=1,
        )
        data = collect_split(args, "train")
        expected_groups = 1 + 2 * len(PROBE_LENGTHS)
        self.assertEqual(data["initial_state"].shape, (expected_groups, 168))
        self.assertEqual(
            data["probe_action"].shape,
            (expected_groups, 4, 51),
        )
        self.assertEqual(
            data["candidate_action"].shape,
            (expected_groups, 13, 8, 51),
        )
        self.assertEqual(data["utility"].shape, (expected_groups, 13))
        expected_order = np.argsort(
            -data["utility"].astype(np.float64),
            axis=1,
            kind="stable",
        )
        expected_rank = np.empty_like(expected_order)
        expected_rank[
            np.arange(expected_order.shape[0])[:, None],
            expected_order,
        ] = np.arange(expected_order.shape[1])[None, :]
        np.testing.assert_array_equal(data["rank"], expected_rank)
        for reset_id in sorted(set(data["reset_id"].tolist())):
            reset_mask = data["reset_id"] == reset_id
            target = data["target_action"][reset_mask]
            np.testing.assert_array_equal(
                target,
                np.repeat(target[:1], target.shape[0], axis=0),
            )
        self.assertTrue(np.isfinite(data["post_probe_state"]).all())


if __name__ == "__main__":
    unittest.main()
