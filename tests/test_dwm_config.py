import unittest

from manip.world_model.dwm.config import (
    ACTION_DIM,
    OBJECT_CONFIGS,
    object_configs_for_split,
)
from manip.world_model.dwm.schema import (
    CONTACT_BODY_COUNT,
    CONTACT_MODE_LABELS,
    FINGER_JOINT_COUNT,
    STATE_DIM,
)


class DWMConfigTest(unittest.TestCase):
    def test_frozen_counts(self):
        self.assertEqual(ACTION_DIM, 51)
        self.assertEqual(FINGER_JOINT_COUNT, 45)
        self.assertEqual(CONTACT_BODY_COUNT, 16)
        self.assertEqual(STATE_DIM, 168)
        self.assertEqual(
            CONTACT_MODE_LABELS,
            ("none", "contact", "slip", "release"),
        )

    def test_object_splits_are_disjoint(self):
        splits = {
            split: {
                config.object_id
                for config in object_configs_for_split(split)
            }
            for split in ("train", "val", "test")
        }
        self.assertEqual(len(OBJECT_CONFIGS), 12)
        self.assertEqual(len(splits["train"]), 6)
        self.assertEqual(len(splits["val"]), 3)
        self.assertEqual(len(splits["test"]), 3)
        self.assertFalse(splits["train"] & splits["val"])
        self.assertFalse(splits["train"] & splits["test"])
        self.assertFalse(splits["val"] & splits["test"])


if __name__ == "__main__":
    unittest.main()
