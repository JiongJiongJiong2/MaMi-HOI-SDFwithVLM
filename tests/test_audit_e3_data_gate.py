import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.audit_e3_data_gate import (
    build_decision,
    inspect_handx_data,
    inspect_hopformer,
)


class E3DataGateTest(unittest.TestCase):
    def test_handx_sample_schema_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(
                root / "test_mano.npz",
                **{
                    "0": np.asarray({
                        "left_pose": np.zeros((2, 48)),
                        "right_pose": np.zeros((2, 48)),
                        "left_trans": np.zeros((2, 3)),
                        "right_trans": np.zeros((2, 3)),
                    }, dtype=object),
                },
            )
            result = inspect_handx_data(root)
            self.assertTrue(result["available"])
            self.assertTrue(result["has_bimanual_motion"])
            self.assertFalse(result["has_object_state"])

    def test_hopformer_dataset_presence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text(
                "EPIC-Contact is gated. 2.3K clips and 62.3K frames.",
                encoding="utf-8",
            )
            (root / "LICENSE").write_text(
                "CC BY-NC 4.0",
                encoding="utf-8",
            )
            (root / "src" / "datasets").mkdir(parents=True)
            (
                root / "src" / "datasets" / "epic_dataset.py"
            ).write_text(
                "dist.or dist.ro idx.or idx.ro",
                encoding="utf-8",
            )
            result = inspect_hopformer(root)
            self.assertEqual(result["license_family"], "CC-BY-NC-4.0")
            self.assertEqual(result["dataset_access"], "gated")
            self.assertFalse(result["epic_data_present"])

    def test_gate_remains_blocked_without_event_data(self):
        mamihoi = {
            "has_pose_hand": False,
            "has_object_pose": True,
        }
        handx = {"has_bimanual_motion": True}
        hopformer = {
            "epic_data_present": False,
            "arctic_data_present": False,
        }
        result = build_decision(mamihoi, handx, hopformer)
        self.assertEqual(result["raw_finger_supervision"], "pass")
        self.assertEqual(
            result["contact_memory_b_gate"],
            "blocked_missing_local_event_data",
        )
        self.assertEqual(
            result["future_handover_c_gate"],
            "blocked_missing_local_event_data",
        )


if __name__ == "__main__":
    unittest.main()
