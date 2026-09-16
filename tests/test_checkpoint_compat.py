import unittest

import torch

from manip.model.checkpoint_compat import (
    FINETUNE_SDF_ONLY_MODEL_KEYS,
    build_compatible_state_dict,
    derive_ema_allowed_missing_keys,
)


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_only_whitelisted_missing_keys_are_filled(self):
        current = {
            key: torch.full((2, 2), float(index + 1))
            for index, key in enumerate(FINETUNE_SDF_ONLY_MODEL_KEYS)
        }
        current["shared.weight"] = torch.ones(1)
        checkpoint = {"shared.weight": torch.ones(1)}

        compatible, report = build_compatible_state_dict(
            current,
            checkpoint,
            FINETUNE_SDF_ONLY_MODEL_KEYS,
            context="model",
        )

        self.assertEqual(set(compatible), set(current))
        for key in FINETUNE_SDF_ONLY_MODEL_KEYS:
            self.assertTrue(torch.equal(compatible[key], current[key]))
        self.assertEqual(
            report["initialized_missing_keys"],
            sorted(FINETUNE_SDF_ONLY_MODEL_KEYS),
        )

    def test_disallowed_missing_key_fails(self):
        current = {
            "shared.weight": torch.ones(1),
            "new.weight": torch.ones(1),
        }
        checkpoint = {"shared.weight": torch.ones(1)}
        with self.assertRaisesRegex(RuntimeError, "missing keys"):
            build_compatible_state_dict(current, checkpoint)

    def test_unexpected_key_fails(self):
        current = {"shared.weight": torch.ones(1)}
        checkpoint = {
            "shared.weight": torch.ones(1),
            "removed.weight": torch.ones(1),
        }
        with self.assertRaisesRegex(RuntimeError, "unexpected keys"):
            build_compatible_state_dict(current, checkpoint)

    def test_shape_and_dtype_mismatch_fails(self):
        current = {"shared.weight": torch.ones(2, dtype=torch.float32)}
        shape_checkpoint = {"shared.weight": torch.ones(3, dtype=torch.float32)}
        with self.assertRaisesRegex(RuntimeError, "shape/dtype mismatches"):
            build_compatible_state_dict(current, shape_checkpoint)

        dtype_checkpoint = {"shared.weight": torch.ones(2, dtype=torch.float64)}
        with self.assertRaisesRegex(RuntimeError, "shape/dtype mismatches"):
            build_compatible_state_dict(current, dtype_checkpoint)

    def test_ema_whitelist_uses_present_prefixes(self):
        allowed_model_keys = FINETUNE_SDF_ONLY_MODEL_KEYS[:2]
        ema_state = {
            "online_model." + key: torch.ones(1)
            for key in allowed_model_keys
        }
        ema_state.update({
            "ema_model." + key: torch.ones(1)
            for key in allowed_model_keys
        })
        allowed = derive_ema_allowed_missing_keys(
            ema_state,
            allowed_model_keys,
        )
        self.assertEqual(len(allowed), 4)
        self.assertIn("online_model." + allowed_model_keys[0], allowed)
        self.assertIn("ema_model." + allowed_model_keys[1], allowed)


if __name__ == "__main__":
    unittest.main()
