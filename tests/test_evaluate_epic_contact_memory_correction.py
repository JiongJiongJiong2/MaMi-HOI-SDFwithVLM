import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_epic_contact_memory_correction import (
    cluster_bootstrap_mean,
    gaussian_kernel,
    lowpass_noise,
    normalize_noise,
    parse_perturbations,
    perturb_sequence,
)


class EpicContactMemoryCorrectionTest(unittest.TestCase):
    def test_parse_perturbations(self):
        self.assertEqual(
            parse_perturbations("2mm,5mm,0.01m"),
            (0.002, 0.005, 0.01),
        )

    def test_gaussian_kernel_is_normalized(self):
        kernel = gaussian_kernel(1.0)
        self.assertAlmostEqual(float(kernel.sum()), 1.0)

    def test_lowpass_noise_is_deterministic(self):
        first = lowpass_noise(
            8,
            2,
            sigma=1.0,
            rng=np.random.default_rng(1),
        )
        second = lowpass_noise(
            8,
            2,
            sigma=1.0,
            rng=np.random.default_rng(1),
        )
        np.testing.assert_array_equal(first, second)
        normalized = normalize_noise(first)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(normalized ** 2))),
            1.0,
            places=6,
        )

    def test_perturb_sequence_respects_amplitude(self):
        arrays = {
            "mano_pose": np.zeros((6, 48), dtype=np.float32),
            "mano_translation": np.zeros((6, 3), dtype=np.float32),
            "object_diameter_m": np.asarray(0.1, dtype=np.float32),
        }
        result = perturb_sequence(arrays, 0.005, seed=3)
        translation_rms = float(np.sqrt(
            np.mean(result["translation"] ** 2)
        ))
        self.assertAlmostEqual(translation_rms, 0.005, places=6)

    def test_cluster_bootstrap_mean(self):
        interval = cluster_bootstrap_mean(
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            ["a", "a", "b", "b"],
            100,
            seed=1,
        )
        self.assertEqual(len(interval), 2)
        self.assertLessEqual(interval[0], interval[1])


if __name__ == "__main__":
    unittest.main()
