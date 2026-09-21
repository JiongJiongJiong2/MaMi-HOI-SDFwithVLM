import unittest

import numpy as np

from scripts.build_arctic_role_switch_windows import extract_window


class BuildArcticRoleSwitchWindowsTest(unittest.TestCase):
    def test_window_excludes_release_frame(self):
        trajectory = {
            "right": np.asarray([0.02] * 40),
            "left": np.asarray([0.02] * 40),
        }
        trajectory["right"][39] = 0.0
        features = extract_window(trajectory, release=39, history=30)
        self.assertEqual(features.shape, (30, 10))
        self.assertEqual(features[-1, 6], 0.0)

    def test_window_padding_uses_earliest_frame(self):
        trajectory = {
            "right": np.asarray([0.001, 0.001, 0.001]),
            "left": np.asarray([0.020, 0.020, 0.020]),
        }
        features = extract_window(trajectory, release=2, history=5)
        np.testing.assert_allclose(features[0], features[1])


if __name__ == "__main__":
    unittest.main()
