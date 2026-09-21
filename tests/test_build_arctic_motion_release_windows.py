import unittest

import numpy as np

from scripts.build_arctic_motion_release_windows import (
    extract_motion_window,
    velocity,
)


class BuildArcticMotionReleaseWindowsTest(unittest.TestCase):
    def test_velocity_first_frame_zero(self):
        values = np.asarray([[1, 0, 0], [3, 0, 0]])
        result = velocity(values)
        np.testing.assert_allclose(result, [[0, 0, 0], [2, 0, 0]])

    def test_motion_window_excludes_release(self):
        values = np.arange(40 * 6, dtype=np.float64).reshape(40, 6)
        window = extract_motion_window(values, release=39, history=30)
        self.assertEqual(window.shape, (30, 6))
        np.testing.assert_allclose(window[-1], values[38])


if __name__ == "__main__":
    unittest.main()
