import unittest

import numpy as np

from scripts.review_arctic_role_switch_events import (
    axis_angle_to_matrix,
    motion_norm,
    rotation_delta_degrees,
)


class ArcticRoleSwitchEventReviewTest(unittest.TestCase):
    def test_axis_angle_to_matrix_identity(self):
        np.testing.assert_allclose(
            axis_angle_to_matrix([0, 0, 0]),
            np.eye(3),
        )

    def test_rotation_delta_degrees(self):
        angle = rotation_delta_degrees(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, np.pi / 2.0],
        )
        self.assertAlmostEqual(angle, 90.0, places=5)

    def test_motion_norm(self):
        values = np.asarray([[0, 0, 0], [0.01, 0, 0]])
        self.assertAlmostEqual(motion_norm(values), 0.01)


if __name__ == "__main__":
    unittest.main()
