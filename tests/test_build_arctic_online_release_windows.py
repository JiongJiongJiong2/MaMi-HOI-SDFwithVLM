import unittest

import numpy as np

from scripts.build_arctic_online_release_windows import (
    confirmed_releases,
    raw_contact,
)


class BuildArcticOnlineReleaseWindowsTest(unittest.TestCase):
    def test_raw_contact(self):
        values = np.asarray([0.001, 0.004, np.nan])
        np.testing.assert_array_equal(
            raw_contact(values, 0.003),
            [True, False, False],
        )

    def test_confirmed_release(self):
        contact = np.asarray(
            [False] * 2
            + [True] * 20
            + [False] * 6
            + [True] * 20,
            dtype=bool,
        )
        releases = confirmed_releases(
            contact,
            minimum_contact_frames=15,
            confirmation_frames=5,
        )
        self.assertEqual(releases, [22])

    def test_recent_recontact_cancels_release(self):
        contact = np.asarray(
            [False] * 2
            + [True] * 20
            + [False, False, True, True]
            + [False] * 6
            + [True] * 20,
            dtype=bool,
        )
        releases = confirmed_releases(
            contact,
            minimum_contact_frames=15,
            confirmation_frames=5,
        )
        self.assertEqual(releases, [])


if __name__ == "__main__":
    unittest.main()
