import unittest

import numpy as np

from scripts.build_oakink_handover_windows import (
    extract_window,
    select_primary_events,
)


class OakInkHandoverWindowsTest(unittest.TestCase):
    def test_select_primary_event(self):
        rows = [
            {
                "sequence": "s",
                "onset_minus_release": 10,
                "giver_release": 20,
            },
            {
                "sequence": "s",
                "onset_minus_release": 2,
                "giver_release": 30,
            },
        ]
        selected = select_primary_events(rows)
        self.assertEqual(selected["s"]["giver_release"], 30)

    def test_window_excludes_release(self):
        giver = np.full(40, 0.02)
        receiver = np.full(40, 0.02)
        giver[39] = 0.0
        window = extract_window(giver, receiver, frame=39, history=30)
        self.assertEqual(window.shape, (30, 10))
        self.assertEqual(window[-1, 6], 0.0)

    def test_window_padding(self):
        giver = np.asarray([0.001, 0.001, 0.001])
        receiver = np.asarray([0.020, 0.020, 0.020])
        window = extract_window(giver, receiver, frame=2, history=5)
        np.testing.assert_allclose(window[0], window[1])


if __name__ == "__main__":
    unittest.main()
