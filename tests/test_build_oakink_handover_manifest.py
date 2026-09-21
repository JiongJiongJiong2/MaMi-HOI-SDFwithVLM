import unittest

import numpy as np

from scripts.build_oakink_handover_manifest import (
    parse_ply_vertices,
    parse_sample_name,
    parse_sequence,
    role_switch_event,
    split_for_subjects,
    write_trajectories,
)


class OakInkHandoverManifestTest(unittest.TestCase):
    def test_parse_sequence(self):
        row = parse_sequence("A15015_0004_0008_0007")
        self.assertEqual(row["object_id"], "A15015")
        self.assertEqual(row["giver"], "0008")
        self.assertEqual(row["receiver"], "0007")

    def test_split_for_subjects(self):
        self.assertEqual(split_for_subjects("0001", "0003"), "test")
        self.assertIsNone(split_for_subjects("0001", "0006"))

    def test_parse_sample_name(self):
        row = parse_sample_name(
            "anno/hand_v/A15015_0004_0008_0007__"
            "2021-10-03-14-45-01__0__106__0.pkl"
        )
        self.assertEqual(row["sequence"], "A15015_0004_0008_0007")
        self.assertEqual(row["frame"], 106)
        self.assertEqual(row["subject_flag"], "0")

    def test_role_switch_event(self):
        giver = np.zeros(80, dtype=bool)
        receiver = np.zeros(80, dtype=bool)
        giver[:20] = True
        receiver[25:60] = True
        _, _, candidates = role_switch_event(
            giver,
            receiver,
            minimum_frames=15,
            window_frames=30,
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["onset_minus_release"], 5)

    def test_parse_binary_ply_vertices(self):
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 2\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        ).encode("ascii")
        payload = np.asarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            dtype="<f4",
        ).tobytes()
        vertices = parse_ply_vertices(header + payload)
        np.testing.assert_allclose(
            vertices,
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        )

    def test_write_trajectories_is_portable_and_padded(self):
        import tempfile

        records = [
            {
                "sequence": "s1",
                "split": "train",
                "participant_pair": "a->b",
                "object_id": "O1",
                "object_name": "obj",
                "giver_distance_m": np.asarray([0.1, 0.2]),
                "receiver_distance_m": np.asarray([0.3, 0.4]),
            },
            {
                "sequence": "s2",
                "split": "val",
                "participant_pair": "c->d",
                "object_id": "O2",
                "object_name": "obj2",
                "giver_distance_m": np.asarray([0.5]),
                "receiver_distance_m": np.asarray([0.6]),
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/trajectories.npz"
            write_trajectories(path, records)
            arrays = np.load(path, allow_pickle=False)
            try:
                self.assertEqual(arrays["giver_distance_m"].shape, (2, 2))
                self.assertTrue(np.isnan(arrays["giver_distance_m"][1, 1]))
                np.testing.assert_array_equal(arrays["lengths"], [2, 1])
            finally:
                arrays.close()


if __name__ == "__main__":
    unittest.main()
