import unittest

from scripts.build_epic_contact_strict_lifecycle import (
    assign_frame_splits,
    build_episodes,
    build_gate,
    participant_id,
)


class EpicContactStrictLifecycleTest(unittest.TestCase):
    def test_participant_id_uses_video_prefix(self):
        self.assertEqual(participant_id("P01_103"), "P01")
        self.assertEqual(participant_id("P30_05"), "P30")

    def test_build_strict_episode_boundaries(self):
        groups = {
            ("P01_01", "clip", "left", "cup"): [
                (0, 0.002),
                (1, 0.0005),
                (2, 0.0005),
                (3, 0.002),
            ]
        }
        episodes, transitions = build_episodes(groups, 0.001, 1)
        self.assertEqual(len(episodes), 1)
        self.assertTrue(episodes[0]["onset_observed"])
        self.assertTrue(episodes[0]["release_observed"])
        self.assertEqual(episodes[0]["contact_frame_count"], 2)
        self.assertEqual(transitions["P01"]["onset"], 1)
        self.assertEqual(transitions["P01"]["release"], 1)

    def test_frame_split_assignment_recomputes_contact(self):
        frames = [
            {
                "split": "train",
                "video_id": "P01_01",
                "frame": 0,
                "left": {
                    "valid": True,
                    "contact": True,
                    "min_distance_m": 0.0005,
                    "clip_id": "c",
                    "object_name": "cup",
                },
                "right": {
                    "valid": False,
                    "contact": False,
                    "min_distance_m": None,
                    "clip_id": None,
                    "object_name": None,
                },
            }
        ]
        result = assign_frame_splits(
            frames,
            {"P01": "dev"},
            0.001,
        )
        self.assertEqual(result[0]["split"], "dev")
        self.assertTrue(result[0]["left"]["contact"])

    def test_gate_accepts_balanced_strict_events(self):
        splits = {
            split: {
                "transition_counts": {
                    "hold": 10,
                    "onset": 150 if split == "train" else 30,
                    "release": 150 if split == "train" else 30,
                },
                "complete_episode_count": 30,
            }
            for split in ("train", "dev", "test")
        }
        frames = [
            {"split": split, "video_id": split}
            for split in ("train", "dev", "test")
        ]
        self.assertTrue(build_gate(splits, frames)["pass"])


if __name__ == "__main__":
    unittest.main()
