import unittest

from scripts.analyze_epic_contact_lifecycle import (
    bimanual_summary,
    build_decision,
    contiguous_runs,
    transition_summary,
    threshold_sensitivity,
)


class EpicContactLifecycleTest(unittest.TestCase):
    def test_contiguous_runs_splits_on_gap(self):
        self.assertEqual(
            contiguous_runs([1, 2, 4, 5, 7], 1),
            [[1, 2], [4, 5], [7]],
        )

    def test_transition_counts(self):
        frames = [
            self.frame(0, True, False),
            self.frame(1, True, False),
            self.frame(2, True, False, left_contact=False),
            self.frame(3, True, False, left_contact=False),
        ]
        result = transition_summary(frames, 1)
        self.assertEqual(result["train"]["contact_to_contact"], 1)
        self.assertEqual(result["train"]["contact_to_noncontact"], 1)
        self.assertEqual(result["train"]["noncontact_to_noncontact"], 1)

    def test_bimanual_groups(self):
        frames = [
            self.frame(0, True, True, both=True),
            self.frame(1, True, True, both=True),
            self.frame(3, True, True, both=True),
        ]
        result = bimanual_summary(frames, 1)
        self.assertEqual(result["train"]["group_count"], 1)
        self.assertEqual(result["train"]["run_count"], 2)

    def test_decision_requires_release_examples(self):
        episodes = {"all": {"complete_count": 10}}
        transitions = {
            "train": {
                "contact_to_contact": 2000,
                "noncontact_to_contact": 200,
                "contact_to_noncontact": 10,
            }
        }
        bimanual = {
            "train": {"group_count": 30, "frame_count": 2000}
        }
        split = {"video_disjoint": True}
        sensitivity = {
            "0.001000": {
                "train": {
                    "hold": 2000,
                    "onset": 10,
                    "release": 10,
                    "complete_episode": 5,
                },
                "test": {"complete_episode": 1},
            }
        }
        result = build_decision(
            episodes,
            transitions,
            bimanual,
            split,
            sensitivity,
        )
        self.assertFalse(result["release_model_labels_ready"])
        self.assertEqual(
            result["recommended_action"],
            "limit_to_hold_or_bimanual_analysis",
        )

    def test_threshold_sensitivity_counts_episode_boundaries(self):
        frames = [
            self.frame(0, True, False, left_contact=False),
            self.frame(1, True, False, left_contact=True),
            self.frame(2, True, False, left_contact=True),
            self.frame(3, True, False, left_contact=False),
        ]
        for row in frames:
            row["left"]["min_distance_m"] = (
                0.0005
                if row["left"]["contact"]
                else 0.002
            )
        result = threshold_sensitivity(
            frames,
            (0.001, 0.003),
            1,
        )
        self.assertEqual(
            result["0.001000"]["train"]["complete_episode"],
            1,
        )
        self.assertEqual(
            result["0.003000"]["train"]["complete_episode"],
            0,
        )

    @staticmethod
    def frame(
        frame,
        left_valid,
        right_valid,
        left_contact=None,
        right_contact=None,
        both=False,
    ):
        left_contact = left_valid if left_contact is None else left_contact
        right_contact = (
            right_valid if right_contact is None else right_contact
        )
        return {
            "split": "train",
            "video_id": "v",
            "frame": frame,
            "same_object_bimanual": both,
            "same_object_both_contact": both,
            "left": {
                "valid": left_valid,
                "contact": left_contact,
                "clip_id": "c" if left_valid else None,
                "object_name": "cup",
            },
            "right": {
                "valid": right_valid,
                "contact": right_contact,
                "clip_id": "c" if right_valid else None,
                "object_name": "cup",
            },
        }


if __name__ == "__main__":
    unittest.main()
