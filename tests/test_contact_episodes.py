import unittest

import numpy as np

from manip.world_model.contact_action.episodes import (
    aggregate_episode_metrics,
    contact_episode_metrics,
    contact_runs,
)


class ContactEpisodeMetricsTest(unittest.TestCase):
    def test_contact_runs_are_inclusive(self):
        self.assertEqual(
            contact_runs([0, 1, 1, 0, 1, 0]),
            [(1, 2), (4, 4)],
        )

    def test_stable_contact_success_and_dropout(self):
        truth = np.asarray([0, 1, 1, 1, 1, 1, 0, 0])
        prediction = np.asarray([2, 1, 1, 0, 1, 1, 0, 1])
        metrics = contact_episode_metrics(
            prediction,
            truth,
            stable_min_frames=2,
        )
        self.assertEqual(metrics["stable_contact_success"], 1)
        self.assertEqual(metrics["dropout_count"], 1)
        self.assertEqual(metrics["onset_delay_frames"], 0)
        self.assertEqual(metrics["early_contact_frames"], 1)
        self.assertEqual(metrics["late_contact_frames"], 1)
        self.assertEqual(metrics["release_delay_frames"], 2)

    def test_unstable_prediction_cannot_succeed(self):
        metrics = contact_episode_metrics(
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            stable_min_frames=2,
        )
        self.assertEqual(metrics["stable_contact_success"], 0)
        self.assertEqual(metrics["stable_contact_false_positive"], 1)

    def test_aggregate_ignores_undefined_values(self):
        aggregate = aggregate_episode_metrics([
            {"value": 1.0, "optional": None},
            {"value": 3.0, "optional": 2.0},
        ])
        self.assertEqual(aggregate["value"], 2.0)
        self.assertEqual(aggregate["optional"], 2.0)
        self.assertEqual(aggregate["optional_defined_count"], 1)


if __name__ == "__main__":
    unittest.main()

