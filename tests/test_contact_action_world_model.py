"""Tests for contact-action state construction and short-horizon model."""

import unittest

import numpy as np
import torch

from manip.world_model.contact_action.features import (
    ACTION_DIM,
    CONTACT_SLICE,
    STATE_DIM,
    build_window_features,
    make_training_samples,
    select_contact_event_anchors,
)
from manip.world_model.contact_action.model import ContactActionTransition
from scripts.evaluate_contact_action_candidate_ranking import (
    make_candidate_actions,
)
from scripts.select_mami_candidates_with_world_model import (
    proxy_contact_metrics,
)
from scripts.evaluate_contact_protocol_v0_lite import (
    hand_metrics,
    pool_counts,
)
from scripts.evaluate_contact_protocol_v0_dense import (
    contact_counts_from_min_distances,
)
from scripts.run_contact_action_chunk_correction import (
    candidate_selection_score,
    make_action_candidates,
)


class ContactActionFeatureTests(unittest.TestCase):
    def test_window_shapes_and_sample_alignment(self):
        length = 20
        motion = np.zeros((length, 204), dtype=np.float32)
        contact = np.zeros((length, 4), dtype=np.float32)
        contact[5:10, 0] = 1.0
        contact[10:15, 1] = 1.0
        object_pos = np.zeros((length, 3), dtype=np.float32)
        object_pos[:, 0] = np.linspace(0.0, 0.2, length)
        identity = np.eye(3, dtype=np.float32)
        object_rot = np.repeat(identity[None], length, axis=0)

        features = build_window_features(
            motion=motion,
            contact_labels=contact,
            object_pos=object_pos,
            object_rot=object_rot,
            jpos_minimum=np.full(72, -1.0, dtype=np.float32),
            jpos_maximum=np.ones(72, dtype=np.float32),
            object_scale=2.0,
        )
        self.assertEqual(features.states.shape, (length, STATE_DIM))
        self.assertEqual(features.actions.shape, (length - 1, ACTION_DIM))
        self.assertTrue(np.isfinite(features.states).all())
        self.assertTrue(np.isfinite(features.actions).all())

        samples = make_training_samples(
            features.states,
            features.actions,
            history=3,
            horizon=4,
            stride=2,
            max_samples=4,
        )
        self.assertEqual(samples["state_history"].shape, (4, 3, STATE_DIM))
        self.assertEqual(samples["action_history"].shape, (4, 3, ACTION_DIM))
        self.assertEqual(samples["future_actions"].shape, (4, 4, ACTION_DIM))
        self.assertEqual(samples["future_states"].shape, (4, 4, STATE_DIM))

    def test_event_anchors_cover_transition_and_background(self):
        states = np.zeros((20, STATE_DIM), dtype=np.float32)
        states[8:, CONTACT_SLICE.start] = 1.0
        anchors = select_contact_event_anchors(
            states,
            history=2,
            horizon=4,
            event_samples=4,
            background_samples=2,
        )
        self.assertIsNotNone(anchors)
        self.assertTrue(set((3, 4, 5, 6)).issubset(set(anchors.tolist())))
        self.assertEqual(len(anchors), 6)


class ContactActionModelTests(unittest.TestCase):
    def test_rollout_shapes_and_action_sensitivity(self):
        torch.manual_seed(2)
        model = ContactActionTransition(hidden_size=32)
        state_history = torch.randn(2, 3, STATE_DIM)
        state_history[..., CONTACT_SLICE] = torch.rand(2, 3, 2)
        action_history = torch.randn(2, 3, ACTION_DIM)
        future_actions = torch.randn(2, 4, ACTION_DIM)
        teacher_states = torch.randn(2, 4, STATE_DIM)
        teacher_states[..., CONTACT_SLICE] = torch.rand(2, 4, 2)

        teacher_output = model.rollout(
            state_history,
            action_history,
            future_actions,
            teacher_states=teacher_states,
            teacher_forcing_ratio=1.0,
        )
        free_output = model.rollout(
            state_history,
            action_history,
            future_actions,
            teacher_states=None,
            teacher_forcing_ratio=0.0,
        )
        zero_output = model.rollout(
            state_history,
            action_history,
            torch.zeros_like(future_actions),
            teacher_states=None,
            teacher_forcing_ratio=0.0,
        )

        self.assertEqual(teacher_output["states"].shape, (2, 4, STATE_DIM))
        self.assertEqual(
            teacher_output["contact_probability"].shape,
            (2, 4, 2),
        )
        self.assertEqual(teacher_output["contact_logits"].shape, (2, 4, 2))
        self.assertEqual(teacher_output["residuals"].shape, (2, 4, 32))
        self.assertTrue(torch.isfinite(free_output["states"]).all())
        self.assertTrue(
            (
                teacher_output["contact_probability"] >= 0.0
            ).all()
        )
        self.assertTrue(
            (
                teacher_output["contact_probability"] <= 1.0
            ).all()
        )
        self.assertFalse(
            torch.allclose(
                free_output["states"],
                zero_output["states"],
            )
        )

    def test_learned_residual_switch_and_palm_mask(self):
        torch.manual_seed(3)
        model = ContactActionTransition(
            hidden_size=32,
            residual_scale=0.05,
            residual_mask="palm",
        )
        with torch.no_grad():
            model.head[-1].weight.normal_(0.0, 0.5)
            model.head[-1].bias.normal_(0.0, 0.5)
        state_history = torch.zeros(1, 3, STATE_DIM)
        state_history[..., 26] = 1.0
        action_history = torch.zeros(1, 3, ACTION_DIM)
        future_actions = torch.zeros(1, 4, ACTION_DIM)
        learned = model.rollout(
            state_history,
            action_history,
            future_actions,
            teacher_states=None,
            teacher_forcing_ratio=0.0,
            use_learned_residual=True,
        )
        analytic = model.rollout(
            state_history,
            action_history,
            future_actions,
            teacher_states=None,
            teacher_forcing_ratio=0.0,
            use_learned_residual=False,
        )
        self.assertTrue(torch.equal(
            learned["residuals"][..., 0:9],
            torch.zeros_like(learned["residuals"][..., 0:9]),
        ))
        self.assertTrue(torch.equal(
            learned["residuals"][..., 21:24],
            torch.zeros_like(learned["residuals"][..., 21:24]),
        ))
        self.assertFalse(torch.allclose(
            learned["states"],
            analytic["states"],
        ))
        self.assertTrue(torch.isfinite(learned["states"]).all())

    def test_zero_rotation_teacher_state_has_finite_gradients(self):
        torch.manual_seed(4)
        model = ContactActionTransition(
            hidden_size=16,
            residual_scale=0.05,
            residual_mask="palm",
        )
        state_history = torch.zeros(2, 3, STATE_DIM)
        action_history = torch.zeros(2, 3, ACTION_DIM)
        future_actions = torch.zeros(2, 4, ACTION_DIM)
        teacher_states = torch.zeros(2, 4, STATE_DIM)
        teacher_states[:, 0, 32] = 1.0
        output = model.rollout(
            state_history,
            action_history,
            future_actions,
            teacher_states=teacher_states,
            teacher_forcing_ratio=0.5,
        )
        output["contact_logits"].sum().backward()
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_candidate_actions_move_along_surface_normal(self):
        split = {
            "state_history": torch.zeros(1, 4, STATE_DIM),
            "action_history": torch.zeros(1, 4, ACTION_DIM),
            "future_actions": torch.zeros(1, 3, ACTION_DIM),
        }
        split["state_history"][0, -1, 26] = 1.0
        candidates = make_candidate_actions(split, alpha=0.1, device="cpu")
        self.assertAlmostEqual(
            candidates["left_inward"][0, 0, 0].item(),
            -0.1,
            places=6,
        )
        self.assertAlmostEqual(
            candidates["left_outward"][0, 0, 0].item(),
            0.1,
            places=6,
        )
        self.assertTrue(torch.equal(
            candidates["left_inward"][:, :, 6:],
            split["future_actions"][:, :, 6:],
        ))
        self.assertTrue(torch.equal(
            candidates["hold"],
            split["future_actions"],
        ))

    def test_proxy_contact_metrics(self):
        states = np.full((4, STATE_DIM), 0.1, dtype=np.float32)
        states[:, 24:26] = 0.01
        truth = np.ones((4, 2), dtype=np.float32)
        metrics = proxy_contact_metrics(
            states,
            truth,
            contact_threshold_norm=0.02,
        )
        self.assertEqual(metrics["contact_f1"], 1.0)
        self.assertEqual(metrics["true_positive"], 8)

    def test_v0_lite_pooled_contact_counts(self):
        raw = {
            "left_precision": 1.0,
            "left_recall": 0.5,
            "left_f1": 2.0 / 3.0,
            "left_tp": 1,
            "left_fp": 0,
            "left_fn": 1,
            "left_pred_contact_frame_count": 1,
            "left_gt_contact_frame_count": 2,
            "right_precision": 0.5,
            "right_recall": 1.0,
            "right_f1": 2.0 / 3.0,
            "right_tp": 1,
            "right_fp": 1,
            "right_fn": 0,
            "right_pred_contact_frame_count": 2,
            "right_gt_contact_frame_count": 1,
        }
        rows = [
            {hand: hand_metrics(raw, hand) for hand in ("left", "right")},
            {hand: hand_metrics(raw, hand) for hand in ("left", "right")},
        ]
        pooled = pool_counts(rows)
        self.assertEqual(pooled["left"]["tp"], 2)
        self.assertEqual(pooled["left"]["fn"], 2)
        self.assertAlmostEqual(pooled["left"]["precision"], 1.0)
        self.assertAlmostEqual(pooled["left"]["recall"], 0.5)

    def test_dense_contact_counts_from_distances(self):
        metrics = contact_counts_from_min_distances(
            np.asarray([0.01, 0.06, 0.03, 0.07], dtype=np.float32),
            np.asarray([1, 1, 0, 0], dtype=np.float32),
            threshold_m=0.05,
        )
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)

    def test_action_chunk_candidates_only_change_target_hand(self):
        base = np.zeros((4, ACTION_DIM), dtype=np.float32)
        normal = np.ones((2, 3), dtype=np.float32)
        candidates = dict(make_action_candidates(
            base,
            normal,
            hand_index=1,
            alpha=0.1,
            num_random=0,
            rng=np.random.default_rng(1),
        ))
        self.assertTrue(np.allclose(candidates["base"], base))
        self.assertTrue(np.allclose(
            candidates["inward"][:, 0:3],
            base[:, 0:3],
        ))
        self.assertTrue(np.allclose(
            candidates["inward"][:, 3:6],
            -0.1,
        ))
        self.assertTrue(np.allclose(
            candidates["inward"][:, 6:],
            base[:, 6:],
        ))

    def test_scorer_modes_only_change_selection_term(self):
        contact = np.asarray([0.5, 0.4], dtype=np.float32)
        penetration = np.asarray([0.01, 0.0], dtype=np.float32)
        event = np.asarray([0.1, 0.9], dtype=np.float32)
        geom = candidate_selection_score(
            "geom_only",
            contact,
            penetration,
            event,
            penetration_weight=20.0,
            event_weight=0.25,
        )
        learned = candidate_selection_score(
            "learned_state",
            contact,
            penetration,
            event,
            penetration_weight=20.0,
            event_weight=0.25,
        )
        hybrid = candidate_selection_score(
            "hybrid_event",
            contact,
            penetration,
            event,
            penetration_weight=20.0,
            event_weight=0.25,
        )
        self.assertTrue(np.array_equal(geom, learned))
        self.assertTrue(np.allclose(hybrid, geom + 0.25 * event))
        self.assertEqual(int(geom.argmax()), 1)
        self.assertEqual(int(hybrid.argmax()), 1)


if __name__ == "__main__":
    unittest.main()
