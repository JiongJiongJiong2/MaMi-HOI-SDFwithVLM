import unittest

import numpy as np

from scripts.project_handx_to_mano import (
    SKELETON_CHAIN,
    bone_length_cv,
    extract_generated_motion,
    initialize_world_transform,
    project_skeleton_to_template,
    rigid_align,
    smooth_sequence,
    trajectory_stat,
)


class ProjectHandXToManoTest(unittest.TestCase):
    def test_extract_generated_motion_accepts_batch_dimension(self):
        value = np.zeros((1, 7, 42, 4))
        motion = extract_generated_motion(value)
        self.assertEqual(motion.shape, (7, 42, 3))

    def test_extract_generated_motion_accepts_flat_feature_dimension(self):
        value = np.zeros((1, 7, 42 * 4))
        motion = extract_generated_motion(value)
        self.assertEqual(motion.shape, (7, 42, 3))

    def test_bone_length_cv_is_zero_for_scaled_rigid_hand(self):
        base = np.zeros((3, 21, 3), dtype=np.float64)
        for chain_index, chain in enumerate(SKELETON_CHAIN):
            for depth, joint in enumerate(chain[1:], start=1):
                base[:, joint, 0] = depth * (0.02 + 0.001 * chain_index)
        base[:, :, 1] = np.asarray([0.0, 0.01, 0.02])[:, None]
        metrics = bone_length_cv(base)
        self.assertLess(metrics["bone_cv_max"], 1e-12)

    def test_rigid_align_recovers_known_transform(self):
        source = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        rotation = np.asarray(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        translation = np.asarray([0.2, -0.1, 0.3])
        target = source @ rotation.T + translation
        estimated_rotation, estimated_translation = rigid_align(
            source, target
        )
        np.testing.assert_allclose(
            estimated_rotation, rotation, atol=1e-12
        )
        np.testing.assert_allclose(
            estimated_translation, translation, atol=1e-12
        )

    def test_trajectory_stats_scale_to_millimeters(self):
        joints = np.zeros((4, 21, 3), dtype=np.float64)
        joints[:, 0, 0] = np.arange(4) * 0.001
        stats = trajectory_stat(joints)
        self.assertAlmostEqual(stats["speed_mm"], 1.0 / 21.0)
        self.assertAlmostEqual(stats["acceleration_mm"], 0.0)
        self.assertAlmostEqual(stats["jerk_mm"], 0.0)

    def test_initial_world_transform_tracks_wrist_exactly(self):
        neutral = np.asarray(
            [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.1, 0.1, 0.0]]
        )
        targets = np.asarray(
            [
                [[0.3, 0.2, 0.1], [0.4, 0.2, 0.1], [0.3, 0.3, 0.1]],
                [[0.0, 0.5, 0.2], [0.0, 0.6, 0.2], [0.1, 0.5, 0.2]],
            ]
        )
        rotations, translations = initialize_world_transform(
            neutral, targets
        )
        transformed_wrists = (
            np.einsum("bij,j->bi", rotations, neutral[0]) + translations
        )
        np.testing.assert_allclose(
            transformed_wrists, targets[:, 0], atol=1e-6
        )

    def test_skeleton_projection_preserves_template_bone_lengths(self):
        template = np.zeros((21, 3), dtype=np.float64)
        for chain in SKELETON_CHAIN:
            for parent, joint in zip(chain[:-1], chain[1:]):
                template[joint] = template[parent]
                template[joint, 0] += 0.01 + joint * 0.0001
        target = np.zeros((2, 21, 3), dtype=np.float64)
        for joint in range(1, 21):
            target[:, joint, 1] = joint * 0.1
        projected = project_skeleton_to_template(target, template)
        for chain in SKELETON_CHAIN:
            projected_lengths = np.linalg.norm(
                projected[:, chain[1:]] - projected[:, chain[:-1]],
                axis=2,
            )
            template_lengths = np.linalg.norm(
                template[chain[1:]] - template[chain[:-1]],
                axis=1,
            )
            np.testing.assert_allclose(
                projected_lengths,
                np.broadcast_to(template_lengths, projected_lengths.shape),
                atol=1e-12,
            )

    def test_smooth_sequence_reduces_single_frame_jitter(self):
        values = np.zeros((5, 2), dtype=np.float64)
        values[2] = 1.0
        smoothed = smooth_sequence(values, 3)
        self.assertLess(smoothed[2, 0], values[2, 0])
        self.assertGreater(smoothed[1, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
