"""Tensor schema and conversion helpers for DWM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CONTACT_BODY_COUNT = 16
FINGER_JOINT_COUNT = 45
OBJECT_POSE_DIM = 9
OBJECT_TWIST_DIM = 6
WRIST_POSE_DIM = 9
WRIST_TWIST_DIM = 6
CONTACT_FORCE_DIM = CONTACT_BODY_COUNT * 3
STATE_DIM = (
    OBJECT_POSE_DIM
    + OBJECT_TWIST_DIM
    + WRIST_POSE_DIM
    + WRIST_TWIST_DIM
    + FINGER_JOINT_COUNT
    + FINGER_JOINT_COUNT
    + CONTACT_FORCE_DIM
)

CONTACT_MODE_LABELS = ("none", "contact", "slip", "release")
CONTACT_MODE_TO_INDEX = {
    label: index for index, label in enumerate(CONTACT_MODE_LABELS)
}


def rotation_matrix_to_6d(rotation):
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"rotation must end in (3, 3), got {rotation.shape}")
    return rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)


def quaternion_to_matrix(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape[-1] != 4:
        raise ValueError("quaternion must end in 4")
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    if np.any(norm <= 1e-12):
        raise ValueError("quaternion has zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    matrix = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - z * w)
    matrix[..., 0, 2] = 2.0 * (x * z + y * w)
    matrix[..., 1, 0] = 2.0 * (x * y + z * w)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - x * w)
    matrix[..., 2, 0] = 2.0 * (x * z - y * w)
    matrix[..., 2, 1] = 2.0 * (y * z + x * w)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


@dataclass(frozen=True)
class DWMStateV1:
    vector: np.ndarray
    object_pose: np.ndarray
    object_twist: np.ndarray
    wrist_pose: np.ndarray
    wrist_twist: np.ndarray
    finger_positions: np.ndarray
    finger_velocities: np.ndarray
    contact_forces: np.ndarray

    def __post_init__(self):
        if self.vector.shape != (STATE_DIM,):
            raise ValueError(
                f"state vector must have shape ({STATE_DIM},), "
                f"got {self.vector.shape}"
            )

    @property
    def contact_magnitudes(self):
        return np.linalg.norm(self.contact_forces, axis=-1)
