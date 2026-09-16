"""Feature construction for the contact-action dynamics pilot.

We intentionally use only information already present in MaMi windows:

* object pose and motion
* de-normalized SMPL-X joint positions
* saved per-hand contact labels

The first version does not require tactile data or articulated finger poses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from manip.model.sdf_utils import (
    compute_sdf_gradients_fd,
    sample_object_sdf_at_points,
)


STATE_DIM = 34
ACTION_DIM = 15
NONCONTACT_DIM = 32
CONTACT_SLICE = slice(32, 34)
CLEARANCE_SLICE = slice(24, 26)
NORMAL_SLICE = slice(26, 32)


@dataclass(frozen=True)
class WindowFeatures:
    states: np.ndarray
    actions: np.ndarray


def _as_float_array(value, ndim, name):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got {array.shape}")
    return array


def _rotation_to_6d(rotation):
    """Match PyTorch3D's first-two-rows rotation-6D encoding."""
    return np.asarray(rotation, dtype=np.float32)[..., :2, :].reshape(
        *rotation.shape[:-2], 6
    )


def _transform_world_to_object(points_world, object_rot, object_pos):
    """Apply row-vector convention: p_obj = (p_world - t) @ R."""
    points_world = np.asarray(points_world, dtype=np.float32)
    object_rot = np.asarray(object_rot, dtype=np.float32)
    object_pos = np.asarray(object_pos, dtype=np.float32)
    while object_rot.ndim < points_world.ndim + 1:
        object_rot = np.expand_dims(object_rot, axis=-3)
    while object_pos.ndim < points_world.ndim:
        object_pos = np.expand_dims(object_pos, axis=-2)
    relative = points_world - object_pos
    return np.matmul(relative[..., None, :], object_rot).squeeze(-2)


def _safe_rotation_delta(next_rot, current_rot):
    return np.matmul(next_rot, np.swapaxes(current_rot, -1, -2))


def _sample_palm_geometry(
    palm_object_metric,
    object_scale,
    object_sdf,
):
    """Query proxy signed clearance and surface normals at palm joints."""
    grid, centroid, extents = object_sdf
    points = torch.tensor(
        palm_object_metric.reshape(-1, 3),
        dtype=torch.float32,
    )[None]
    with torch.no_grad():
        signed_distance = sample_object_sdf_at_points(
            grid,
            points,
            centroid,
            extents,
        )
        normal = compute_sdf_gradients_fd(
            grid,
            points,
            centroid,
            extents.max(dim=-1, keepdim=True).values,
        )
        normal = F.normalize(normal, dim=-1, eps=1e-6)
    signed_distance = (
        signed_distance.squeeze(0).squeeze(-1).numpy()
        / float(object_scale)
    )
    normal = normal.squeeze(0).numpy().reshape(palm_object_metric.shape)
    return signed_distance.reshape(-1, 2), normal.reshape(
        palm_object_metric.shape[0], 6
    )


def build_window_features(
    motion,
    contact_labels,
    object_pos,
    object_rot,
    jpos_minimum,
    jpos_maximum,
    object_scale,
    object_sdf=None,
):
    """Convert one MaMi window into state/action trajectories.

    State layout (34):
      0:3    object translation relative to anchor, anchor-local, /scale
      3:9    object rotation relative to anchor, 6D
      9:15   left/right palm in object frame, /scale
      15:21  left/right palm local velocity, /scale
      21:24  object local translation velocity, /scale
      24:26  left/right palm proxy signed clearance, /scale
      26:32  left/right palm proxy surface normal
      32:34  left/right contact probability

    Action layout (15):
      0:6    left/right palm delta in current object frame, /scale
      6:9    object translation delta in current object frame, /scale
      9:15   object rotation delta, 6D
    """
    motion = _as_float_array(motion, 2, "motion")
    contact_labels = _as_float_array(contact_labels, 2, "contact_labels")
    object_pos = _as_float_array(object_pos, 2, "object_pos")
    object_rot = _as_float_array(object_rot, 3, "object_rot")

    length = motion.shape[0]
    if contact_labels.shape[0] != length:
        raise ValueError("contact labels and motion must have equal length")
    if object_pos.shape != (length, 3):
        raise ValueError(
            f"object_pos must be [{length}, 3], got {object_pos.shape}"
        )
    if object_rot.shape != (length, 3, 3):
        raise ValueError(
            f"object_rot must be [{length}, 3, 3], got {object_rot.shape}"
        )
    if contact_labels.shape[1] < 2:
        raise ValueError("contact labels must contain left/right hand channels")

    scale = float(object_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"object_scale must be positive, got {object_scale}")

    # Processed windows store the first 72 channels as raw global joint
    # positions. The dataset layer normalizes them separately for the
    # diffusion model, so applying de-normalization here corrupts geometry.
    global_jpos = motion[:, : 24 * 3].reshape(length, 24, 3)
    if not np.isfinite(global_jpos).all():
        raise ValueError("motion contains non-finite global joint positions")
    if not np.isfinite(jpos_minimum).all() or not np.isfinite(jpos_maximum).all():
        raise ValueError("joint normalization statistics are non-finite")
    palm_world = global_jpos[:, [22, 23], :]

    anchor_rot = object_rot[0:1]
    object_pos_rel = np.einsum(
        "tj,ji->ti",
        object_pos - object_pos[0:1],
        anchor_rot[0],
    )
    object_pos_rel = object_pos_rel / scale
    object_rot_rel = np.matmul(
        object_rot,
        np.swapaxes(anchor_rot[0], -1, -2),
    )
    object_rot_6d = _rotation_to_6d(object_rot_rel)
    palm_object = _transform_world_to_object(
        palm_world,
        object_rot[:, None],
        object_pos[:, None],
    )
    palm_object_metric = palm_object
    palm_object = palm_object_metric / scale
    if object_sdf is None:
        signed_clearance = np.zeros((length, 2), dtype=np.float32)
        surface_normal = np.zeros((length, 6), dtype=np.float32)
    else:
        signed_clearance, surface_normal = _sample_palm_geometry(
            palm_object_metric,
            scale,
            object_sdf,
        )

    palm_velocity = np.zeros_like(palm_object)
    object_velocity = np.zeros_like(object_pos)
    if length > 1:
        palm_world_delta = palm_world[1:] - palm_world[:-1]
        palm_velocity[1:] = _transform_world_to_object(
            palm_world_delta,
            object_rot[1:, None],
            np.zeros_like(object_pos[1:, None]),
        ) / scale
        object_velocity[1:] = np.einsum(
            "tj,tji->ti",
            object_pos[1:] - object_pos[:-1],
            object_rot[1:],
        ) / scale

    hand_contact = np.clip(contact_labels[:, :2], 0.0, 1.0)
    states = np.concatenate(
        [
            object_pos_rel,
            object_rot_6d,
            palm_object.reshape(length, 6),
            palm_velocity.reshape(length, 6),
            object_velocity,
            signed_clearance,
            surface_normal,
            hand_contact,
        ],
        axis=1,
    ).astype(np.float32)

    palm_delta_world = palm_world[1:] - palm_world[:-1]
    palm_action = _transform_world_to_object(
        palm_delta_world,
        object_rot[:-1, None],
        np.zeros_like(object_pos[:-1, None]),
    ) / scale
    object_translation_action = np.einsum(
        "tj,tji->ti",
        object_pos[1:] - object_pos[:-1],
        object_rot[:-1],
    ) / scale
    object_rotation_action = _rotation_to_6d(
        _safe_rotation_delta(object_rot[1:], object_rot[:-1])
    )
    actions = np.concatenate(
        [
            palm_action.reshape(length - 1, 6),
            object_translation_action,
            object_rotation_action,
        ],
        axis=1,
    ).astype(np.float32)

    return WindowFeatures(states=states, actions=actions)


def make_training_samples(
    states,
    actions,
    history,
    horizon,
    stride,
    max_samples,
    anchor_indices=None,
):
    """Create aligned history/action/future windows.

    When ``anchor_indices`` is provided, those current-state indices are used
    directly. Otherwise anchors are spread across the full sequence instead of
    keeping only the first few candidates when ``max_samples`` is small.
    """
    states = _as_float_array(states, 2, "states")
    actions = _as_float_array(actions, 2, "actions")
    if states.shape[1] != STATE_DIM:
        raise ValueError(f"states must have width {STATE_DIM}")
    if actions.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have width {ACTION_DIM}")
    if states.shape[0] != actions.shape[0] + 1:
        raise ValueError("states must contain exactly one more frame than actions")
    if history < 1 or horizon < 1 or stride < 1:
        raise ValueError("history, horizon and stride must be positive")

    first_anchor = 1
    last_anchor = states.shape[0] - history - horizon
    if last_anchor < first_anchor:
        return None

    if anchor_indices is None:
        candidates = list(range(first_anchor, last_anchor + 1, stride))
        if max_samples is not None and len(candidates) > max_samples:
            positions = np.linspace(0, len(candidates) - 1, num=max_samples)
            candidates = [
                candidates[int(round(position))]
                for position in positions
            ]
        selected_anchors = candidates
    else:
        selected_anchors = sorted(set(
            int(anchor)
            for anchor in np.asarray(anchor_indices).reshape(-1)
            if first_anchor <= int(anchor) <= last_anchor
        ))
        if not selected_anchors:
            return None

    state_history = []
    action_history = []
    future_actions = []
    future_states = []
    selected_current_indices = []

    for anchor in selected_anchors:
        current = anchor + history - 1
        state_history.append(states[anchor : anchor + history])
        action_history.append(actions[anchor - 1 : anchor + history - 1])
        future_actions.append(actions[current : current + horizon])
        future_states.append(states[current + 1 : current + horizon + 1])
        selected_current_indices.append(current)

    if not state_history:
        return None
    return {
        "state_history": np.stack(state_history).astype(np.float32),
        "action_history": np.stack(action_history).astype(np.float32),
        "future_actions": np.stack(future_actions).astype(np.float32),
        "future_states": np.stack(future_states).astype(np.float32),
        "anchor_indices": np.asarray(selected_current_indices, dtype=np.int64),
    }


def select_contact_event_anchors(
    states,
    history,
    horizon,
    event_samples,
    background_samples,
    threshold=0.5,
):
    """Choose anchors around contact transitions and matched background."""
    states = _as_float_array(states, 2, "states")
    if states.shape[1] != STATE_DIM:
        raise ValueError(f"states must have width {STATE_DIM}")
    if history < 1 or horizon < 1:
        raise ValueError("history and horizon must be positive")
    if event_samples < 0 or background_samples < 0:
        raise ValueError("sample budgets must be non-negative")

    first_anchor = 1
    last_anchor = states.shape[0] - history - horizon
    if last_anchor < first_anchor:
        return None

    binary_contact = states[:, CONTACT_SLICE] >= threshold
    changed = np.any(
        binary_contact[1:] != binary_contact[:-1],
        axis=1,
    )
    transition_indices = np.flatnonzero(changed) + 1

    all_candidates = np.arange(first_anchor, last_anchor + 1)
    event_candidates = set()
    for transition_index in transition_indices:
        for lead in range(1, horizon + 1):
            # ``anchor`` is the first history frame consumed by
            # make_training_samples, while ``current`` is anchor + history - 1.
            # Place the transition exactly ``lead`` frames after current.
            anchor = int(transition_index - lead - history + 1)
            if first_anchor <= anchor <= last_anchor:
                event_candidates.add(anchor)
    all_event_candidates = np.asarray(sorted(event_candidates), dtype=np.int64)
    selected_event_candidates = all_event_candidates
    if (
        event_samples is not None
        and len(selected_event_candidates) > event_samples
    ):
        positions = np.linspace(
            0,
            len(selected_event_candidates) - 1,
            num=event_samples,
        )
        selected_event_candidates = selected_event_candidates[
            np.asarray([int(round(position)) for position in positions])
        ]

    event_candidate_set = set(all_event_candidates.tolist())
    background_candidates = np.asarray(
        [
            anchor
            for anchor in all_candidates
            if anchor not in event_candidate_set
        ],
        dtype=np.int64,
    )
    if (
        background_samples is not None
        and len(background_candidates) > background_samples
    ):
        positions = np.linspace(
            0,
            len(background_candidates) - 1,
            num=background_samples,
        )
        background_candidates = background_candidates[
            np.asarray([int(round(position)) for position in positions])
        ]

    selected = np.concatenate(
        [selected_event_candidates, background_candidates]
    )
    selected = np.unique(selected)
    return selected if selected.size else None
