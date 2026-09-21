"""Deterministic counterfactual action branches for DWM D0-B."""

from __future__ import annotations

import re

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

from .config import ACTION_DIM


HORIZON = 8
BRANCH_NAMES = (
    "hold",
    "close",
    "open",
    "push_x+",
    "push_x-",
    "push_y+",
    "push_y-",
    "lift",
    "lower",
    "random_0",
    "random_1",
    "random_2",
    "random_3",
)


def _finger_groups(env):
    groups = {}
    for actuator_index, joint_id in enumerate(env._actuator_joint_ids):
        if actuator_index < 6:
            continue
        name = mujoco.mj_id2name(
            env.model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_index,
        )
        if name is None or not name.endswith("_x"):
            continue
        finger_name = re.sub(r"\d+_x$", "", name)
        groups.setdefault(finger_name, []).append(actuator_index)
    if len(groups) != 5:
        raise ValueError(f"expected five finger groups, got {sorted(groups)}")
    return groups


def _last_phalanx_body_id(env, finger_name):
    body_name = f"{finger_name}3"
    body_id = mujoco.mj_name2id(
        env.model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )
    if body_id < 0:
        raise ValueError(f"missing body {body_name}")
    return body_id


def calibrate_flexion_signs(env):
    if mujoco is None:
        raise RuntimeError("mujoco is required")
    saved = mujoco.MjData(env.model)
    saved.qpos[:] = env.data.qpos
    saved.qvel[:] = env.data.qvel
    base = env.position_targets().astype(np.float64)
    signs = np.ones(ACTION_DIM, dtype=np.float64)
    try:
        wrist_position = env.data.xpos[env._wrist_body_id].copy()
        for finger_name, actuator_indices in _finger_groups(env).items():
            body_id = _last_phalanx_body_id(env, finger_name)
            distances = []
            for sign in (-1.0, 1.0):
                for actuator_index in actuator_indices:
                    joint_id = env._actuator_joint_ids[actuator_index]
                    qpos_adr = int(env.model.jnt_qposadr[joint_id])
                    env.data.qpos[qpos_adr] = (
                        base[actuator_index] + sign * 0.20
                    )
                mujoco.mj_forward(env.model, env.data)
                distances.append(float(np.linalg.norm(
                    env.data.xpos[body_id] - wrist_position
                )))
                for actuator_index in actuator_indices:
                    joint_id = env._actuator_joint_ids[actuator_index]
                    qpos_adr = int(env.model.jnt_qposadr[joint_id])
                    env.data.qpos[qpos_adr] = base[actuator_index]
            close_sign = -1.0 if distances[0] < distances[1] else 1.0
            for actuator_index in actuator_indices:
                signs[actuator_index] = close_sign
    finally:
        env.data.qpos[:] = saved.qpos
        env.data.qvel[:] = saved.qvel
        mujoco.mj_forward(env.model, env.data)
    return signs


def _repeat_targets(targets, horizon):
    return np.repeat(
        np.asarray(targets, dtype=np.float64)[None],
        horizon,
        axis=0,
    )


def _smooth_residual(horizon, rng, scale, size):
    knot_count = 3
    knot_times = np.linspace(0.0, 1.0, knot_count)
    times = np.linspace(0.0, 1.0, horizon)
    knots = rng.normal(0.0, scale, size=(knot_count, size))
    return np.stack([
        np.interp(times, knot_times, knots[:, index])
        for index in range(size)
    ], axis=-1)


def make_action_branches(env, seed=20260922):
    if mujoco is None:
        raise RuntimeError("mujoco is required")
    base = env.position_targets().astype(np.float64)
    signs = calibrate_flexion_signs(env)
    finger_indices = np.arange(6, ACTION_DIM)
    actions = {
        "hold": _repeat_targets(base, HORIZON),
    }

    close = base.copy()
    close[finger_indices] += signs[finger_indices] * np.asarray([
        0.20 if (index - 6) % 3 == 0 else
        0.30 if (index - 6) % 3 == 1 else
        0.25
        for index in finger_indices
    ])
    actions["close"] = _repeat_targets(close, HORIZON)

    open_action = base.copy()
    open_action[finger_indices] -= signs[finger_indices] * 0.15
    actions["open"] = _repeat_targets(open_action, HORIZON)

    for name, axis, delta in (
        ("push_x+", 0, 0.010),
        ("push_x-", 0, -0.010),
        ("push_y+", 1, 0.010),
        ("push_y-", 1, -0.010),
        ("lift", 2, 0.010),
        ("lower", 2, -0.010),
    ):
        target = base.copy()
        target[axis] += delta
        actions[name] = _repeat_targets(target, HORIZON)

    rng = np.random.default_rng(seed)
    for random_index in range(4):
        residual = np.zeros((HORIZON, ACTION_DIM), dtype=np.float64)
        residual[:, :3] = _smooth_residual(
            HORIZON,
            rng,
            scale=0.004,
            size=3,
        )
        residual[:, finger_indices] = (
            _smooth_residual(
                HORIZON,
                rng,
                scale=0.08,
                size=len(finger_indices),
            )
            * signs[finger_indices][None]
        )
        actions[f"random_{random_index}"] = (
            base[None] + residual
        )

    if tuple(actions) != BRANCH_NAMES:
        raise ValueError("branch order does not match frozen names")
    return actions
