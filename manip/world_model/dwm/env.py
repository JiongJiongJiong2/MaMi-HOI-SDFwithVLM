"""Deterministic MuJoCo environment for the DWM pilot."""

from __future__ import annotations

import hashlib

import numpy as np

from .config import (
    ACTION_DIM,
    CONTROL_PERIOD,
    OBJECT_CONFIGS,
    PD_GAINS,
    PHYSICS_STEPS_PER_CONTROL,
    PHYSICS_TIMESTEP,
    RESET_SEED,
    ObjectConfig,
)
from .schema import (
    CONTACT_BODY_COUNT,
    CONTACT_MODE_LABELS,
    CONTACT_MODE_TO_INDEX,
    DWMStateV1,
    FINGER_JOINT_COUNT,
    STATE_DIM,
    quaternion_to_matrix,
    rotation_matrix_to_6d,
)
from .xml import build_model_xml

try:
    import mujoco
except ImportError:
    mujoco = None


CONTACT_BODY_NAMES = (
    "R_Wrist",
    "R_Index1",
    "R_Index2",
    "R_Index3",
    "R_Middle1",
    "R_Middle2",
    "R_Middle3",
    "R_Pinky1",
    "R_Pinky2",
    "R_Pinky3",
    "R_Ring1",
    "R_Ring2",
    "R_Ring3",
    "R_Thumb1",
    "R_Thumb2",
    "R_Thumb3",
)


def _require_mujoco():
    if mujoco is None:
        raise RuntimeError(
            "mujoco is required for DWMSimEnv; install mujoco==3.2.3"
        )


def _object_seed(config):
    index = OBJECT_CONFIGS.index(config)
    return int.from_bytes(
        hashlib.sha256(config.object_id.encode("utf-8")).digest()[:4],
        "little",
    ) + index


class DWMSimEnv:
    def __init__(self, config: ObjectConfig, seed=RESET_SEED):
        _require_mujoco()
        if not isinstance(config, ObjectConfig):
            raise TypeError("config must be an ObjectConfig")
        self.config = config
        self.seed = int(seed)
        self.xml = build_model_xml(config)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.model.opt.timestep = PHYSICS_TIMESTEP
        self.data = mujoco.MjData(self.model)
        self._right_hand_joint_ids = [
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in (
                "R_Wrist_sx",
                "R_Wrist_sy",
                "R_Wrist_sz",
                "R_Wrist_x",
                "R_Wrist_y",
                "R_Wrist_z",
                "R_Index1_x",
                "R_Index1_y",
                "R_Index1_z",
                "R_Index2_x",
                "R_Index2_y",
                "R_Index2_z",
                "R_Index3_x",
                "R_Index3_y",
                "R_Index3_z",
                "R_Middle1_x",
                "R_Middle1_y",
                "R_Middle1_z",
                "R_Middle2_x",
                "R_Middle2_y",
                "R_Middle2_z",
                "R_Middle3_x",
                "R_Middle3_y",
                "R_Middle3_z",
                "R_Pinky1_x",
                "R_Pinky1_y",
                "R_Pinky1_z",
                "R_Pinky2_x",
                "R_Pinky2_y",
                "R_Pinky2_z",
                "R_Pinky3_x",
                "R_Pinky3_y",
                "R_Pinky3_z",
                "R_Ring1_x",
                "R_Ring1_y",
                "R_Ring1_z",
                "R_Ring2_x",
                "R_Ring2_y",
                "R_Ring2_z",
                "R_Ring3_x",
                "R_Ring3_y",
                "R_Ring3_z",
                "R_Thumb1_x",
                "R_Thumb1_y",
                "R_Thumb1_z",
                "R_Thumb2_x",
                "R_Thumb2_y",
                "R_Thumb2_z",
                "R_Thumb3_x",
                "R_Thumb3_y",
                "R_Thumb3_z",
            )
        ]
        if len(self._right_hand_joint_ids) != ACTION_DIM:
            raise ValueError("right-hand joint list must have 51 entries")
        if any(joint_id < 0 for joint_id in self._right_hand_joint_ids):
            raise ValueError("right-hand MJCF is missing a named joint")
        self._actuator_joint_ids = self._resolve_actuator_joints()
        self._contact_body_ids = np.asarray([
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                name,
            )
            for name in CONTACT_BODY_NAMES
        ], dtype=np.int64)
        if np.any(self._contact_body_ids < 0):
            raise ValueError("right-hand MJCF is missing contact bodies")
        self._object_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "dwm_object",
        )
        self._object_joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "dwm_object_free",
        )
        if self._object_body_id < 0 or self._object_joint_id < 0:
            raise ValueError("object body/freejoint missing")
        self._object_qpos_adr = int(
            self.model.jnt_qposadr[self._object_joint_id]
        )
        self._object_dof_adr = int(
            self.model.jnt_dofadr[self._object_joint_id]
        )
        self._wrist_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "R_Wrist",
        )
        if self._wrist_body_id < 0:
            raise ValueError("R_Wrist body missing")
        self._contact_seen = False
        self._initial_contact = False
        self._last_reset_id = None

    def _resolve_actuator_joints(self):
        joint_ids = np.asarray(
            self.model.actuator_trnid[:, 0],
            dtype=np.int64,
        )
        if len(joint_ids) != ACTION_DIM:
            raise ValueError(
                f"expected {ACTION_DIM} actuators, got {len(joint_ids)}"
            )
        return joint_ids

    def reset(self, reset_id=0):
        reset_id = int(reset_id)
        if reset_id < 0:
            raise ValueError("reset_id must be non-negative")
        seed_sequence = np.random.SeedSequence([
            self.seed,
            _object_seed(self.config),
            reset_id,
        ])
        rng = np.random.default_rng(seed_sequence)
        mujoco.mj_resetData(self.model, self.data)

        object_x = float(rng.uniform(-0.010, 0.010))
        object_y = float(rng.uniform(-0.010, 0.010))
        object_yaw = float(rng.uniform(-np.pi, np.pi))
        rest_height = self.config.rest_height
        contact_mode = reset_id % 5 != 4
        wrist_yaw = object_yaw
        if contact_mode:
            wrist_height = rest_height + self.config.contact_wrist_offset
            wrist_x = object_x
            wrist_y = object_y
        else:
            wrist_height = (
                rest_height
                + self.config.near_contact_wrist_offset
                + float(
                rng.uniform(-0.004, 0.004)
            )
            )
            wrist_x = object_x + 0.006 * float(rng.normal())
            wrist_y = object_y + 0.006 * float(rng.normal())
            wrist_yaw += 0.05 * float(rng.normal())
        quaternion = np.asarray([
            np.cos(0.5 * object_yaw),
            0.0,
            0.0,
            np.sin(0.5 * object_yaw),
        ])
        self.data.qpos[
            self._object_qpos_adr:self._object_qpos_adr + 3
        ] = [object_x, object_y, rest_height]
        self.data.qpos[
            self._object_qpos_adr + 3:self._object_qpos_adr + 7
        ] = quaternion
        self.data.qvel[self._object_dof_adr:self._object_dof_adr + 6] = 0.0

        for index, joint_id in enumerate(self._right_hand_joint_ids):
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            dof_adr = int(self.model.jnt_dofadr[joint_id])
            self.data.qpos[qpos_adr] = 0.0
            self.data.qvel[dof_adr] = 0.0
        wrist_slide = {
            "R_Wrist_sx": wrist_x,
            "R_Wrist_sy": wrist_y,
            "R_Wrist_sz": wrist_height,
        }
        for name, value in wrist_slide.items():
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = value
        wrist_yaw_joint = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "R_Wrist_z",
        )
        self.data.qpos[int(self.model.jnt_qposadr[wrist_yaw_joint])] = (
            wrist_yaw
        )

        mujoco.mj_forward(self.model, self.data)
        self._contact_seen = bool(self._contact_is_active())
        self._initial_contact = self._contact_seen
        self._last_reset_id = reset_id
        return self.snapshot()

    def position_targets(self):
        targets = np.zeros(ACTION_DIM, dtype=np.float64)
        for index, joint_id in enumerate(self._actuator_joint_ids):
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            targets[index] = self.data.qpos[qpos_adr]
        return targets.astype(np.float32)

    def _gain_for_actuator(self, actuator_index):
        if actuator_index < 3:
            return PD_GAINS["wrist_translation"]
        if actuator_index < 6:
            return PD_GAINS["wrist_rotation"]
        return PD_GAINS["finger"]

    def _contact_is_active(self):
        return bool(np.any(
            np.linalg.norm(self._compute_contact_forces(), axis=-1)
            > 1e-4
        ))

    def _compute_contact_forces(self):
        forces = np.zeros((CONTACT_BODY_COUNT, 3), dtype=np.float64)
        body_to_contact_index = {
            int(body_id): index
            for index, body_id in enumerate(self._contact_body_ids)
        }
        force_result = np.zeros(6, dtype=np.float64)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            geom_bodies = (
                int(self.model.geom_bodyid[contact.geom1]),
                int(self.model.geom_bodyid[contact.geom2]),
            )
            hand_side = None
            if geom_bodies[0] in body_to_contact_index:
                hand_side = 0
            elif geom_bodies[1] in body_to_contact_index:
                hand_side = 1
            if hand_side is None:
                continue
            other_body = geom_bodies[1 - hand_side]
            if other_body != self._object_body_id:
                continue
            mujoco.mj_contactForce(
                self.model,
                self.data,
                contact_index,
                force_result,
            )
            world_force = contact.frame.reshape(3, 3) @ force_result[:3]
            if hand_side == 0:
                world_force = -world_force
            body_id = geom_bodies[hand_side]
            index = body_to_contact_index[body_id]
            forces[index] += world_force
        return forces

    def step(self, action_targets):
        action_targets = np.asarray(action_targets, dtype=np.float64)
        if action_targets.shape != (ACTION_DIM,):
            raise ValueError(
                f"action must have shape ({ACTION_DIM},), "
                f"got {action_targets.shape}"
            )
        for _ in range(PHYSICS_STEPS_PER_CONTROL):
            for index, joint_id in enumerate(self._actuator_joint_ids):
                qpos_adr = int(self.model.jnt_qposadr[joint_id])
                dof_adr = int(self.model.jnt_dofadr[joint_id])
                kp, kd = self._gain_for_actuator(index)
                torque = (
                    kp
                    * (action_targets[index] - self.data.qpos[qpos_adr])
                    - kd * self.data.qvel[dof_adr]
                )
                lower, upper = self.model.actuator_ctrlrange[index]
                if self.model.actuator_ctrllimited[index]:
                    torque = np.clip(torque, lower, upper)
                self.data.ctrl[index] = torque
            mujoco.mj_step(self.model, self.data)
        active = self._contact_is_active()
        self._contact_seen = self._contact_seen or active
        return self.snapshot()

    def rollout(self, action_chunk):
        action_chunk = np.asarray(action_chunk, dtype=np.float64)
        if action_chunk.ndim != 2 or action_chunk.shape[1] != ACTION_DIM:
            raise ValueError("action_chunk must be [H, 51]")
        states = [self.snapshot()]
        for action in action_chunk:
            states.append(self.step(action))
        return states

    def snapshot(self):
        object_position = self.data.qpos[
            self._object_qpos_adr:self._object_qpos_adr + 3
        ].copy()
        object_quaternion = self.data.qpos[
            self._object_qpos_adr + 3:self._object_qpos_adr + 7
        ]
        object_rotation = rotation_matrix_to_6d(
            quaternion_to_matrix(object_quaternion)
        )
        object_pose = np.concatenate(
            [object_position, object_rotation],
        )
        object_twist = self.data.qvel[
            self._object_dof_adr:self._object_dof_adr + 6
        ].copy()

        wrist_position = self.data.xpos[self._wrist_body_id].copy()
        wrist_rotation = rotation_matrix_to_6d(
            quaternion_to_matrix(self.data.xquat[self._wrist_body_id])
        )
        wrist_pose = np.concatenate([wrist_position, wrist_rotation])
        wrist_cvel = self.data.cvel[self._wrist_body_id].copy()
        wrist_twist = np.concatenate([wrist_cvel[3:6], wrist_cvel[0:3]])

        finger_positions = np.asarray([
            self.data.qpos[int(self.model.jnt_qposadr[joint_id])]
            for joint_id in self._right_hand_joint_ids[6:]
        ], dtype=np.float64)
        finger_velocities = np.asarray([
            self.data.qvel[int(self.model.jnt_dofadr[joint_id])]
            for joint_id in self._right_hand_joint_ids[6:]
        ], dtype=np.float64)
        if finger_positions.shape != (FINGER_JOINT_COUNT,):
            raise ValueError("finger position extraction failed")
        contact_forces = self._compute_contact_forces()
        if contact_forces.shape != (CONTACT_BODY_COUNT, 3):
            raise ValueError("contact force extraction failed")

        vector = np.concatenate([
            object_pose,
            object_twist,
            wrist_pose,
            wrist_twist,
            finger_positions,
            finger_velocities,
            contact_forces.reshape(-1),
        ])
        if vector.shape != (STATE_DIM,):
            raise ValueError(
                f"state extraction produced {vector.shape}, "
                f"expected ({STATE_DIM},)"
            )
        return DWMStateV1(
            vector=vector.astype(np.float32),
            object_pose=object_pose.astype(np.float32),
            object_twist=object_twist.astype(np.float32),
            wrist_pose=wrist_pose.astype(np.float32),
            wrist_twist=wrist_twist.astype(np.float32),
            finger_positions=finger_positions.astype(np.float32),
            finger_velocities=finger_velocities.astype(np.float32),
            contact_forces=contact_forces.astype(np.float32),
        )

    def contact_mode(self):
        active = self._contact_is_active()
        if not self._contact_seen:
            return "none"
        if self._initial_contact and not active:
            return "release"
        if active:
            object_velocity = self.data.qvel[
                self._object_dof_adr:self._object_dof_adr + 3
            ]
            wrist_cvel = self.data.cvel[self._wrist_body_id]
            relative = object_velocity - wrist_cvel[3:6]
            force = float(np.linalg.norm(
                self._compute_contact_forces(),
                axis=-1,
            ).max())
            if np.linalg.norm(relative) > 0.020 or force < 0.050:
                return "slip"
            return "contact"
        return "none"

    @property
    def contact_mode_index(self):
        return CONTACT_MODE_TO_INDEX[self.contact_mode()]

    @property
    def control_period(self):
        return CONTROL_PERIOD
