#!/usr/bin/env python3
"""Project HandX 21-joint sequences onto legal MANO parameters.

The projection is deterministic for a fixed input and seed. It optimizes one
shared MANO shape and per-frame MANO pose/translation for each hand, then
reports skeleton validity, fit error, and temporal ratios against the input.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


SKELETON_CHAIN = np.asarray(
    [
        [0, 13, 14, 15, 16],
        [0, 1, 2, 3, 17],
        [0, 4, 5, 6, 18],
        [0, 10, 11, 12, 19],
        [0, 7, 8, 9, 20],
    ],
    dtype=np.int64,
)
ARTICULATED_CHAIN = SKELETON_CHAIN[:, :-1]
MANO_TO_HANDX_ORDER = np.asarray(
    [0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20],
    dtype=np.int64,
)
WRIST_JOINTS = (0, 21)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--mano-root",
        type=Path,
        default=Path("/root/autodl-tmp/external/ContactOpt/mano/models"),
    )
    parser.add_argument(
        "--contactopt-root",
        type=Path,
        default=Path("/root/autodl-tmp/external/ContactOpt"),
    )
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--beta-reg-weight", type=float, default=1e-4)
    parser.add_argument("--pose-reg-weight", type=float, default=1e-5)
    parser.add_argument("--temporal-velocity-weight", type=float, default=0.0)
    parser.add_argument(
        "--temporal-acceleration-weight", type=float, default=0.0
    )
    parser.add_argument("--joint-acceleration-weight", type=float, default=0.0)
    parser.add_argument(
        "--rotation-smoothing-kernel", type=int, default=5
    )
    parser.add_argument(
        "--direction-smoothing-kernel", type=int, default=5
    )
    parser.add_argument("--output-smoothing-kernel", type=int, default=1)
    parser.add_argument("--max-fit-mean-mm", type=float, default=15.0)
    parser.add_argument("--max-wrist-rmse-mm", type=float, default=3.0)
    parser.add_argument("--max-projected-bone-cv", type=float, default=0.001)
    parser.add_argument("--max-speed-ratio", type=float, default=2.0)
    parser.add_argument("--max-acceleration-ratio", type=float, default=2.0)
    parser.add_argument("--max-jerk-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def extract_generated_motion(value):
    motion = np.asarray(value, dtype=np.float64)
    if motion.ndim == 4 and motion.shape[0] == 1:
        motion = motion[0]
    if motion.ndim == 3 and motion.shape[0] == 1 and motion.shape[-1] == 42 * 4:
        motion = motion[0]
    if motion.ndim == 2 and motion.shape[-1] == 42 * 4:
        motion = motion.reshape(motion.shape[0], 42, 4)
    if motion.ndim != 3 or motion.shape[1:] != (42, 4):
        raise ValueError(
            f"Expected generated motion (frames, 42, 4), got {motion.shape}"
        )
    return motion[:, :, :3]


def chain_lengths(joints, chains):
    joints = np.asarray(joints, dtype=np.float64)
    values = []
    for chain in chains:
        values.append(
            np.linalg.norm(
                joints[:, chain[1:]] - joints[:, chain[:-1]],
                axis=2,
            )
        )
    return np.concatenate(values, axis=1)


def chain_length_cv(joints, chains):
    lengths = chain_lengths(joints, chains)
    cv = lengths.std(axis=0) / (lengths.mean(axis=0) + 1e-12)
    return {
        "cv_mean": float(cv.mean()),
        "cv_max": float(cv.max()),
        "cv_p95": float(np.percentile(cv, 95)),
    }


def bone_length_cv(joints):
    articulated = chain_length_cv(joints, ARTICULATED_CHAIN)
    tips = chain_length_cv(joints, SKELETON_CHAIN)
    return {
        "bone_cv_mean": articulated["cv_mean"],
        "bone_cv_max": articulated["cv_max"],
        "bone_cv_p95": articulated["cv_p95"],
        "tip_cv_mean": tips["cv_mean"],
        "tip_cv_max": tips["cv_max"],
        "tip_cv_p95": tips["cv_p95"],
    }


def trajectory_stat(joints):
    joints = np.asarray(joints, dtype=np.float64)
    norm = lambda array: np.linalg.norm(array, axis=-1)
    return {
        "speed_mm": float(norm(np.diff(joints, axis=0)).mean() * 1000.0),
        "acceleration_mm": float(
            norm(np.diff(joints, n=2, axis=0)).mean() * 1000.0
        ),
        "jerk_mm": float(
            norm(np.diff(joints, n=3, axis=0)).mean() * 1000.0
        ),
    }


def ratio(value, baseline):
    if baseline <= 0:
        return None
    return float(value / baseline)


def rigid_align(source, target):
    """Return rotation and translation mapping source points to target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def initialize_world_transform(neutral_joints, target_joints):
    rotations = np.zeros((len(target_joints), 3, 3), dtype=np.float32)
    translations = np.zeros((len(target_joints), 3), dtype=np.float32)
    for index, target in enumerate(target_joints):
        rotation, _ = rigid_align(neutral_joints, target)
        translation = target[0] - rotation @ neutral_joints[0]
        rotations[index] = rotation
        translations[index] = translation
    return rotations, translations


def smooth_world_rotations(
    rotations,
    target_wrists,
    neutral_wrist,
    kernel_size,
):
    from scipy.spatial.transform import Rotation

    if kernel_size <= 1:
        rotations = np.asarray(rotations)
        translations = (
            target_wrists
            - np.einsum("bij,j->bi", rotations, neutral_wrist)
        )
        return rotations, translations
    if kernel_size % 2 == 0:
        raise ValueError("rotation smoothing kernel must be odd")
    kernel = np.asarray([1.0])
    for _ in range(kernel_size - 1):
        kernel = np.convolve(kernel, [1.0, 1.0])
    kernel = kernel / kernel.sum()
    quaternions = Rotation.from_matrix(rotations).as_quat()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1.0
    radius = kernel_size // 2
    padded = np.pad(
        quaternions,
        ((radius, radius), (0, 0)),
        mode="edge",
    )
    smoothed = np.stack(
        [
            np.sum(
                padded[start : start + kernel_size] * kernel[:, None],
                axis=0,
            )
            for start in range(len(quaternions))
        ]
    )
    smoothed = smoothed / np.linalg.norm(smoothed, axis=1, keepdims=True)
    smoothed_rotations = Rotation.from_quat(smoothed).as_matrix()
    smoothed_translations = (
        target_wrists
        - np.einsum("bij,j->bi", smoothed_rotations, neutral_wrist)
    )
    return smoothed_rotations, smoothed_translations


def smooth_unit_vectors(vectors, kernel_size):
    if kernel_size <= 1:
        return np.asarray(vectors)
    if kernel_size % 2 == 0:
        raise ValueError("direction smoothing kernel must be odd")
    kernel = np.asarray([1.0])
    for _ in range(kernel_size - 1):
        kernel = np.convolve(kernel, [1.0, 1.0])
    kernel = kernel / kernel.sum()
    values = np.asarray(vectors, dtype=np.float64)
    for index in range(1, len(values)):
        if np.dot(values[index - 1], values[index]) < 0:
            values[index] *= -1.0
    radius = kernel_size // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.stack(
        [
            np.sum(
                padded[start : start + kernel_size] * kernel[:, None],
                axis=0,
            )
            for start in range(len(values))
        ]
    )
    return smoothed / np.maximum(
        np.linalg.norm(smoothed, axis=1, keepdims=True), 1e-12
    )


def smooth_sequence(values, kernel_size):
    if kernel_size <= 1:
        return np.asarray(values)
    if kernel_size % 2 == 0:
        raise ValueError("sequence smoothing kernel must be odd")
    kernel = np.asarray([1.0])
    for _ in range(kernel_size - 1):
        kernel = np.convolve(kernel, [1.0, 1.0])
    kernel = kernel / kernel.sum()
    radius = kernel_size // 2
    padded = np.pad(
        np.asarray(values),
        ((radius, radius),) + ((0, 0),) * (np.asarray(values).ndim - 1),
        mode="edge",
    )
    return np.stack(
        [
            np.sum(
                padded[start : start + kernel_size]
                * kernel.reshape((-1,) + (1,) * (padded.ndim - 1)),
                axis=0,
            )
            for start in range(len(values))
        ]
    )


def project_skeleton_to_template(
    target_joints,
    template_joints,
    direction_smoothing_kernel=1,
):
    target = np.asarray(target_joints, dtype=np.float64)
    template = np.asarray(template_joints, dtype=np.float64)
    projected = np.zeros_like(target)
    projected[:, 0] = target[:, 0]
    for chain in SKELETON_CHAIN:
        for parent, joint in zip(chain[:-1], chain[1:]):
            direction = target[:, joint] - target[:, parent]
            direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
            direction = direction / np.maximum(direction_norm, 1e-12)
            direction = smooth_unit_vectors(
                direction, direction_smoothing_kernel
            )
            template_length = np.linalg.norm(
                template[joint] - template[parent]
            )
            projected[:, joint] = (
                projected[:, parent] + direction * template_length
            )
    return projected


def make_mano_layer(contactopt_root, mano_root, side):
    import sys
    import torch

    if str(contactopt_root) not in sys.path:
        sys.path.insert(0, str(contactopt_root))
    from manopth.manolayer import ManoLayer

    return ManoLayer(
        mano_root=str(mano_root),
        use_pca=False,
        side=side,
        flat_hand_mean=False,
    )


def mano_forward(layer, pose, betas, trans):
    vertices, joints = layer(pose, betas, trans)
    return vertices, joints[:, MANO_TO_HANDX_ORDER]


def project_hand(
    target_joints,
    side,
    contactopt_root,
    mano_root,
    iterations,
    lr,
    beta_reg_weight,
    pose_reg_weight,
    temporal_velocity_weight,
    temporal_acceleration_weight,
    joint_acceleration_weight,
    rotation_smoothing_kernel,
    direction_smoothing_kernel,
    seed,
    target_override=None,
):
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(seed)
    layer = make_mano_layer(contactopt_root, mano_root, side).cuda()
    layer.eval()
    for parameter in layer.parameters():
        parameter.requires_grad_(False)

    with torch.no_grad():
        _, neutral = mano_forward(
            layer,
            torch.zeros(1, 48, device="cuda"),
            torch.zeros(1, 10, device="cuda"),
            torch.zeros(1, 3, device="cuda"),
        )
    neutral = (neutral[0] / 1000.0).cpu().numpy()
    if target_override is None:
        canonical_target = project_skeleton_to_template(
            target_joints,
            neutral,
            direction_smoothing_kernel,
        )
    else:
        canonical_target = np.asarray(
            target_override, dtype=np.float64
        )
    target = torch.as_tensor(
        canonical_target, dtype=torch.float32, device="cuda"
    )
    initial_rotation, initial_translation = initialize_world_transform(
        neutral, canonical_target
    )
    initial_rotation, initial_translation = smooth_world_rotations(
        initial_rotation,
        canonical_target[:, 0],
        neutral[0],
        rotation_smoothing_kernel,
    )

    world_rotation = torch.as_tensor(
        initial_rotation, dtype=torch.float32, device="cuda"
    )
    world_translation = torch.as_tensor(
        initial_translation, dtype=torch.float32, device="cuda"
    )
    betas = torch.zeros(10, dtype=torch.float32, device="cuda")
    finger_pose = torch.nn.Parameter(
        torch.zeros(
            len(target_joints), 45, dtype=torch.float32, device="cuda"
        )
    )
    optimizer = torch.optim.Adam([finger_pose], lr=lr)

    def full_pose():
        roots = torch.zeros(
            len(finger_pose), 3, dtype=torch.float32, device="cuda"
        )
        return torch.cat([roots, finger_pose], dim=1)

    def to_world(local_joints):
        return torch.einsum(
            "bij,bkj->bki", world_rotation, local_joints
        ) + world_translation.unsqueeze(1)

    with torch.no_grad():
        _, local_joints = mano_forward(
            layer,
            full_pose(),
            betas.unsqueeze(0).expand(len(finger_pose), -1),
            torch.zeros(len(finger_pose), 3, device="cuda"),
        )
        joints = to_world(local_joints / 1000.0)
        initial_loss = functional.smooth_l1_loss(
            joints, target, beta=0.005
        ).item()

    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        _, local_joints = mano_forward(
            layer,
            full_pose(),
            betas.unsqueeze(0).expand(len(finger_pose), -1),
            torch.zeros(len(finger_pose), 3, device="cuda"),
        )
        joints = to_world(local_joints / 1000.0)
        fit = functional.smooth_l1_loss(joints, target, beta=0.005)
        loss = fit + pose_reg_weight * finger_pose.square().mean()
        if len(finger_pose) > 1:
            velocity = finger_pose[1:] - finger_pose[:-1]
            loss = loss + temporal_velocity_weight * velocity.square().mean()
        if len(finger_pose) > 2:
            acceleration = (
                finger_pose[2:]
                - 2.0 * finger_pose[1:-1]
                + finger_pose[:-2]
            )
            loss = (
                loss
                + temporal_acceleration_weight * acceleration.square().mean()
            )
            joint_acceleration = (
                joints[2:] - 2.0 * joints[1:-1] + joints[:-2]
            )
            loss = (
                loss
                + joint_acceleration_weight
                * joint_acceleration.square().mean()
            )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        local_vertices, local_joints = mano_forward(
            layer,
            full_pose(),
            betas.unsqueeze(0).expand(len(finger_pose), -1),
            torch.zeros(len(finger_pose), 3, device="cuda"),
        )
        vertices = to_world(local_vertices / 1000.0)
        joints = to_world(local_joints / 1000.0)

    result = {
        "vertices": vertices.cpu().numpy().astype(np.float64),
        "joints": joints.cpu().numpy().astype(np.float64),
        "pose": full_pose().detach().cpu().numpy().astype(np.float64),
        "trans": world_translation.detach().cpu().numpy().astype(np.float64),
        "world_rotation": (
            world_rotation.detach().cpu().numpy().astype(np.float64)
        ),
        "betas": betas.detach().cpu().numpy().astype(np.float64),
        "canonical_target": canonical_target,
        "initial_smooth_l1": float(initial_loss),
    }
    return result


def sample_metrics(sample_path, projection):
    with sample_path.open("rb") as handle:
        sample = pickle.load(handle)
    target = extract_generated_motion(sample["generated_real"])
    target_left = target[:, :21]
    target_right = target[:, 21:]
    output = {
        "sample": sample_path.name,
        "frames": int(len(target)),
        "target": {
            "left": {
                **bone_length_cv(target_left),
                "trajectory": trajectory_stat(target_left),
            },
            "right": {
                **bone_length_cv(target_right),
                "trajectory": trajectory_stat(target_right),
            },
        },
        "projected": {},
    }
    for side, target_side in (
        ("left", target_left),
        ("right", target_right),
    ):
        projected = projection[side]
        projected_joints = projected["joints"]
        canonical_target = projected["canonical_target"]
        canonical_error = np.linalg.norm(
            projected_joints - canonical_target, axis=2
        )
        raw_error = np.linalg.norm(projected_joints - target_side, axis=2)
        wrist_error = np.linalg.norm(
            projected_joints[:, 0] - target_side[:, 0], axis=1
        )
        target_stats = trajectory_stat(target_side)
        projected_stats = trajectory_stat(projected_joints)
        output["projected"][side] = {
            **bone_length_cv(projected_joints),
            "canonical_fit_mean_mm": float(
                canonical_error.mean() * 1000.0
            ),
            "canonical_fit_p95_mm": float(
                np.percentile(canonical_error, 95) * 1000.0
            ),
            "canonical_fit_max_mm": float(
                canonical_error.max() * 1000.0
            ),
            "raw_fit_mean_mm": float(raw_error.mean() * 1000.0),
            "raw_fit_p95_mm": float(
                np.percentile(raw_error, 95) * 1000.0
            ),
            "raw_fit_max_mm": float(raw_error.max() * 1000.0),
            "wrist_rmse_mm": float(
                np.sqrt(np.mean(np.square(wrist_error))) * 1000.0
            ),
            "trajectory": projected_stats,
            "speed_ratio": ratio(
                projected_stats["speed_mm"], target_stats["speed_mm"]
            ),
            "acceleration_ratio": ratio(
                projected_stats["acceleration_mm"],
                target_stats["acceleration_mm"],
            ),
            "jerk_ratio": ratio(
                projected_stats["jerk_mm"], target_stats["jerk_mm"]
            ),
        }
    return output


def sample_gate(case, args):
    gates = {}
    for side in ("left", "right"):
        row = case["projected"][side]
        gates[side] = {
            "fit_mean": (
                row["canonical_fit_mean_mm"] <= args.max_fit_mean_mm
            ),
            "wrist_rmse": (
                row["wrist_rmse_mm"] <= args.max_wrist_rmse_mm
            ),
            "bone_cv": (
                row["bone_cv_mean"] <= args.max_projected_bone_cv
            ),
            "speed": (
                row["speed_ratio"] is None
                or row["speed_ratio"] <= args.max_speed_ratio
            ),
            "acceleration": (
                row["acceleration_ratio"] is None
                or row["acceleration_ratio"]
                <= args.max_acceleration_ratio
            ),
            "jerk": (
                row["jerk_ratio"] is None
                or row["jerk_ratio"] <= args.max_jerk_ratio
            ),
        }
        gates[side]["passed"] = all(gates[side].values())
    return gates


def save_projection(path, projection):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for side in ("left", "right"):
        payload[f"{side}_pose"] = projection[side]["pose"].astype(np.float32)
        payload[f"{side}_trans"] = projection[side]["trans"].astype(np.float32)
        payload[f"{side}_world_rotation"] = projection[side][
            "world_rotation"
        ].astype(np.float32)
        payload[f"{side}_betas"] = projection[side]["betas"].astype(np.float32)
        payload[f"{side}_joints"] = projection[side]["joints"].astype(np.float32)
        payload[f"{side}_canonical_target"] = projection[side][
            "canonical_target"
        ].astype(np.float32)
        payload[f"{side}_vertices"] = projection[side][
            "vertices"
        ].astype(np.float32)
    np.savez_compressed(path, **payload)


def main():
    args = parse_args()
    sample_paths = sorted(args.input_dir.glob("val_sample_*.pkl"))
    if not sample_paths:
        raise ValueError(f"No val_sample_*.pkl files in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    cases = []
    for sample_path in sample_paths:
        with sample_path.open("rb") as handle:
            sample = pickle.load(handle)
        motion = extract_generated_motion(sample["generated_real"])
        projection = {}
        for index, side in enumerate(("left", "right")):
            projection[side] = project_hand(
                motion[:, index * 21 : (index + 1) * 21],
                side,
                args.contactopt_root,
                args.mano_root,
                args.iterations,
                args.lr,
                args.beta_reg_weight,
                args.pose_reg_weight,
                args.temporal_velocity_weight,
                args.temporal_acceleration_weight,
                args.joint_acceleration_weight,
                args.rotation_smoothing_kernel,
                args.direction_smoothing_kernel,
                args.seed + index,
            )
            if args.output_smoothing_kernel > 1:
                smoothed_joints = smooth_sequence(
                    projection[side]["joints"],
                    args.output_smoothing_kernel,
                )
                smoothed_joints[:, 0] = projection[side]["joints"][:, 0]
                projection[side] = project_hand(
                    motion[:, index * 21 : (index + 1) * 21],
                    side,
                    args.contactopt_root,
                    args.mano_root,
                    args.iterations,
                    args.lr,
                    args.beta_reg_weight,
                    args.pose_reg_weight,
                    args.temporal_velocity_weight,
                    args.temporal_acceleration_weight,
                    args.joint_acceleration_weight,
                    args.rotation_smoothing_kernel,
                    args.direction_smoothing_kernel,
                    args.seed + index,
                    target_override=smoothed_joints,
                )
        case = sample_metrics(sample_path, projection)
        case["gate"] = sample_gate(case, args)
        case["passed"] = all(
            case["gate"][side]["passed"] for side in ("left", "right")
        )
        output_path = args.output_dir / f"{sample_path.stem}_mano.npz"
        save_projection(output_path, projection)
        case["output_npz"] = str(output_path)
        cases.append(case)
        print(
            json.dumps(
                {
                    "sample": case["sample"],
                    "passed": case["passed"],
                    "gate": case["gate"],
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )

    summary = {
        "input_dir": str(args.input_dir),
        "case_count": len(cases),
        "passed_case_count": sum(case["passed"] for case in cases),
        "args": {
            "iterations": args.iterations,
            "lr": args.lr,
            "beta_reg_weight": args.beta_reg_weight,
            "pose_reg_weight": args.pose_reg_weight,
            "temporal_velocity_weight": args.temporal_velocity_weight,
            "temporal_acceleration_weight": (
                args.temporal_acceleration_weight
            ),
            "joint_acceleration_weight": args.joint_acceleration_weight,
            "rotation_smoothing_kernel": args.rotation_smoothing_kernel,
            "direction_smoothing_kernel": (
                args.direction_smoothing_kernel
            ),
            "output_smoothing_kernel": args.output_smoothing_kernel,
            "max_fit_mean_mm": args.max_fit_mean_mm,
            "max_wrist_rmse_mm": args.max_wrist_rmse_mm,
            "max_projected_bone_cv": args.max_projected_bone_cv,
            "max_speed_ratio": args.max_speed_ratio,
            "max_acceleration_ratio": args.max_acceleration_ratio,
            "max_jerk_ratio": args.max_jerk_ratio,
            "seed": args.seed,
        },
        "cases": cases,
    }
    summary["passed"] = (
        len(cases) > 0
        and summary["passed_case_count"] == summary["case_count"]
    )
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
