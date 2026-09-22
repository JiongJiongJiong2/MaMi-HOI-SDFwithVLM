#!/usr/bin/env python3
"""Compare fixed ContactOpt temporal-correction arms with direction control."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


ARMS = (
    "direction_constrained",
    "ordinary_smoothing",
    "contact_projection",
    "strong_contact",
    "unconstrained_continuation",
)
NEAR_THRESHOLD_M = 0.02
DEFAULT_NORMAL_WEIGHT = 20.0
DEFAULT_STRONG_CONTACT_WEIGHT = 100.0
DEFAULT_MAX_POSE_DELTA_L2 = 1.8
DEFAULT_PROJECTION_DAMPING = 1e-4
DEFAULT_PROJECTION_MAX_STEP = 0.1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--e5-root", type=Path, required=True)
    parser.add_argument("--e5t-root", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--mano-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--normal-tolerance-m", type=float, default=0.0005)
    parser.add_argument("--contact-tolerance-m", type=float, default=0.001)
    parser.add_argument(
        "--contactopt-root",
        type=Path,
        default=Path("/root/autodl-tmp/external/ContactOpt"),
    )
    parser.add_argument(
        "--split",
        nargs="+",
        choices=("train", "dev"),
        default=("train", "dev"),
    )
    parser.add_argument("--sequence-limit", type=int)
    parser.add_argument("--sequence", nargs="*", default=[])
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--object-samples", type=int, default=4096)
    parser.add_argument("--anchor-weight", type=float, default=0.2)
    parser.add_argument("--pose-anchor-weight", type=float, default=0.05)
    parser.add_argument("--acceleration-weight", type=float, default=2.0)
    parser.add_argument("--jerk-weight", type=float, default=2.0)
    parser.add_argument("--tail-weight", type=float, default=8.0)
    parser.add_argument("--contact-weight", type=float, default=10.0)
    parser.add_argument("--contact-margin-m", type=float, default=0.001)
    parser.add_argument(
        "--normal-weight",
        type=float,
        default=DEFAULT_NORMAL_WEIGHT,
    )
    parser.add_argument(
        "--strong-contact-weight",
        type=float,
        default=DEFAULT_STRONG_CONTACT_WEIGHT,
    )
    parser.add_argument(
        "--max-pose-delta-l2",
        type=float,
        default=DEFAULT_MAX_POSE_DELTA_L2,
    )
    parser.add_argument(
        "--projection-damping",
        type=float,
        default=DEFAULT_PROJECTION_DAMPING,
    )
    parser.add_argument(
        "--projection-max-step",
        type=float,
        default=DEFAULT_PROJECTION_MAX_STEP,
    )
    parser.add_argument("--max-speed-ratio", type=float, default=2.0)
    parser.add_argument("--max-acceleration-ratio", type=float, default=2.0)
    parser.add_argument("--max-jerk-ratio", type=float, default=3.0)
    parser.add_argument("--max-wrist-drift-m", type=float, default=0.00001)
    parser.add_argument("--max-object-drift-m", type=float, default=0.000001)
    parser.add_argument(
        "--max-distance-relative-change",
        type=float,
        default=0.10,
    )
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.iterations < 0:
        raise ValueError("iterations must be non-negative")
    for name in (
        "normal_tolerance_m",
        "contact_tolerance_m",
        "max_pose_delta_l2",
        "projection_damping",
        "projection_max_step",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_delta_norms(pose, base_pose):
    pose = np.asarray(pose, dtype=np.float64)
    base_pose = np.asarray(base_pose, dtype=np.float64)
    if pose.shape != base_pose.shape:
        raise ValueError("pose and base_pose must align")
    return np.linalg.norm(pose[:, 3:18] - base_pose[:, 3:18], axis=1)


def clip_pose_delta(delta, max_norm):
    delta = np.asarray(delta, dtype=np.float64)
    if max_norm < 0:
        raise ValueError("max_norm must be non-negative")
    norms = np.linalg.norm(delta, axis=-1, keepdims=True)
    scales = np.ones_like(norms)
    active = norms[..., 0] > max_norm
    if max_norm == 0:
        scales[active] = 0.0
    else:
        scales[active] = max_norm / norms[active]
    return delta * scales


def nearest_object_vertices(points, objects):
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        distances, indices = cKDTree(objects).query(points, k=1)
        return distances, indices

    distances = np.empty(len(points), dtype=np.float64)
    indices = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), 64):
        stop = start + 64
        squared = np.sum(
            (
                points[start:stop, None, :]
                - objects[None, :, :]
            )
            ** 2,
            axis=2,
        )
        local_indices = np.argmin(squared, axis=1)
        indices[start:stop] = local_indices
        distances[start:stop] = np.sqrt(
            squared[np.arange(len(local_indices)), local_indices]
        )
    return distances, indices


def normal_displacement_components(
    base_vertices,
    candidate_vertices,
    object_vertices,
    near_threshold_m=NEAR_THRESHOLD_M,
):
    """Return E5-to-candidate normal displacement for near-surface vertices."""
    base = np.asarray(base_vertices, dtype=np.float64)
    candidate = np.asarray(candidate_vertices, dtype=np.float64)
    objects = np.asarray(object_vertices, dtype=np.float64)
    if base.shape != candidate.shape:
        raise ValueError("base and candidate vertices must align")
    if near_threshold_m <= 0:
        raise ValueError("near_threshold_m must be positive")

    distances, nearest_indices = nearest_object_vertices(base, objects)
    nearest = objects[nearest_indices]
    vectors = base - nearest
    vector_norms = np.linalg.norm(vectors, axis=1)
    valid = vector_norms > 1e-9
    normals = np.zeros_like(vectors)
    normals[valid] = vectors[valid] / vector_norms[valid, None]
    candidate_distances = nearest_object_vertices(candidate, objects)[0]
    displacement = candidate - base
    outward = np.einsum("ij,ij->i", displacement, normals)
    near = distances <= near_threshold_m
    selected = near & valid
    return {
        "base_distance_m": distances,
        "candidate_distance_m": candidate_distances,
        "normals": normals,
        "outward_m": outward,
        "near": near,
        "selected": selected,
    }


def select_projection_vertices(
    base_vertices,
    candidate_vertices,
    object_vertices,
    normal_tolerance_m,
    contact_tolerance_m,
    near_threshold_m=NEAR_THRESHOLD_M,
):
    components = normal_displacement_components(
        base_vertices,
        candidate_vertices,
        object_vertices,
        near_threshold_m,
    )
    outward_excess = components["outward_m"] - normal_tolerance_m
    distance_excess = (
        components["candidate_distance_m"]
        - components["base_distance_m"]
        - contact_tolerance_m
    )
    selected = (
        components["selected"]
        & (outward_excess > 0.0)
        & (distance_excess > 0.0)
    )
    return {
        **components,
        "selected": selected,
        "outward_excess_m": outward_excess,
        "distance_excess_m": distance_excess,
    }


def damped_projection_step(
    jacobian,
    residual,
    damping,
    max_step,
):
    """Solve a damped least-squares finger-pose correction."""
    jacobian = np.asarray(jacobian, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if jacobian.ndim != 2:
        raise ValueError("jacobian must have shape [rows, coefficients]")
    if residual.shape != (jacobian.shape[0],):
        raise ValueError("residual must match jacobian rows")
    if damping < 0 or max_step < 0:
        raise ValueError("damping and max_step must be non-negative")
    if jacobian.shape[0] == 0:
        return np.zeros(jacobian.shape[1], dtype=np.float64)
    normal_matrix = jacobian.T @ jacobian
    normal_matrix += damping * np.eye(jacobian.shape[1])
    rhs = jacobian.T @ residual
    step = np.linalg.solve(normal_matrix, rhs)
    norm = np.linalg.norm(step)
    if max_step == 0:
        return np.zeros_like(step)
    if norm > max_step:
        step *= max_step / norm
    return step


def optimization_weights(arm, args):
    weights = {
        "anchor_weight": args.anchor_weight,
        "pose_anchor_weight": args.pose_anchor_weight,
        "acceleration_weight": args.acceleration_weight,
        "jerk_weight": args.jerk_weight,
        "tail_weight": args.tail_weight,
        "contact_weight": args.contact_weight,
        "normal_weight": 0.0,
    }
    if arm == "strong_contact":
        weights["contact_weight"] = args.strong_contact_weight
    elif arm == "direction_constrained":
        weights["normal_weight"] = args.normal_weight
    elif arm != "unconstrained_continuation":
        raise ValueError(f"{arm} does not use optimization weights")
    return weights


def temporal_rollback_reasons(
    pre_passed,
    post_passed,
    pose_delta_l2,
    max_pose_delta_l2,
):
    reasons = []
    if pose_delta_l2 > max_pose_delta_l2:
        reasons.append("pose_budget")
    if pre_passed and not post_passed:
        reasons.append("temporal_gate")
    return reasons


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_mano_layer(mano_root):
    from manopth.manolayer import ManoLayer

    return ManoLayer(
        mano_root=str(mano_root),
        use_pca=True,
        ncomps=15,
        side="right",
        flat_hand_mean=False,
    ).cuda()


def forward_vertices(pose, beta, transforms, mano_layer):
    import torch

    from contactopt import util

    with torch.no_grad():
        vertices, _ = util.forward_mano(
            mano_layer,
            torch.as_tensor(pose, dtype=torch.float32, device="cuda"),
            torch.as_tensor(beta, dtype=torch.float32, device="cuda"),
            [
                torch.as_tensor(
                    transforms,
                    dtype=torch.float32,
                    device="cuda",
                )
            ],
        )
    return vertices.cpu().numpy()


def project_candidate_pose(
    base_pose,
    candidate_pose,
    beta,
    transforms,
    object_vertices,
    mano_layer,
    normal_tolerance_m,
    contact_tolerance_m,
    max_pose_delta_l2,
    damping,
    max_step,
):
    """Apply one damped normal-direction contact reprojection per frame."""
    import torch

    from contactopt import util

    base_pose = np.asarray(base_pose, dtype=np.float32)
    candidate_pose = np.asarray(
        candidate_pose,
        dtype=np.float32,
    ).copy()
    beta = np.asarray(beta, dtype=np.float32)
    transforms = np.asarray(transforms, dtype=np.float32)
    object_vertices = np.asarray(object_vertices, dtype=np.float32)
    base_vertices = forward_vertices(
        base_pose,
        beta,
        transforms,
        mano_layer,
    )
    candidate_vertices = forward_vertices(
        candidate_pose,
        beta,
        transforms,
        mano_layer,
    )
    stats = {
        "projection_frame_count": 0,
        "projection_selected_vertex_count": 0,
        "projection_before_excess_mm": 0.0,
        "projection_after_excess_mm": 0.0,
        "projection_step_max_l2": 0.0,
    }

    for frame_index in range(len(candidate_pose)):
        selected = select_projection_vertices(
            base_vertices[frame_index],
            candidate_vertices[frame_index],
            object_vertices[frame_index],
            normal_tolerance_m,
            contact_tolerance_m,
        )
        mask = selected["selected"]
        if not np.any(mask):
            continue

        mask_tensor = torch.as_tensor(
            mask,
            dtype=torch.bool,
            device="cuda",
        )
        normals = torch.as_tensor(
            selected["normals"][mask],
            dtype=torch.float32,
            device="cuda",
        )
        base_tensor = torch.as_tensor(
            base_vertices[frame_index][mask],
            dtype=torch.float32,
            device="cuda",
        )
        beta_tensor = torch.as_tensor(
            beta[frame_index],
            dtype=torch.float32,
            device="cuda",
        )[None]
        transform_tensor = torch.as_tensor(
            transforms[frame_index],
            dtype=torch.float32,
            device="cuda",
        )[None]
        delta = torch.zeros(15, dtype=torch.float32, device="cuda")

        def normal_positions(local_delta):
            pose = torch.as_tensor(
                candidate_pose[frame_index],
                dtype=torch.float32,
                device="cuda",
            )[None].clone()
            pose[:, 3:] = pose[:, 3:] + local_delta[None]
            vertices, _ = util.forward_mano(
                mano_layer,
                pose,
                beta_tensor,
                [transform_tensor],
            )
            displacement = vertices[0, mask_tensor] - base_tensor
            return torch.einsum(
                "ij,ij->i",
                displacement,
                normals,
            )

        jacobian = torch.autograd.functional.jacobian(
            normal_positions,
            delta,
            vectorize=True,
        )
        before = normal_positions(delta).detach().cpu().numpy()
        before_excess = np.maximum(before - normal_tolerance_m, 0.0)
        residual = -(
            selected["outward_excess_m"][mask]
        )
        raw_step = damped_projection_step(
            jacobian.detach().cpu().numpy(),
            residual,
            damping,
            max_step,
        )
        step = torch.as_tensor(
            raw_step,
            dtype=torch.float32,
            device="cuda",
        )
        accepted = None
        accepted_excess = None
        accepted_step = None
        for _ in range(4):
            trial_delta = (
                torch.as_tensor(
                    candidate_pose[frame_index, 3:],
                    dtype=torch.float32,
                    device="cuda",
                )
                + step
            )
            if torch.linalg.vector_norm(trial_delta).item() > (
                max_pose_delta_l2
            ):
                step = step * 0.5
                continue
            with torch.no_grad():
                pose = torch.as_tensor(
                    candidate_pose[frame_index],
                    dtype=torch.float32,
                    device="cuda",
                )[None].clone()
                pose[:, 3:] = pose[:, 3:] + step[None]
                vertices, _ = util.forward_mano(
                    mano_layer,
                    pose,
                    beta_tensor,
                    [transform_tensor],
                )
                displacement = vertices[0, mask_tensor] - base_tensor
                trial = torch.einsum(
                    "ij,ij->i",
                    displacement,
                    normals,
                )
            trial_excess = torch.relu(
                trial - normal_tolerance_m
            ).detach().cpu().numpy()
            if (
                np.sum(trial_excess)
                < np.sum(before_excess) * 0.999
            ):
                accepted = trial.detach().cpu().numpy()
                accepted_excess = trial_excess
                accepted_step = step.detach().cpu().numpy()
                break
            step = step * 0.5
        if accepted is None:
            continue

        candidate_pose[frame_index, 3:] += accepted_step
        candidate_vertices[frame_index][mask] = (
            base_vertices[frame_index][mask]
            + torch.as_tensor(
                accepted - before,
                dtype=torch.float32,
                device="cuda",
            ).cpu().numpy()[:, None] * selected["normals"][mask]
        )
        stats["projection_frame_count"] += 1
        stats["projection_selected_vertex_count"] += int(mask.sum())
        stats["projection_before_excess_mm"] += (
            float(np.sum(before_excess)) * 1000.0
        )
        stats["projection_after_excess_mm"] += (
            float(np.sum(accepted_excess)) * 1000.0
        )
        stats["projection_step_max_l2"] = max(
            stats["projection_step_max_l2"],
            float(np.linalg.norm(accepted_step)),
        )
    return candidate_pose, stats


def optimize_segment_custom(
    runs,
    start,
    end,
    mano_layer,
    args,
    base_pose,
    weights,
):
    import torch

    from contactopt import util
    from optimize_contactopt_sequence import (
        sample_object_indices,
        temporal_losses,
    )

    selected = runs[start : end + 1]
    base_pose = torch.as_tensor(
        np.asarray(base_pose[start : end + 1], dtype=np.float32),
        device="cuda",
    )
    beta = torch.as_tensor(
        np.asarray(
            [run["out_ho"].hand_beta for run in selected],
            dtype=np.float32,
        ),
        device="cuda",
    )
    transforms = torch.as_tensor(
        np.asarray(
            [run["out_ho"].hand_mTc for run in selected],
            dtype=np.float32,
        ),
        device="cuda",
    )
    vertices0 = torch.as_tensor(
        np.asarray(
            [run["out_ho"].hand_verts for run in selected],
            dtype=np.float32,
        ),
        device="cuda",
    )
    with torch.no_grad():
        base_vertices, _ = util.forward_mano(
            mano_layer,
            base_pose,
            beta,
            [transforms],
        )
    object_samples = []
    baseline_distances = []
    with torch.no_grad():
        for run_index, run in enumerate(selected):
            object_vertices = np.asarray(
                run["out_ho"].obj_verts,
                dtype=np.float32,
            )
            indices = sample_object_indices(
                object_vertices,
                args.object_samples,
                args.seed + start + run_index,
            )
            object_tensor = torch.as_tensor(
                object_vertices[indices],
                device="cuda",
            )
            object_samples.append(object_tensor)
            baseline_distances.append(
                torch.cdist(
                    vertices0[run_index],
                    object_tensor,
                ).min(dim=1).values
            )
    object_tensor = torch.stack(object_samples)
    baseline_distance = torch.stack(baseline_distances)
    delta = torch.nn.Parameter(
        torch.zeros(len(selected), 15, device="cuda")
    )
    optimizer = torch.optim.Adam([delta], lr=args.lr)

    for _ in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        pose = torch.cat(
            [base_pose[:, :3], base_pose[:, 3:] + delta],
            dim=1,
        )
        vertices, _ = util.forward_mano(
            mano_layer,
            pose,
            beta,
            [transforms],
        )
        anchor = torch.nn.functional.smooth_l1_loss(
            vertices,
            vertices0,
            beta=0.002,
        )
        pose_anchor = delta.square().mean()
        acc_mean, jerk_mean, acc_tail, jerk_tail = temporal_losses(
            vertices
        )
        distances = torch.cdist(
            vertices,
            object_tensor,
        ).min(dim=2).values
        contact_excess = torch.relu(
            distances - baseline_distance - args.contact_margin_m
        )
        contact_loss = contact_excess.square().mean()
        loss = (
            weights["anchor_weight"] * anchor
            + weights["pose_anchor_weight"] * pose_anchor
            + weights["acceleration_weight"] * acc_mean
            + weights["jerk_weight"] * jerk_mean
            + weights["tail_weight"] * (acc_tail + jerk_tail)
            + weights["contact_weight"] * contact_loss
        )
        normal_loss = torch.zeros((), device="cuda")
        if weights["normal_weight"] > 0:
            with torch.no_grad():
                base_distances, nearest = torch.cdist(
                    base_vertices,
                    object_tensor,
                ).min(dim=2)
                nearest_vertices = torch.gather(
                    object_tensor[:, None].expand(
                        -1,
                        base_vertices.shape[1],
                        -1,
                        -1,
                    ),
                    2,
                    nearest[:, :, None, None].expand(
                        -1,
                        -1,
                        -1,
                        3,
                    ),
                ).squeeze(2)
                vectors = base_vertices - nearest_vertices
                norms = torch.linalg.vector_norm(
                    vectors,
                    dim=2,
                    keepdim=True,
                )
                normals = vectors / torch.clamp(norms, min=1e-9)
                near = base_distances <= NEAR_THRESHOLD_M
            displacement = vertices - base_vertices
            outward = torch.einsum(
                "tvi,tvi->tv",
                displacement,
                normals,
            )
            excess_mm = torch.relu(
                outward - args.normal_tolerance_m
            ) * 1000.0
            selected_excess = excess_mm[near]
            if selected_excess.numel():
                normal_loss = torch.nn.functional.smooth_l1_loss(
                    selected_excess,
                    torch.zeros_like(selected_excess),
                    beta=1.0,
                )
            loss = loss + weights["normal_weight"] * normal_loss
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            norms = torch.linalg.vector_norm(
                delta,
                dim=1,
                keepdim=True,
            )
            scales = torch.clamp(
                args.max_pose_delta_l2
                / torch.clamp(norms, min=1e-12),
                max=1.0,
            )
            delta.mul_(scales)

    with torch.no_grad():
        pose = torch.cat(
            [base_pose[:, :3], base_pose[:, 3:] + delta],
            dim=1,
        )
        vertices, _ = util.forward_mano(
            mano_layer,
            pose,
            beta,
            [transforms],
        )
        acc_mean, jerk_mean, acc_tail, jerk_tail = temporal_losses(
            vertices
        )
        normal_excess = torch.zeros((), device="cuda")
        if weights["normal_weight"] > 0:
            base_distances, nearest = torch.cdist(
                base_vertices,
                object_tensor,
            ).min(dim=2)
            nearest_vertices = torch.gather(
                object_tensor[:, None].expand(
                    -1,
                    base_vertices.shape[1],
                    -1,
                    -1,
                ),
                2,
                nearest[:, :, None, None].expand(-1, -1, -1, 3),
            ).squeeze(2)
            vectors = base_vertices - nearest_vertices
            norms = torch.linalg.vector_norm(
                vectors,
                dim=2,
                keepdim=True,
            )
            normals = vectors / torch.clamp(norms, min=1e-9)
            near = base_distances <= NEAR_THRESHOLD_M
            outward = torch.einsum(
                "tvi,tvi->tv",
                vertices - base_vertices,
                normals,
            )
            normal_excess = torch.relu(
                outward[near] - args.normal_tolerance_m
            ).mean() if torch.any(near) else torch.zeros((), device="cuda")
    stats = {
        "start": int(start),
        "end": int(end),
        "frames": int(end - start + 1),
        "pose_delta_abs_mean": float(delta.detach().abs().mean()),
        "pose_delta_max_l2": float(
            torch.linalg.vector_norm(
                delta,
                dim=1,
            ).max()
        ),
        "anchor_vertex_l1_mm": float(
            (vertices - vertices0).abs().mean() * 1000.0
        ),
        "acceleration_mean_mm": float(acc_mean * 1000.0),
        "jerk_mean_mm": float(jerk_mean * 1000.0),
        "acceleration_tail_mm": float(acc_tail * 1000.0),
        "jerk_tail_mm": float(jerk_tail * 1000.0),
        "normal_excess_mean_mm": float(normal_excess * 1000.0),
    }
    return pose.detach().cpu().numpy(), stats


def temporal_gate_flags(
    row,
    runs,
    pose,
    mano_layer,
    thresholds,
):
    from analyze_contactopt_sequence_smoothing import (
        arm_metrics,
        frame_map,
    )

    beta = np.asarray(
        [run["out_ho"].hand_beta for run in runs],
        dtype=np.float32,
    )
    transforms = np.asarray(
        [run["out_ho"].hand_mTc for run in runs],
        dtype=np.float32,
    )
    input_vertices = np.asarray(
        [run["in_ho"].hand_verts for run in runs],
        dtype=np.float32,
    )
    candidate_vertices = forward_vertices(
        pose,
        beta,
        transforms,
        mano_layer,
    )
    chunk_index = frame_map(row["frames"])
    flags = []
    for window in row["windows"]:
        indices = [chunk_index[int(frame)] for frame in window]
        metrics = arm_metrics(
            input_vertices[indices],
            candidate_vertices[indices],
            thresholds,
        )
        flags.append(bool(metrics["temporal_gate"]["passed"]))
    return flags


def load_runs(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "optimized").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "frames").mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    os.chdir(args.contactopt_root)
    if str(args.contactopt_root) not in sys.path:
        sys.path.insert(0, str(args.contactopt_root))

    import torch

    from analyze_contactopt_sequence_smoothing import (
        arm_metrics,
        contact_gate,
    )
    from optimize_contactopt_sequence import (
        aggregate_windows,
        contiguous_segments,
        evaluate_row,
        optimize_segment,
    )
    from smooth_contactopt_temporal import smooth_pose_sequence

    torch.manual_seed(args.seed)
    batch = json.loads(
        args.batch_summary.read_text(encoding="utf-8")
    )
    if batch["completed_chunk_count"] != batch["chunk_count"]:
        raise ValueError("batch summary is incomplete")
    rows = [
        row for row in batch["rows"]
        if row["split"] in set(args.split)
    ]
    if args.sequence:
        selected_sequences = set(args.sequence)
        rows = [
            row for row in rows if row["sequence"] in selected_sequences
        ]
    if args.sequence_limit is not None:
        rows = rows[: args.sequence_limit]
    if (
        set(args.split) == {"train", "dev"}
        and not args.sequence
        and args.sequence_limit is None
    ):
        window_count = sum(len(row["windows"]) for row in rows)
        if len(rows) != 49 or window_count != 376:
            raise ValueError(
                "full train/dev must contain 49 chunks and 376 windows"
            )

    thresholds = {
        "max_speed_ratio": args.max_speed_ratio,
        "max_acceleration_ratio": args.max_acceleration_ratio,
        "max_jerk_ratio": args.max_jerk_ratio,
        "max_wrist_drift_m": args.max_wrist_drift_m,
        "max_object_drift_m": args.max_object_drift_m,
        "max_distance_relative_change": args.max_distance_relative_change,
    }
    mano_layer = build_mano_layer(args.mano_root)
    all_windows = []
    segment_rows = []
    reconstruction_errors = {}

    for row_index, row in enumerate(rows, start=1):
        chunk_id = row["chunk_id"]
        optimized_path = (
            args.output_dir / "optimized" / f"{chunk_id}_optimized.npz"
        )
        frame_path = args.output_dir / "frames" / f"{chunk_id}.json"
        if args.resume and optimized_path.is_file() and frame_path.is_file():
            with np.load(optimized_path, allow_pickle=True) as archive:
                optimized_pose = np.asarray(
                    archive["hand_pose"],
                    dtype=np.float32,
                )
            cached = json.loads(frame_path.read_text(encoding="utf-8"))
            all_windows.extend(cached["windows"])
            segment_rows.extend(cached.get("segments", []))
            reconstruction_errors[chunk_id] = cached.get(
                "mano_reconstruction_max_error_m"
            )
            print(
                f"[{row_index}/{len(rows)}] {chunk_id} (resume)",
                flush=True,
            )
            continue

        runs = load_runs(row["optimized_pkl"])
        with np.load(
            args.e5_root / "optimized" / f"{chunk_id}_optimized.npz"
        ) as archive:
            base_pose = np.asarray(
                archive["hand_pose"],
                dtype=np.float32,
            )
            base_frames = [
                int(frame) for frame in archive["frames"]
            ]
        if base_frames != [int(frame) for frame in row["frames"]]:
            raise ValueError(f"{chunk_id} E5 frame mismatch")
        with np.load(
            args.geometry_dir / f"{chunk_id}.npz",
            allow_pickle=True,
        ) as archive:
            object_vertices = np.asarray(
                archive["object_vertices"],
                dtype=np.float32,
            )
        if object_vertices.shape[0] != len(row["frames"]):
            raise ValueError(f"{chunk_id} geometry frame mismatch")

        optimized_pose = base_pose.copy()
        pre_projection_pose = base_pose.copy()
        projected_segments = set()
        chunk_segments = []
        for segment_index, (start, end) in enumerate(
            contiguous_segments(row["frames"])
        ):
            segment_runs = runs[start : end + 1]
            segment_object_vertices = object_vertices[start : end + 1]
            segment_stats = {
                "segment_index": int(segment_index),
                "start": int(start),
                "end": int(end),
                "frames": int(end - start + 1),
                "projection_applied": False,
                "projection_rollback": [],
            }
            if args.arm == "ordinary_smoothing":
                candidate_segment = smooth_pose_sequence(
                    base_pose[start : end + 1],
                    kernel=(1.0, 4.0, 6.0, 4.0, 1.0),
                )
            elif args.arm == "contact_projection":
                candidate_segment = smooth_pose_sequence(
                    base_pose[start : end + 1],
                    kernel=(1.0, 4.0, 6.0, 4.0, 1.0),
                )
                pre_projection_pose[start : end + 1] = candidate_segment
                candidate_segment, projection_stats = (
                    project_candidate_pose(
                        base_pose[start : end + 1],
                        candidate_segment,
                        np.asarray(
                            [
                                run["out_ho"].hand_beta
                                for run in segment_runs
                            ],
                            dtype=np.float32,
                        ),
                        np.asarray(
                            [
                                run["out_ho"].hand_mTc
                                for run in segment_runs
                            ],
                            dtype=np.float32,
                        ),
                        segment_object_vertices,
                        mano_layer,
                        args.normal_tolerance_m,
                        args.contact_tolerance_m,
                        args.max_pose_delta_l2,
                        args.projection_damping,
                        args.projection_max_step,
                    )
                )
                segment_stats.update(projection_stats)
                segment_stats["projection_applied"] = (
                    projection_stats["projection_frame_count"] > 0
                )
                if segment_stats["projection_applied"]:
                    projected_segments.add(segment_index)
            elif args.arm == "unconstrained_continuation":
                candidate_segment, solve_stats = optimize_segment(
                    runs,
                    start,
                    end,
                    mano_layer,
                    args,
                    initial_pose=base_pose,
                )
                segment_stats.update(solve_stats)
                segment_stats["projection_applied"] = False
                segment_stats["normal_excess_mean_mm"] = None
            else:
                weights = optimization_weights(args.arm, args)
                candidate_segment, solve_stats = (
                    optimize_segment_custom(
                        runs,
                        start,
                        end,
                        mano_layer,
                        args,
                        base_pose,
                        weights,
                    )
                )
                segment_stats.update(solve_stats)
                segment_stats["projection_applied"] = False
                if args.arm == "direction_constrained":
                    pre_projection_pose[start : end + 1] = candidate_segment
                    candidate_segment, projection_stats = (
                        project_candidate_pose(
                            base_pose[start : end + 1],
                            candidate_segment,
                            np.asarray(
                                [
                                    run["out_ho"].hand_beta
                                    for run in segment_runs
                                ],
                                dtype=np.float32,
                            ),
                            np.asarray(
                                [
                                    run["out_ho"].hand_mTc
                                    for run in segment_runs
                                ],
                                dtype=np.float32,
                            ),
                            segment_object_vertices,
                            mano_layer,
                            args.normal_tolerance_m,
                            args.contact_tolerance_m,
                            args.max_pose_delta_l2,
                            args.projection_damping,
                            args.projection_max_step,
                        )
                    )
                    segment_stats.update(projection_stats)
                    segment_stats["projection_applied"] = (
                        projection_stats["projection_frame_count"] > 0
                    )
                    if segment_stats["projection_applied"]:
                        projected_segments.add(segment_index)

            optimized_pose[start : end + 1] = candidate_segment
            delta = (
                optimized_pose[start : end + 1, 3:18]
                - base_pose[start : end + 1, 3:18]
            )
            clipped_delta = clip_pose_delta(
                delta,
                args.max_pose_delta_l2,
            )
            if not np.allclose(delta, clipped_delta):
                optimized_pose[start : end + 1, 3:18] = (
                    base_pose[start : end + 1, 3:18]
                    + clipped_delta
                )
                segment_stats["projection_rollback"].append("pose_budget")
            segment_stats["pose_delta_max_l2"] = float(
                np.linalg.norm(
                    optimized_pose[start : end + 1, 3:18]
                    - base_pose[start : end + 1, 3:18],
                    axis=1,
                ).max()
            )
            chunk_segments.append(segment_stats)

        if projected_segments:
            pre_flags = temporal_gate_flags(
                row,
                runs,
                pre_projection_pose,
                mano_layer,
                thresholds,
            )
            post_flags = temporal_gate_flags(
                row,
                runs,
                optimized_pose,
                mano_layer,
                thresholds,
            )
            frame_to_segment = {}
            for segment_index, (start, end) in enumerate(
                contiguous_segments(row["frames"])
            ):
                for frame_index in range(start, end + 1):
                    frame_to_segment[frame_index] = segment_index
            rollback_segments = set()
            for window_index, window in enumerate(row["windows"]):
                if pre_flags[window_index] and not post_flags[window_index]:
                    chunk_index = {
                        int(frame): index
                        for index, frame in enumerate(row["frames"])
                    }
                    for frame in window:
                        segment_index = frame_to_segment[
                            chunk_index[int(frame)]
                        ]
                        if segment_index in projected_segments:
                            rollback_segments.add(segment_index)
            if rollback_segments:
                for segment_index in rollback_segments:
                    stats = chunk_segments[segment_index]
                    start = stats["start"]
                    end = stats["end"]
                    optimized_pose[start : end + 1] = (
                        pre_projection_pose[start : end + 1]
                    )
                    stats["projection_rollback"].append(
                        "temporal_gate"
                    )
                remaining_flags = temporal_gate_flags(
                    row,
                    runs,
                    optimized_pose,
                    mano_layer,
                    thresholds,
                )
                if any(
                    pre and not post
                    for pre, post in zip(
                        pre_flags,
                        remaining_flags,
                    )
                ):
                    for segment_index in projected_segments:
                        stats = chunk_segments[segment_index]
                        optimized_pose[
                            stats["start"] : stats["end"] + 1
                        ] = pre_projection_pose[
                            stats["start"] : stats["end"] + 1
                        ]
                        stats["projection_rollback"].append(
                            "temporal_gate_all"
                        )

        np.savez_compressed(
            optimized_path,
            hand_pose=optimized_pose,
            frames=np.asarray(row["frames"], dtype=np.int64),
        )
        chunk_windows = evaluate_row(
            row,
            optimized_pose,
            thresholds,
        )
        all_windows.extend(chunk_windows)
        segment_rows.extend(chunk_segments)
        output_pose = np.asarray(
            [run["out_ho"].hand_pose for run in runs],
            dtype=np.float32,
        )
        output_beta = np.asarray(
            [run["out_ho"].hand_beta for run in runs],
            dtype=np.float32,
        )
        output_transforms = np.asarray(
            [run["out_ho"].hand_mTc for run in runs],
            dtype=np.float32,
        )
        reconstructed = forward_vertices(
            output_pose,
            output_beta,
            output_transforms,
            mano_layer,
        )
        expected = np.asarray(
            [run["out_ho"].hand_verts for run in runs],
            dtype=np.float32,
        )
        reconstruction_error = float(
            np.max(np.abs(reconstructed - expected))
        )
        if reconstruction_error > 1e-6:
            raise ValueError(
                f"{chunk_id} MANO reconstruction error "
                f"{reconstruction_error}"
            )
        reconstruction_errors[chunk_id] = reconstruction_error
        write_json(
            frame_path,
            {
                "chunk_id": chunk_id,
                "sequence": row["sequence"],
                "split": row["split"],
                "object_name": row["object_name"],
                "frames": [int(frame) for frame in row["frames"]],
                "segments": chunk_segments,
                "windows": chunk_windows,
            },
        )
        print(
            f"[{row_index}/{len(rows)}] {chunk_id}",
            flush=True,
        )

    summary = {
        "method": (
            "optimize ContactOpt finger PCA coefficients 3:18 from frozen "
            "E5 poses with matched temporal and pose budgets; "
            "direction_constrained adds near-surface outward penalties and "
            "one damped normal contact reprojection"
        ),
        "arm": args.arm,
        "splits": sorted(set(args.split)),
        "sequence_filter": args.sequence,
        "sequence_limit": args.sequence_limit,
        "inputs": {
            "batch_summary": str(args.batch_summary),
            "batch_summary_sha256": sha256_file(args.batch_summary),
            "e5_result": str(args.e5_root / "result.json"),
            "e5_result_sha256": sha256_file(
                args.e5_root / "result.json"
            ),
            "e5t_result": str(args.e5t_root / "result.json"),
            "e5t_result_sha256": sha256_file(
                args.e5t_root / "result.json"
            ),
            "geometry_dir": str(args.geometry_dir),
        },
        "args": {
            "iterations": args.iterations,
            "lr": args.lr,
            "object_samples": args.object_samples,
            "anchor_weight": args.anchor_weight,
            "pose_anchor_weight": args.pose_anchor_weight,
            "acceleration_weight": args.acceleration_weight,
            "jerk_weight": args.jerk_weight,
            "tail_weight": args.tail_weight,
            "contact_weight": args.contact_weight,
            "contact_margin_m": args.contact_margin_m,
            "normal_weight": args.normal_weight,
            "strong_contact_weight": args.strong_contact_weight,
            "normal_tolerance_m": args.normal_tolerance_m,
            "contact_tolerance_m": args.contact_tolerance_m,
            "max_pose_delta_l2": args.max_pose_delta_l2,
            "projection_damping": args.projection_damping,
            "projection_max_step": args.projection_max_step,
            "seed": args.seed,
        },
        "thresholds": thresholds,
        "near_threshold_m": NEAR_THRESHOLD_M,
        "chunk_count": len(rows),
        "window_count": len(all_windows),
        "aggregate": aggregate_windows(all_windows),
        "segments": segment_rows,
        "windows": all_windows,
        "test_split": "untouched",
        "decision": "RUN_COMPLETE",
    }
    write_json(args.output_dir / "result.json", summary)
    print(
        json.dumps(
            {
                "arm": args.arm,
                "aggregate": summary["aggregate"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
