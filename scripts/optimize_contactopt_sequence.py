#!/usr/bin/env python3
"""Optimize ContactOpt pose sequences with fidelity, contact, and tail losses."""

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from analyze_contactopt_sequence_smoothing import (
    arm_metrics,
    contact_gate,
)
from smooth_contactopt_temporal import (
    contact_metrics,
    smooth_pose_sequence,
    trajectory_stat,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
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
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--object-samples", type=int, default=4096)
    parser.add_argument("--anchor-weight", type=float, default=0.2)
    parser.add_argument("--pose-anchor-weight", type=float, default=0.05)
    parser.add_argument("--acceleration-weight", type=float, default=1.0)
    parser.add_argument("--jerk-weight", type=float, default=1.0)
    parser.add_argument("--tail-weight", type=float, default=2.0)
    parser.add_argument("--contact-weight", type=float, default=10.0)
    parser.add_argument("--contact-margin-m", type=float, default=0.001)
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
    parser.add_argument(
        "--initial-smoothing",
        choices=("none", "binomial5"),
        default="binomial5",
    )
    return parser.parse_args()


def contiguous_segments(frames):
    frames = sorted(int(frame) for frame in frames)
    segments = []
    start = 0
    for index in range(1, len(frames)):
        if frames[index] != frames[index - 1] + 1:
            segments.append((start, index - 1))
            start = index
    if frames:
        segments.append((start, len(frames) - 1))
    return segments


def top_fraction_mean(values, fraction):
    if values.numel() == 0:
        return values.new_zeros(())
    count = max(1, int(np.ceil(fraction * values.numel())))
    return torch.topk(values.reshape(-1), count).values.mean()


def temporal_losses(vertices):
    if len(vertices) < 3:
        zero = vertices.new_zeros(())
        return zero, zero, zero, zero
    acceleration = torch.linalg.vector_norm(
        vertices[2:] - 2.0 * vertices[1:-1] + vertices[:-2],
        dim=2,
    )
    acc_mean = acceleration.mean()
    acc_tail = top_fraction_mean(acceleration, 0.10)
    if len(vertices) < 4:
        return acc_mean, vertices.new_zeros(()), acc_tail, vertices.new_zeros(())
    jerk = torch.linalg.vector_norm(
        vertices[3:]
        - 3.0 * vertices[2:-1]
        + 3.0 * vertices[1:-2]
        - vertices[:-3],
        dim=2,
    )
    return acc_mean, jerk.mean(), acc_tail, top_fraction_mean(jerk, 0.10)


def sample_object_indices(object_vertices, count, seed):
    if len(object_vertices) <= count:
        return np.arange(len(object_vertices), dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(len(object_vertices), count, replace=False))


def build_mano_layer(contactopt_root):
    if str(contactopt_root) not in sys.path:
        sys.path.insert(0, str(contactopt_root))
    from manopth.manolayer import ManoLayer

    return ManoLayer(
        mano_root="mano/models",
        use_pca=True,
        ncomps=15,
        side="right",
        flat_hand_mean=False,
    ).cuda()


def optimize_segment(
    runs,
    start,
    end,
    mano_layer,
    args,
):
    from contactopt import util

    selected = runs[start : end + 1]
    pose0 = torch.as_tensor(
        np.asarray(
            [run["out_ho"].hand_pose for run in selected],
            dtype=np.float32,
        ),
        device="cuda",
    )
    if args.initial_smoothing == "binomial5":
        base_pose_np = smooth_pose_sequence(
            pose0.detach().cpu().numpy(),
            kernel=(1.0, 4.0, 6.0, 4.0, 1.0),
        )
        base_pose = torch.as_tensor(
            base_pose_np, device="cuda"
        )
    else:
        base_pose = pose0
    beta = torch.as_tensor(
        np.asarray(
            [run["out_ho"].hand_beta for run in selected],
            dtype=np.float32,
        ),
        device="cuda",
    )
    tforms = [
        torch.as_tensor(
            np.asarray(
                [run["out_ho"].hand_mTc for run in selected],
                dtype=np.float32,
            ),
            device="cuda",
        )
    ]
    vertices0 = torch.as_tensor(
        np.asarray(
            [run["out_ho"].hand_verts for run in selected],
            dtype=np.float32,
        ),
        device="cuda",
    )

    object_samples = []
    baseline_distances = []
    with torch.no_grad():
        for run_index, run in enumerate(selected):
            object_vertices = np.asarray(
                run["out_ho"].obj_verts, dtype=np.float32
            )
            indices = sample_object_indices(
                object_vertices,
                args.object_samples,
                args.seed + start + run_index,
            )
            object_tensor = torch.as_tensor(
                object_vertices[indices], device="cuda"
            )
            object_samples.append(object_tensor)
            baseline_distances.append(
                torch.cdist(vertices0[run_index], object_tensor).min(dim=1).values
            )
    object_tensor = torch.stack(object_samples)
    baseline_distance = torch.stack(baseline_distances)

    delta = torch.nn.Parameter(
        torch.zeros(len(selected), 15, device="cuda")
    )
    optimizer = torch.optim.Adam([delta], lr=args.lr)

    for iteration in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        pose = torch.cat(
            [base_pose[:, :3], base_pose[:, 3:] + delta], dim=1
        )
        vertices, _ = util.forward_mano(mano_layer, pose, beta, tforms)
        anchor = torch.nn.functional.smooth_l1_loss(
            vertices, vertices0, beta=0.002
        )
        pose_anchor = delta.square().mean()
        acc_mean, jerk_mean, acc_tail, jerk_tail = temporal_losses(vertices)
        distances = torch.cdist(vertices, object_tensor).min(dim=2).values
        contact_excess = torch.relu(
            distances - baseline_distance - args.contact_margin_m
        )
        contact_loss = contact_excess.square().mean()
        loss = (
            args.anchor_weight * anchor
            + args.pose_anchor_weight * pose_anchor
            + args.acceleration_weight * acc_mean
            + args.jerk_weight * jerk_mean
            + args.tail_weight * (acc_tail + jerk_tail)
            + args.contact_weight * contact_loss
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        pose = torch.cat(
            [base_pose[:, :3], base_pose[:, 3:] + delta], dim=1
        )
        vertices, _ = util.forward_mano(mano_layer, pose, beta, tforms)
        acc_mean, jerk_mean, acc_tail, jerk_tail = temporal_losses(vertices)
    return pose.detach().cpu().numpy(), {
        "start": int(start),
        "end": int(end),
        "frames": int(end - start + 1),
        "pose_delta_abs_mean": float(delta.detach().abs().mean()),
        "anchor_vertex_l1_mm": float(
            (vertices - vertices0).abs().mean() * 1000.0
        ),
        "acceleration_mean_mm": float(acc_mean * 1000.0),
        "jerk_mean_mm": float(jerk_mean * 1000.0),
        "acceleration_tail_mm": float(acc_tail * 1000.0),
        "jerk_tail_mm": float(jerk_tail * 1000.0),
    }


def evaluate_row(row, optimized_pose, thresholds):
    with Path(row["optimized_pkl"]).open("rb") as handle:
        runs = pickle.load(handle)
    case = json.loads(
        Path(row["output_json"]).read_text(encoding="utf-8")
    )
    from analyze_contactopt_sequence_smoothing import frame_map

    eligible_index = frame_map(case["eligible_frames"])
    chunk_index = frame_map(row["frames"])
    input_vertices = np.asarray(
        [run["in_ho"].hand_verts for run in runs], dtype=np.float32
    )
    raw_vertices = np.asarray(
        [run["out_ho"].hand_verts for run in runs], dtype=np.float32
    )

    optimized_vertices = []
    optimized_contact_rows = []
    for run, pose in zip(runs, optimized_pose):
        input_hand = run["in_ho"]
        optimized_hand = copy.deepcopy(input_hand)
        optimized_hand.hand_pose = pose
        optimized_hand.run_mano()
        optimized_vertices.append(optimized_hand.hand_verts)
        optimized_contact_rows.append(
            contact_metrics(input_hand, optimized_hand)
        )
    optimized_vertices = np.asarray(optimized_vertices, dtype=np.float32)

    windows = []
    for window in row["windows"]:
        indices = [chunk_index[int(frame)] for frame in window]
        raw_arm = arm_metrics(
            input_vertices[indices],
            raw_vertices[indices],
            thresholds,
        )
        optimized_arm = arm_metrics(
            input_vertices[indices],
            optimized_vertices[indices],
            thresholds,
        )
        raw_contact = contact_gate(
            [
                case["rows"][eligible_index[int(frame)]]
                for frame in window
            ],
            thresholds,
        )
        optimized_contact = contact_gate(
            [optimized_contact_rows[index] for index in indices],
            thresholds,
        )
        windows.append(
            {
                "chunk_id": row["chunk_id"],
                "sequence": row["sequence"],
                "split": row["split"],
                "object_name": row["object_name"],
                "frames": [int(frame) for frame in window],
                "raw_temporal_gate": raw_arm["temporal_gate"],
                "raw_contact_gate": raw_contact,
                "raw_combined_pass": (
                    raw_arm["temporal_gate"]["passed"]
                    and raw_contact["passed"]
                ),
                "optimized_speed_ratio": optimized_arm["speed_ratio"],
                "optimized_acceleration_ratio": optimized_arm[
                    "acceleration_ratio"
                ],
                "optimized_jerk_ratio": optimized_arm["jerk_ratio"],
                "optimized_temporal_gate": optimized_arm["temporal_gate"],
                "optimized_contact_gate": optimized_contact,
                "optimized_combined_pass": (
                    optimized_arm["temporal_gate"]["passed"]
                    and optimized_contact["passed"]
                ),
            }
        )
    return windows


def aggregate_windows(windows):
    groups = {}
    for split in ("train", "dev"):
        groups[f"split:{split}"] = [
            row for row in windows if row["split"] == split
        ]
    for object_name in sorted({row["object_name"] for row in windows}):
        groups[f"object:{object_name}"] = [
            row for row in windows if row["object_name"] == object_name
        ]
    groups["all"] = windows
    result = {}
    for name, group in groups.items():
        result[name] = {
            "window_count": len(group),
            "raw_combined_pass_count": sum(
                row["raw_combined_pass"] for row in group
            ),
            "optimized_temporal_pass_count": sum(
                row["optimized_temporal_gate"]["passed"] for row in group
            ),
            "optimized_contact_pass_count": sum(
                row["optimized_contact_gate"]["passed"] for row in group
            ),
            "optimized_combined_pass_count": sum(
                row["optimized_combined_pass"] for row in group
            ),
            "optimized_acceleration_ratio_mean": float(
                np.mean(
                    [
                        row["optimized_acceleration_ratio"]
                        for row in group
                        if row["optimized_acceleration_ratio"] is not None
                    ]
                )
            ),
            "optimized_jerk_ratio_mean": float(
                np.mean(
                    [
                        row["optimized_jerk_ratio"]
                        for row in group
                        if row["optimized_jerk_ratio"] is not None
                    ]
                )
            ),
        }
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    os.chdir(args.contactopt_root)
    if str(args.contactopt_root) not in sys.path:
        sys.path.insert(0, str(args.contactopt_root))
    torch.manual_seed(args.seed)

    batch = json.loads(
        args.batch_summary.read_text(encoding="utf-8")
    )
    if batch["completed_chunk_count"] != batch["chunk_count"]:
        raise ValueError("Batch summary is incomplete")
    thresholds = {
        "max_speed_ratio": args.max_speed_ratio,
        "max_acceleration_ratio": args.max_acceleration_ratio,
        "max_jerk_ratio": args.max_jerk_ratio,
        "max_wrist_drift_m": args.max_wrist_drift_m,
        "max_object_drift_m": args.max_object_drift_m,
        "max_distance_relative_change": args.max_distance_relative_change,
    }
    rows = [
        row
        for row in batch["rows"]
        if row["split"] in set(args.split)
    ]
    if args.sequence:
        selected_sequences = set(args.sequence)
        rows = [
            row for row in rows if row["sequence"] in selected_sequences
        ]
    if args.sequence_limit is not None:
        rows = rows[: args.sequence_limit]

    mano_layer = build_mano_layer(args.contactopt_root)
    all_windows = []
    segment_rows = []
    for row_index, row in enumerate(rows, start=1):
        with Path(row["optimized_pkl"]).open("rb") as handle:
            runs = pickle.load(handle)
        optimized_pose = np.asarray(
            [run["out_ho"].hand_pose for run in runs],
            dtype=np.float32,
        )
        for start, end in contiguous_segments(row["frames"]):
            segment_pose, segment_stats = optimize_segment(
                runs,
                start,
                end,
                mano_layer,
                args,
            )
            optimized_pose[start : end + 1] = segment_pose
            segment_stats.update(
                {
                    "chunk_id": row["chunk_id"],
                    "sequence": row["sequence"],
                    "split": row["split"],
                    "object_name": row["object_name"],
                }
            )
            segment_rows.append(segment_stats)
        np.savez_compressed(
            args.output_dir / f"{row['chunk_id']}_optimized.npz",
            hand_pose=optimized_pose,
            frames=np.asarray(row["frames"], dtype=np.int64),
        )
        all_windows.extend(
            evaluate_row(row, optimized_pose, thresholds)
        )
        print(
            f"[{row_index}/{len(rows)}] {row['chunk_id']}",
            flush=True,
        )

    summary = {
        "batch_summary": str(args.batch_summary),
        "splits": list(args.split),
        "sequence_limit": args.sequence_limit,
        "sequence_filter": args.sequence,
        "method": (
            "optimize ContactOpt PCA finger coefficients with fidelity, "
            "contact-distance preservation, acceleration/jerk tail losses; "
            "global pose and hand_mTc remain frozen"
        ),
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
            "initial_smoothing": args.initial_smoothing,
            "seed": args.seed,
        },
        "thresholds": thresholds,
        "chunk_count": len(rows),
        "window_count": len(all_windows),
        "aggregate": aggregate_windows(all_windows),
        "segments": segment_rows,
        "windows": all_windows,
    }
    summary["decision"] = (
        "GO"
        if all(row["optimized_combined_pass"] for row in all_windows)
        else "NO-GO"
    )
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "aggregate": summary["aggregate"],
                "output_json": str(args.output_json),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
