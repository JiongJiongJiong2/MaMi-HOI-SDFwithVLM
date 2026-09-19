#!/usr/bin/env python3
"""Evaluate fixed pose smoothing over the sequence ContactOpt dataset."""

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np

from analyze_contactopt_sequence_temporal import (
    frame_map,
    threshold_pass,
    trajectory_stat,
)
from smooth_contactopt_temporal import (
    contact_metrics,
    smooth_pose_sequence,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--kernel",
        choices=("binomial3", "binomial5"),
        default="binomial5",
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
    return parser.parse_args()


KERNELS = {
    "binomial3": (1.0, 2.0, 1.0),
    "binomial5": (1.0, 4.0, 6.0, 4.0, 1.0),
}


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


def ratio(value, baseline):
    if baseline <= 0:
        return None
    return float(value / baseline)


def arm_metrics(input_vertices, candidate_vertices, thresholds):
    input_stats = trajectory_stat(input_vertices)
    candidate_stats = trajectory_stat(candidate_vertices)
    speed_ratio = ratio(
        candidate_stats["speed_mm"],
        input_stats["speed_mm"],
    )
    acceleration_ratio = ratio(
        candidate_stats["acceleration_mm"],
        input_stats["acceleration_mm"],
    )
    jerk_ratio = ratio(
        candidate_stats["jerk_mm"],
        input_stats["jerk_mm"],
    )
    gate = {
        "speed_ratio": threshold_pass(
            speed_ratio,
            thresholds["max_speed_ratio"],
        ),
        "acceleration_ratio": threshold_pass(
            acceleration_ratio,
            thresholds["max_acceleration_ratio"],
        ),
        "jerk_ratio": threshold_pass(
            jerk_ratio,
            thresholds["max_jerk_ratio"],
        ),
    }
    gate["passed"] = all(gate.values())
    return {
        "input_trajectory": input_stats,
        "candidate_trajectory": candidate_stats,
        "speed_ratio": speed_ratio,
        "acceleration_ratio": acceleration_ratio,
        "jerk_ratio": jerk_ratio,
        "temporal_gate": gate,
    }


def contact_gate(contact_rows, thresholds):
    required = max(3, int(np.ceil(0.7 * len(contact_rows))))
    improved = sum(
        (
            row.get(
                "contact_relative_change",
                row.get("hand_contact_relative_change"),
            )
            is not None
        )
        and row.get(
            "contact_relative_change",
            row.get("hand_contact_relative_change"),
        )
        > 0
        for row in contact_rows
    )
    distance_changes = [
        row["distance_relative_change"] for row in contact_rows
    ]
    mean_distance_change = float(np.mean(distance_changes))
    max_wrist_drift = max(
        row["wrist_root_drift_m"] for row in contact_rows
    )
    max_object_drift = max(
        row["object_vertex_drift_m"] for row in contact_rows
    )
    result = {
        "contact_improved": improved >= required,
        "distance_not_regressed": (
            mean_distance_change
            <= thresholds["max_distance_relative_change"]
        ),
        "wrist_frozen": max_wrist_drift <= thresholds["max_wrist_drift_m"],
        "object_frozen": max_object_drift <= thresholds["max_object_drift_m"],
        "contact_improved_frames": int(improved),
        "required_improved_frames": required,
        "mean_distance_relative_change": mean_distance_change,
        "max_wrist_drift_m": float(max_wrist_drift),
        "max_object_drift_m": float(max_object_drift),
    }
    result["passed"] = all(
        result[key]
        for key in (
            "contact_improved",
            "distance_not_regressed",
            "wrist_frozen",
            "object_frozen",
        )
    )
    return result


def aggregate_rows(rows):
    groups = {}
    for split in ("train", "dev"):
        groups[f"split:{split}"] = [
            row for row in rows if row["split"] == split
        ]
    for object_name in sorted({row["object_name"] for row in rows}):
        groups[f"object:{object_name}"] = [
            row for row in rows if row["object_name"] == object_name
        ]
    groups["all"] = rows

    result = {}
    for name, group in groups.items():
        result[name] = {
            "window_count": len(group),
            "smoothed_temporal_pass_count": sum(
                row["smoothed_temporal_gate"]["passed"] for row in group
            ),
            "smoothed_contact_pass_count": sum(
                row["smoothed_contact_gate"]["passed"] for row in group
            ),
            "smoothed_combined_pass_count": sum(
                row["smoothed_combined_pass"] for row in group
            ),
            "smoothed_speed_ratio_mean": float(
                np.mean(
                    [
                        row["smoothed_speed_ratio"]
                        for row in group
                        if row["smoothed_speed_ratio"] is not None
                    ]
                )
            ),
            "smoothed_acceleration_ratio_mean": float(
                np.mean(
                    [
                        row["smoothed_acceleration_ratio"]
                        for row in group
                        if row["smoothed_acceleration_ratio"] is not None
                    ]
                )
            ),
            "smoothed_jerk_ratio_mean": float(
                np.mean(
                    [
                        row["smoothed_jerk_ratio"]
                        for row in group
                        if row["smoothed_jerk_ratio"] is not None
                    ]
                )
            ),
        }
    return result


def main():
    args = parse_args()
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
    kernel = KERNELS[args.kernel]
    windows = []

    for row_index, row in enumerate(batch["rows"], start=1):
        optimized_path = Path(row["optimized_pkl"])
        with optimized_path.open("rb") as handle:
            runs = pickle.load(handle)
        output_pose = np.asarray(
            [run["out_ho"].hand_pose for run in runs],
            dtype=np.float32,
        )
        smoothed_pose = output_pose.copy()
        for start, end in contiguous_segments(row["frames"]):
            smoothed_pose[start : end + 1] = smooth_pose_sequence(
                output_pose[start : end + 1],
                kernel=kernel,
            )
        input_vertices = np.asarray(
            [run["in_ho"].hand_verts for run in runs],
            dtype=np.float32,
        )
        raw_vertices = np.asarray(
            [run["out_ho"].hand_verts for run in runs],
            dtype=np.float32,
        )

        smoothed_hands = []
        smoothed_vertices = []
        smoothed_contact_rows = []
        for run, pose in zip(runs, smoothed_pose):
            input_hand = run["in_ho"]
            smoothed_hand = copy.deepcopy(input_hand)
            smoothed_hand.hand_pose = pose
            smoothed_hand.run_mano()
            smoothed_hands.append(smoothed_hand)
            smoothed_vertices.append(smoothed_hand.hand_verts)
            smoothed_contact_rows.append(
                contact_metrics(input_hand, smoothed_hand)
            )
        smoothed_vertices = np.asarray(smoothed_vertices, dtype=np.float32)

        case = json.loads(
            Path(row["output_json"]).read_text(encoding="utf-8")
        )
        eligible_index = frame_map(case["eligible_frames"])
        chunk_index = frame_map(row["frames"])
        for window in row["windows"]:
            indices = [chunk_index[int(frame)] for frame in window]
            raw_arm = arm_metrics(
                input_vertices[indices],
                raw_vertices[indices],
                thresholds,
            )
            smoothed_arm = arm_metrics(
                input_vertices[indices],
                smoothed_vertices[indices],
                thresholds,
            )
            raw_contact = contact_gate(
                [
                    case["rows"][eligible_index[int(frame)]]
                    for frame in window
                ],
                thresholds,
            )
            smoothed_contact = contact_gate(
                [smoothed_contact_rows[index] for index in indices],
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
                    "smoothed_speed_ratio": smoothed_arm["speed_ratio"],
                    "smoothed_acceleration_ratio": smoothed_arm[
                        "acceleration_ratio"
                    ],
                    "smoothed_jerk_ratio": smoothed_arm["jerk_ratio"],
                    "smoothed_temporal_gate": smoothed_arm["temporal_gate"],
                    "smoothed_contact_gate": smoothed_contact,
                    "smoothed_combined_pass": (
                        smoothed_arm["temporal_gate"]["passed"]
                        and smoothed_contact["passed"]
                    ),
                }
            )
        print(
            f"[{row_index}/{len(batch['rows'])}] {row['chunk_id']}",
            flush=True,
        )

    result = {
        "batch_summary": str(args.batch_summary),
        "kernel": args.kernel,
        "kernel_values": list(kernel),
        "thresholds": thresholds,
        "window_count": len(windows),
        "aggregate": aggregate_rows(windows),
        "windows": windows,
        "decision": (
            "GO"
            if all(row["smoothed_combined_pass"] for row in windows)
            else "NO-GO"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "kernel": args.kernel,
                "window_count": result["window_count"],
                "aggregate": result["aggregate"],
                "decision": result["decision"],
                "output_json": str(args.output_json),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
