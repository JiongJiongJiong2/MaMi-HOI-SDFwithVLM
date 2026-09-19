#!/usr/bin/env python3
"""Compute per-window contact and temporal metrics for ContactOpt outputs."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
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


def trajectory_stat(values):
    values = np.asarray(values, dtype=np.float64)
    norm = lambda array: np.linalg.norm(array, axis=-1)
    speed = norm(np.diff(values, axis=0)).mean() if len(values) >= 2 else 0.0
    acceleration = (
        norm(np.diff(values, n=2, axis=0)).mean()
        if len(values) >= 3
        else 0.0
    )
    jerk = (
        norm(np.diff(values, n=3, axis=0)).mean()
        if len(values) >= 4
        else 0.0
    )
    return {
        "speed_mm": float(speed * 1000.0),
        "acceleration_mm": float(acceleration * 1000.0),
        "jerk_mm": float(jerk * 1000.0),
    }


def ratio(value, baseline):
    if baseline <= 0:
        return None
    return float(value / baseline)


def threshold_pass(value, maximum):
    return value is None or value <= maximum


def frame_map(values):
    return {int(frame): index for index, frame in enumerate(values)}


def summarize_window(
    geometry,
    case,
    row,
    window,
    thresholds,
):
    geometry_frames = geometry["frames"]
    geometry_index = frame_map(geometry_frames)
    missing = [int(frame) for frame in window if int(frame) not in geometry_index]
    if missing:
        raise ValueError(
            f"Window frames are absent from geometry: {missing}"
        )
    indices = [geometry_index[int(frame)] for frame in window]
    input_vertices = np.asarray(
        geometry["input_hand_vertices"], dtype=np.float64
    )[indices]
    refined_vertices = np.asarray(
        geometry["refined_hand_vertices"], dtype=np.float64
    )[indices]
    input_stats = trajectory_stat(input_vertices)
    refined_stats = trajectory_stat(refined_vertices)
    speed_ratio = ratio(
        refined_stats["speed_mm"],
        input_stats["speed_mm"],
    )
    acceleration_ratio = ratio(
        refined_stats["acceleration_mm"],
        input_stats["acceleration_mm"],
    )
    jerk_ratio = ratio(
        refined_stats["jerk_mm"],
        input_stats["jerk_mm"],
    )
    temporal_gate = {
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
    temporal_gate["passed"] = all(temporal_gate.values())

    eligible_index = frame_map(case["eligible_frames"])
    rows = case["rows"]
    window_rows = []
    for frame in window:
        row_index = eligible_index.get(int(frame))
        if row_index is None:
            raise ValueError(f"Eligible frame {frame} is missing from case JSON")
        window_rows.append(rows[row_index])

    required_improved = max(
        3,
        int(np.ceil(0.7 * len(window_rows))),
    )
    contact_improved = sum(
        item["hand_contact_relative_change"] is not None
        and item["hand_contact_relative_change"] > 0
        for item in window_rows
    )
    distance_improved = sum(
        item["distance_relative_change"] < 0 for item in window_rows
    )
    mean_distance_change = float(
        np.mean([item["distance_relative_change"] for item in window_rows])
    )
    max_wrist_drift = max(
        item["wrist_root_drift_m"] for item in window_rows
    )
    max_object_drift = max(
        item["object_vertex_drift_m"] for item in window_rows
    )
    contact_gate = {
        "contact_improved": contact_improved >= required_improved,
        "distance_not_regressed": (
            mean_distance_change
            <= thresholds["max_distance_relative_change"]
        ),
        "wrist_frozen": max_wrist_drift <= thresholds["max_wrist_drift_m"],
        "object_frozen": max_object_drift <= thresholds["max_object_drift_m"],
    }
    contact_gate["passed"] = all(contact_gate.values())

    return {
        "chunk_id": row["chunk_id"],
        "sequence": row["sequence"],
        "split": row["split"],
        "object_name": row["object_name"],
        "frames": [int(frame) for frame in window],
        "input_trajectory": input_stats,
        "refined_trajectory": refined_stats,
        "speed_ratio": speed_ratio,
        "acceleration_ratio": acceleration_ratio,
        "jerk_ratio": jerk_ratio,
        "contact_improved_frames": int(contact_improved),
        "distance_improved_frames": int(distance_improved),
        "required_improved_frames": required_improved,
        "mean_distance_relative_change": mean_distance_change,
        "max_wrist_drift_m": float(max_wrist_drift),
        "max_object_drift_m": float(max_object_drift),
        "temporal_gate": temporal_gate,
        "contact_gate": contact_gate,
        "combined_gate": {
            "passed": temporal_gate["passed"] and contact_gate["passed"]
        },
    }


def finite_values(rows, key):
    return [row[key] for row in rows if row[key] is not None]


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
            "temporal_pass_count": sum(
                row["temporal_gate"]["passed"] for row in group
            ),
            "contact_pass_count": sum(
                row["contact_gate"]["passed"] for row in group
            ),
            "combined_pass_count": sum(
                row["combined_gate"]["passed"] for row in group
            ),
            "speed_ratio_mean": (
                float(np.mean(finite_values(group, "speed_ratio")))
                if finite_values(group, "speed_ratio")
                else None
            ),
            "acceleration_ratio_mean": (
                float(np.mean(finite_values(group, "acceleration_ratio")))
                if finite_values(group, "acceleration_ratio")
                else None
            ),
            "jerk_ratio_mean": (
                float(np.mean(finite_values(group, "jerk_ratio")))
                if finite_values(group, "jerk_ratio")
                else None
            ),
            "jerk_ratio_max": (
                float(np.max(finite_values(group, "jerk_ratio")))
                if finite_values(group, "jerk_ratio")
                else None
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

    windows = []
    for row in batch["rows"]:
        geometry_path = args.geometry_dir / f"{row['chunk_id']}.npz"
        case_path = Path(row["output_json"])
        if not geometry_path.is_file():
            raise FileNotFoundError(geometry_path)
        if not case_path.is_file():
            raise FileNotFoundError(case_path)
        with np.load(geometry_path, allow_pickle=True) as geometry:
            geometry_data = {
                key: np.asarray(geometry[key])
                for key in geometry.files
            }
        case = json.loads(case_path.read_text(encoding="utf-8"))
        for window in row["windows"]:
            windows.append(
                summarize_window(
                    geometry_data,
                    case,
                    row,
                    window,
                    thresholds,
                )
            )

    result = {
        "batch_summary": str(args.batch_summary),
        "geometry_dir": str(args.geometry_dir),
        "thresholds": thresholds,
        "window_count": len(windows),
        "aggregate": aggregate_rows(windows),
        "windows": windows,
        "decision": (
            "GO" if all(
                row["combined_gate"]["passed"] for row in windows
            ) else "NO-GO"
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
