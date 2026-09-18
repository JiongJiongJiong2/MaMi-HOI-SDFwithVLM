#!/usr/bin/env python3
"""Evaluate temporal smoothness of independently refined ContactOpt frames."""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--case-json-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-speed-ratio", type=float, default=2.0)
    parser.add_argument("--max-acceleration-ratio", type=float, default=2.0)
    parser.add_argument("--max-jerk-ratio", type=float, default=3.0)
    return parser.parse_args()


def trajectory_stat(values):
    norm = lambda array: np.linalg.norm(array, axis=-1)
    if len(values) < 2:
        return {"speed_mm": 0.0, "acceleration_mm": 0.0, "jerk_mm": 0.0}
    speed = norm(np.diff(values, axis=0)).mean()
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


def ratio(refined, baseline):
    if baseline <= 0:
        return None
    return float(refined / baseline)


def main():
    args = parse_args()
    cases = []
    for geometry_path in sorted(args.geometry_dir.glob("*.npz")):
        sequence = geometry_path.stem
        case_path = args.case_json_dir / f"{sequence}.json"
        case = json.loads(case_path.read_text(encoding="utf-8"))
        with np.load(geometry_path, allow_pickle=True) as geometry:
            input_vertices = np.asarray(
                geometry["input_hand_vertices"], dtype=np.float64
            )
            refined_vertices = np.asarray(
                geometry["refined_hand_vertices"], dtype=np.float64
            )
            frames = np.asarray(geometry["frames"], dtype=np.int64)

        input_stats = trajectory_stat(input_vertices)
        refined_stats = trajectory_stat(refined_vertices)
        speed_ratio = ratio(
            refined_stats["speed_mm"], input_stats["speed_mm"]
        )
        acceleration_ratio = ratio(
            refined_stats["acceleration_mm"],
            input_stats["acceleration_mm"],
        )
        jerk_ratio = ratio(
            refined_stats["jerk_mm"], input_stats["jerk_mm"]
        )
        frame_motion = np.linalg.norm(
            refined_vertices - input_vertices, axis=2
        ).mean(axis=1)
        temporal_gate = {
            "speed_ratio": speed_ratio is None
            or speed_ratio <= args.max_speed_ratio,
            "acceleration_ratio": acceleration_ratio is None
            or acceleration_ratio <= args.max_acceleration_ratio,
            "jerk_ratio": jerk_ratio is None
            or jerk_ratio <= args.max_jerk_ratio,
        }
        temporal_gate["passed"] = all(temporal_gate.values())
        cases.append(
            {
                "sequence": sequence,
                "object_name": case["object_name"],
                "frames": frames.tolist(),
                "input_trajectory": input_stats,
                "refined_trajectory": refined_stats,
                "speed_ratio": speed_ratio,
                "acceleration_ratio": acceleration_ratio,
                "jerk_ratio": jerk_ratio,
                "frame_hand_motion_mean_mm": (
                    frame_motion * 1000.0
                ).tolist(),
                "contact_gate": case["gate"],
                "contact_improved_frames": case[
                    "contact_improved_frames"
                ],
                "eligible_frame_count": len(case["eligible_frames"]),
                "temporal_gate": temporal_gate,
            }
        )

    summary = {
        "geometry_dir": str(args.geometry_dir),
        "case_json_dir": str(args.case_json_dir),
        "thresholds": {
            "max_speed_ratio": args.max_speed_ratio,
            "max_acceleration_ratio": args.max_acceleration_ratio,
            "max_jerk_ratio": args.max_jerk_ratio,
        },
        "case_count": len(cases),
        "temporal_pass_count": sum(
            case["temporal_gate"]["passed"] for case in cases
        ),
        "contact_and_temporal_pass_count": sum(
            case["contact_gate"].get("passed", False)
            and case["temporal_gate"]["passed"]
            for case in cases
        ),
        "cases": cases,
    }
    summary["passed"] = (
        summary["case_count"] > 0
        and summary["contact_and_temporal_pass_count"]
        == summary["case_count"]
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
