#!/usr/bin/env python3
"""Evaluate HandX wrist-conditioned smoke outputs."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


WRIST_JOINTS = (0, 21)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--joint-subset",
        choices=("all", "wrist"),
        default="all",
    )
    return parser.parse_args()


def motion_xyz(motion, joint_subset):
    joints = np.asarray(motion, dtype=np.float64).reshape(
        motion.shape[0], 42, 4
    )[:, :, :3]
    if joint_subset == "wrist":
        joints = joints[:, WRIST_JOINTS]
    return joints


def trajectory_stat(values):
    norm = lambda array: np.linalg.norm(array, axis=-1)
    speed = norm(np.diff(values, axis=0)).mean()
    acceleration = norm(np.diff(values, n=2, axis=0)).mean()
    jerk = norm(np.diff(values, n=3, axis=0)).mean()
    return {
        "speed_mm": float(speed * 1000.0),
        "acceleration_mm": float(acceleration * 1000.0),
        "jerk_mm": float(jerk * 1000.0),
    }


def ratio(value, baseline):
    if baseline <= 0:
        return None
    return float(value / baseline)


def main():
    args = parse_args()
    cases = []
    for sample_path in sorted(args.input_dir.glob("val_sample_*.pkl")):
        with sample_path.open("rb") as handle:
            sample = pickle.load(handle)
        gt = motion_xyz(sample["gt_motion_real"], args.joint_subset)
        generated = motion_xyz(
            sample["generated_real"][0],
            args.joint_subset,
        )

        gt_wrist = (
            gt
            if args.joint_subset == "wrist"
            else gt[:, WRIST_JOINTS]
        )
        generated_wrist = (
            generated
            if args.joint_subset == "wrist"
            else generated[:, WRIST_JOINTS]
        )
        wrist_error = np.linalg.norm(
            generated_wrist - gt_wrist,
            axis=-1,
        )
        gt_stats = trajectory_stat(gt)
        generated_stats = trajectory_stat(generated)
        cases.append(
            {
                "sample": sample_path.name,
                "frames": int(gt.shape[0]),
                "wrist_rmse_mm": float(
                    np.sqrt(np.mean(np.square(wrist_error))) * 1000.0
                ),
                "wrist_mean_error_mm": float(
                    wrist_error.mean() * 1000.0
                ),
                "wrist_max_error_mm": float(
                    wrist_error.max() * 1000.0
                ),
                "gt_trajectory": gt_stats,
                "generated_trajectory": generated_stats,
                "speed_ratio": ratio(
                    generated_stats["speed_mm"],
                    gt_stats["speed_mm"],
                ),
                "acceleration_ratio": ratio(
                    generated_stats["acceleration_mm"],
                    gt_stats["acceleration_mm"],
                ),
                "jerk_ratio": ratio(
                    generated_stats["jerk_mm"],
                    gt_stats["jerk_mm"],
                ),
            }
        )

    summary = {
        "input_dir": str(args.input_dir),
        "joint_subset": args.joint_subset,
        "case_count": len(cases),
        "cases": cases,
    }
    summary["wrist_rmse_pass"] = all(
        case["wrist_rmse_mm"] <= 1.0 for case in cases
    )
    summary["time_ratio_pass"] = all(
        case["speed_ratio"] is None or case["speed_ratio"] <= 2.0
        for case in cases
    ) and all(
        case["acceleration_ratio"] is None
        or case["acceleration_ratio"] <= 2.0
        for case in cases
    ) and all(
        case["jerk_ratio"] is None or case["jerk_ratio"] <= 3.0
        for case in cases
    )
    summary["passed"] = (
        summary["case_count"] > 0
        and summary["wrist_rmse_pass"]
        and summary["time_ratio_pass"]
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
