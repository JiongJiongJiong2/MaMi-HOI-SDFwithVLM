#!/usr/bin/env python3
"""Analyze skeleton consistency of HandX generated hand trajectories."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handx-root", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--include-gt", action="store_true")
    return parser.parse_args()


def motion_xyz(motion):
    return np.asarray(motion, dtype=np.float64).reshape(
        motion.shape[0], 42, 4
    )[:, :, :3]


def trajectory_stat(values):
    norm = lambda array: np.linalg.norm(array, axis=-1)
    return {
        "speed_mm": float(
            norm(np.diff(values, axis=0)).mean() * 1000.0
        ),
        "acceleration_mm": float(
            norm(np.diff(values, n=2, axis=0)).mean() * 1000.0
        ),
        "jerk_mm": float(
            norm(np.diff(values, n=3, axis=0)).mean() * 1000.0
        ),
    }


def bone_metrics(motion, chains):
    result = {}
    for offset, side in ((0, "left"), (21, "right")):
        hand = motion[:, offset : offset + 21]
        chain_cvs = []
        for chain in chains:
            if max(chain) >= 21:
                continue
            lengths = np.linalg.norm(
                hand[:, chain[1:]] - hand[:, chain[:-1]],
                axis=2,
            )
            chain_cvs.append(
                lengths.std(axis=0) / (lengths.mean(axis=0) + 1e-9)
            )
        values = np.concatenate(chain_cvs)
        result[side] = {
            "bone_cv_mean": float(values.mean()),
            "bone_cv_max": float(values.max()),
        }
    return result


def main():
    args = parse_args()
    diffusion_root = args.handx_root / "diffusion"
    sys.path.insert(0, str(diffusion_root))
    from src.constant import SKELETON_CHAIN

    cases = []
    for sample_path in sorted(args.input_dir.glob("val_sample_*.pkl")):
        with sample_path.open("rb") as handle:
            sample = pickle.load(handle)
        generated = motion_xyz(sample["generated_real"][0])
        case = {
            "sample": sample_path.name,
            "finite": bool(np.isfinite(generated).all()),
            "generated": {
                **bone_metrics(generated, SKELETON_CHAIN),
                "trajectory": trajectory_stat(generated),
            },
        }
        if args.include_gt:
            gt = motion_xyz(sample["gt_motion_real"])
            case["gt"] = {
                **bone_metrics(gt, SKELETON_CHAIN),
                "trajectory": trajectory_stat(gt),
            }
        cases.append(case)

    summary = {
        "input_dir": str(args.input_dir),
        "include_gt": args.include_gt,
        "case_count": len(cases),
        "cases": cases,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
