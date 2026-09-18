#!/usr/bin/env python3
"""Smooth ContactOpt finger PCA trajectories and re-evaluate contact."""

import argparse
import copy
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from contactopt_contact_metrics import sanitized_contact_mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--optimized-pkl-dir", type=Path, required=True)
    parser.add_argument("--case-tag", required=True)
    parser.add_argument("--case-json-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-speed-ratio", type=float, default=2.0)
    parser.add_argument("--max-acceleration-ratio", type=float, default=2.0)
    parser.add_argument("--max-jerk-ratio", type=float, default=3.0)
    return parser.parse_args()


def smooth_pose_sequence(pose, finger_start=3):
    smoothed = np.asarray(pose, dtype=np.float32).copy()
    coefficients = smoothed[:, finger_start:]
    padded = np.pad(coefficients, ((1, 1), (0, 0)), mode="edge")
    smoothed[:, finger_start:] = (
        padded[:-2] + 2.0 * padded[1:-1] + padded[2:]
    ) / 4.0
    return smoothed


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


def ratio(value, baseline):
    if baseline <= 0:
        return None
    return float(value / baseline)


def contact_metrics(input_hand, candidate_hand):
    input_hand.calc_dist_contact(hand=True, obj=True)
    candidate_hand.calc_dist_contact(hand=True, obj=True)

    input_contact, input_finite = sanitized_contact_mean(
        input_hand.hand_contact
    )
    candidate_contact, candidate_finite = sanitized_contact_mean(
        candidate_hand.hand_contact
    )
    input_distance = cKDTree(input_hand.obj_verts).query(
        input_hand.hand_verts
    )[0]
    candidate_distance = cKDTree(candidate_hand.obj_verts).query(
        candidate_hand.hand_verts
    )[0]
    wrist_drift = float(
        np.linalg.norm(
            candidate_hand.hand_joints[0] - input_hand.hand_joints[0]
        )
    )
    object_drift = float(
        np.linalg.norm(
            candidate_hand.obj_verts - input_hand.obj_verts, axis=1
        ).max()
    )
    return {
        "contact_improved": bool(candidate_contact > input_contact),
        "input_contact_mean": float(input_contact),
        "candidate_contact_mean": float(candidate_contact),
        "contact_relative_change": (
            float(candidate_contact / input_contact - 1.0)
            if input_contact > 0
            else None
        ),
        "input_contact_finite_fraction": input_finite,
        "candidate_contact_finite_fraction": candidate_finite,
        "input_distance_mean_m": float(input_distance.mean()),
        "candidate_distance_mean_m": float(candidate_distance.mean()),
        "distance_relative_change": float(
            candidate_distance.mean() / input_distance.mean() - 1.0
        ),
        "wrist_root_drift_m": wrist_drift,
        "object_vertex_drift_m": object_drift,
        "hand_vertex_motion_mean_m": float(
            np.linalg.norm(
                candidate_hand.hand_verts - input_hand.hand_verts,
                axis=1,
            ).mean()
        ),
    }


def summarize_arm(
    input_vertices,
    candidate_vertices,
    thresholds,
):
    input_trajectory = trajectory_stat(input_vertices)
    candidate_trajectory = trajectory_stat(candidate_vertices)
    speed_ratio = ratio(
        candidate_trajectory["speed_mm"],
        input_trajectory["speed_mm"],
    )
    acceleration_ratio = ratio(
        candidate_trajectory["acceleration_mm"],
        input_trajectory["acceleration_mm"],
    )
    jerk_ratio = ratio(
        candidate_trajectory["jerk_mm"],
        input_trajectory["jerk_mm"],
    )
    temporal_gate = {
        "speed_ratio": (
            speed_ratio is None
            or speed_ratio <= thresholds["max_speed_ratio"]
        ),
        "acceleration_ratio": (
            acceleration_ratio is None
            or acceleration_ratio
            <= thresholds["max_acceleration_ratio"]
        ),
        "jerk_ratio": (
            jerk_ratio is None
            or jerk_ratio <= thresholds["max_jerk_ratio"]
        ),
    }
    temporal_gate["passed"] = all(temporal_gate.values())
    return {
        "input_trajectory": input_trajectory,
        "candidate_trajectory": candidate_trajectory,
        "speed_ratio": speed_ratio,
        "acceleration_ratio": acceleration_ratio,
        "jerk_ratio": jerk_ratio,
        "temporal_gate": temporal_gate,
    }


def per_frame_summary(rows):
    contact_changes = [
        row["contact_relative_change"]
        for row in rows
        if row["contact_relative_change"] is not None
    ]
    distance_changes = [
        row["distance_relative_change"] for row in rows
    ]
    return {
        "frame_count": len(rows),
        "contact_improved_frames": int(
            sum(change > 0 for change in contact_changes)
        ),
        "distance_improved_frames": int(
            sum(change < 0 for change in distance_changes)
        ),
        "mean_contact_relative_change": (
            float(np.mean(contact_changes)) if contact_changes else None
        ),
        "mean_distance_relative_change": float(np.mean(distance_changes)),
        "max_wrist_drift_m": float(
            max(row["wrist_root_drift_m"] for row in rows)
        ),
        "max_object_drift_m": float(
            max(row["object_vertex_drift_m"] for row in rows)
        ),
    }


def main():
    args = parse_args()
    manifest = json.loads(
        args.manifest_json.read_text(encoding="utf-8")
    )
    thresholds = {
        "max_speed_ratio": args.max_speed_ratio,
        "max_acceleration_ratio": args.max_acceleration_ratio,
        "max_jerk_ratio": args.max_jerk_ratio,
    }
    cases = []

    for candidate in manifest["selected"]:
        sequence = candidate["sequence"]
        case = json.loads(
            (args.case_json_dir / f"{sequence}.json").read_text(
                encoding="utf-8"
            )
        )
        optimized_path = (
            args.optimized_pkl_dir
            / f"optimized_{args.case_tag}_{sequence}.pkl"
        )
        with optimized_path.open("rb") as handle:
            runs = pickle.load(handle)

        output_pose = np.asarray(
            [run["out_ho"].hand_pose for run in runs],
            dtype=np.float32,
        )
        smoothed_pose = smooth_pose_sequence(output_pose)
        input_vertices = np.asarray(
            [run["in_ho"].hand_verts for run in runs],
            dtype=np.float32,
        )
        raw_vertices = np.asarray(
            [run["out_ho"].hand_verts for run in runs],
            dtype=np.float32,
        )

        smoothed_vertices = []
        smoothed_runs = []
        for run, pose in zip(runs, smoothed_pose):
            input_hand = run["in_ho"]
            smoothed_hand = copy.deepcopy(input_hand)
            smoothed_hand.hand_pose = pose
            smoothed_hand.run_mano()
            smoothed_runs.append(smoothed_hand)
            smoothed_vertices.append(smoothed_hand.hand_verts)
        smoothed_vertices = np.asarray(
            smoothed_vertices, dtype=np.float32
        )

        raw_rows = []
        smoothed_rows = []
        for index, run in enumerate(runs):
            input_hand = run["in_ho"]
            raw_metrics = contact_metrics(input_hand, run["out_ho"])
            smooth_metrics = contact_metrics(
                input_hand, smoothed_runs[index]
            )
            raw_rows.append(raw_metrics)
            smoothed_rows.append(smooth_metrics)

        raw_summary = summarize_arm(
            input_vertices,
            raw_vertices,
            thresholds,
        )
        raw_summary["frame_summary"] = per_frame_summary(raw_rows)
        smoothed_summary = summarize_arm(
            input_vertices,
            smoothed_vertices,
            thresholds,
        )
        smoothed_summary["frame_summary"] = per_frame_summary(
            smoothed_rows
        )

        required = max(3, int(np.ceil(0.7 * len(runs))))
        smooth_contact = smoothed_summary["frame_summary"]
        smoothed_summary["promotion_gate"] = {
            "contact_improved": (
                smooth_contact["contact_improved_frames"] >= required
            ),
            "distance_not_regressed": (
                smooth_contact["mean_distance_relative_change"] <= 0.10
            ),
            "wrist_frozen": (
                smooth_contact["max_wrist_drift_m"] <= 0.00001
            ),
            "object_frozen": (
                smooth_contact["max_object_drift_m"] <= 0.000001
            ),
            **smoothed_summary["temporal_gate"],
        }
        smoothed_summary["promotion_gate"]["passed"] = all(
            smoothed_summary["promotion_gate"].values()
        )

        cases.append(
            {
                "sequence": sequence,
                "object_name": candidate["object_name"],
                "frames": candidate["frames"],
                "required_improved_frames": required,
                "raw": raw_summary,
                "smoothed": smoothed_summary,
            }
        )

    summary = {
        "manifest_json": str(args.manifest_json),
        "optimized_pkl_dir": str(args.optimized_pkl_dir),
        "case_tag": args.case_tag,
        "method": (
            "smooth ContactOpt pose coefficients 3:18 with [1,2,1]/4; "
            "keep global pose coefficients 0:3 and hand_mTc unchanged; "
            "rerun MANO forward geometry"
        ),
        "thresholds": thresholds,
        "case_count": len(cases),
        "raw_temporal_pass_count": sum(
            case["raw"]["temporal_gate"]["passed"] for case in cases
        ),
        "smoothed_temporal_pass_count": sum(
            case["smoothed"]["temporal_gate"]["passed"] for case in cases
        ),
        "smoothed_promotion_pass_count": sum(
            case["smoothed"]["promotion_gate"]["passed"]
            for case in cases
        ),
        "cases": cases,
    }
    summary["passed"] = (
        summary["case_count"] > 0
        and summary["smoothed_promotion_pass_count"]
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
