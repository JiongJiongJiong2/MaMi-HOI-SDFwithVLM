#!/usr/bin/env python3
"""Extract object and hand-root motion features for ARCTIC Tier A events."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic-root", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-m", type=float, default=0.003)
    parser.add_argument("--pre-window", type=int, default=15)
    parser.add_argument("--post-window", type=int, default=30)
    return parser.parse_args()


def read_jsonl_gzip(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def write_jsonl_gzip(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def axis_angle_to_matrix(axis_angle):
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    theta = np.linalg.norm(axis_angle)
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis_angle / theta
    x, y, z = axis
    skew = np.asarray([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
    return (
        np.eye(3)
        + math.sin(theta) * skew
        + (1.0 - math.cos(theta)) * (skew @ skew)
    )


def rotation_delta_degrees(first, second):
    first_rotation = axis_angle_to_matrix(first)
    second_rotation = axis_angle_to_matrix(second)
    relative = first_rotation.T @ second_rotation
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def motion_norm(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 0.0
    return float(np.linalg.norm(values[-1] - values[0]))


def hand_root_displacement(mano, hand, start, end):
    values = np.asarray(mano[hand]["trans"][start:end], dtype=np.float64)
    return motion_norm(values)


def hand_retreat(mano, hand, release, end):
    values = np.asarray(
        mano[hand]["trans"][release:end],
        dtype=np.float64,
    )
    if len(values) < 2:
        return 0.0
    return float(np.linalg.norm(values - values[0], axis=1).max())


def summarize_values(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {
            "count": 0,
        }
    return {
        "count": int(len(values)),
        "min": float(values.min()),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def split_summary(rows, split):
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        return {"count": 0}
    return {
        "count": len(selected),
        "sequences": len({row["sequence"] for row in selected}),
        "participants": len({
            row["participant_id"] for row in selected
        }),
        "object_translation_m": summarize_values([
            row["object_translation_m"] for row in selected
        ]),
        "object_rotation_deg": summarize_values([
            row["object_rotation_deg"] for row in selected
        ]),
        "outgoing_retreat_m": summarize_values([
            row["outgoing_retreat_m"] for row in selected
        ]),
        "receiving_displacement_m": summarize_values([
            row["receiving_displacement_m"] for row in selected
        ]),
        "object_translation_ge_5mm": sum(
            row["object_translation_m"] >= 0.005
            for row in selected
        ),
        "object_rotation_ge_5deg": sum(
            row["object_rotation_deg"] >= 5.0
            for row in selected
        ),
        "either_object_motion": sum(
            row["object_translation_m"] >= 0.005
            or row["object_rotation_deg"] >= 5.0
            for row in selected
        ),
        "outgoing_retreat_ge_5mm": sum(
            row["outgoing_retreat_m"] >= 0.005
            for row in selected
        ),
        "receiving_displacement_ge_5mm": sum(
            row["receiving_displacement_m"] >= 0.005
            for row in selected
        ),
    }


def main():
    args = parse_args()
    candidates = [
        row for row in read_jsonl_gzip(args.candidates)
        if abs(float(row["threshold_m"]) - args.threshold_m) < 1e-12
    ]
    enriched = []
    for row in candidates:
        sequence = row["sequence"]
        sequence_root = args.arctic_root / "raw_seqs"
        subject, sequence_name = sequence.split("/", 1)
        mano_path = sequence_root / subject / f"{sequence_name}.mano.npy"
        object_path = sequence_root / subject / f"{sequence_name}.object.npy"
        mano = np.load(mano_path, allow_pickle=True).item()
        object_parameters = np.load(object_path, allow_pickle=True)
        frame_count = len(object_parameters)

        release = int(row["outgoing_release"])
        receiving_start = int(row["receiving_start"])
        pre_start = max(0, release - args.pre_window)
        post_end = min(frame_count, receiving_start + args.post_window)
        if post_end - pre_start < 2:
            continue

        object_translation = (
            np.asarray(object_parameters[:, 4:7], dtype=np.float64)
            / 1000.0
        )
        object_translation_m = motion_norm(
            object_translation[pre_start:post_end]
        )
        object_rotation_deg = rotation_delta_degrees(
            object_parameters[pre_start, 1:4],
            object_parameters[post_end - 1, 1:4],
        )
        outgoing_hand = row["outgoing_hand"]
        receiving_hand = row["receiving_hand"]
        outgoing_before = hand_root_displacement(
            mano,
            outgoing_hand,
            pre_start,
            release + 1,
        )
        outgoing_retreat_m = hand_retreat(
            mano,
            outgoing_hand,
            release,
            post_end,
        )
        receiving_displacement_m = hand_root_displacement(
            mano,
            receiving_hand,
            receiving_start,
            post_end,
        )
        left_root = np.asarray(mano["left"]["trans"], dtype=np.float64)
        right_root = np.asarray(mano["right"]["trans"], dtype=np.float64)
        separation = np.linalg.norm(left_root - right_root, axis=1)
        separation_change_m = motion_norm(separation[pre_start:post_end])
        enriched.append({
            **row,
            "pre_start": pre_start,
            "post_end": post_end,
            "object_translation_m": object_translation_m,
            "object_rotation_deg": object_rotation_deg,
            "outgoing_root_motion_before_release_m": outgoing_before,
            "outgoing_retreat_m": outgoing_retreat_m,
            "receiving_displacement_m": receiving_displacement_m,
            "hand_separation_change_m": separation_change_m,
        })

    if not enriched:
        raise RuntimeError("no Tier A candidates were enriched")
    result = {
        "threshold_m": args.threshold_m,
        "pre_window": args.pre_window,
        "post_window": args.post_window,
        "summary": {
            split: split_summary(enriched, split)
            for split in ("train", "val")
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "event_review_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl_gzip(
        args.output_dir / "tier_a_event_features.jsonl.gz",
        enriched,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
