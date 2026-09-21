#!/usr/bin/env python3
"""Build event-centered candidate-quality windows for ARCTIC."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


HISTORY = 30
DISTANCE_CAP_M = 0.020
DELTA_SCALE_M = 0.010


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--raw-candidates", type=Path, required=True)
    parser.add_argument("--tier-a-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--threshold-m", type=float, default=0.003)
    parser.add_argument("--history", type=int, default=HISTORY)
    return parser.parse_args()


def read_jsonl_gzip(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def load_trajectories(path):
    arrays = np.load(path, allow_pickle=True)
    result = {}
    for index, key in enumerate(arrays["keys"]):
        length = int(arrays["lengths"][index])
        result[str(key)] = {
            "right": arrays["right_distance_m"][index, :length],
            "left": arrays["left_distance_m"][index, :length],
        }
    return result


def candidate_key(row):
    return (
        row["sequence"],
        row["outgoing_hand"],
        int(row["outgoing_release"]),
        int(row["receiving_start"]),
        int(row["receiving_end"]),
    )


def capped_distance(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, DISTANCE_CAP_M)
    return np.clip(values, 0.0, DISTANCE_CAP_M) / DISTANCE_CAP_M


def normalized_delta(current, previous):
    delta = current - previous
    return np.clip(delta / DELTA_SCALE_M, -5.0, 5.0)


def extract_window(trajectory, release, history):
    right = np.asarray(trajectory["right"], dtype=np.float64)
    left = np.asarray(trajectory["left"], dtype=np.float64)
    release = int(release)
    if release <= 0:
        return np.zeros((history, 10), dtype=np.float32)
    start = max(0, release - history)
    indices = np.arange(start, release)
    if len(indices) < history:
        padding = np.repeat(indices[:1], history - len(indices))
        indices = np.concatenate((padding, indices))

    right_raw = right[indices]
    left_raw = left[indices]
    right_prev1 = right[np.maximum(indices - 1, 0)]
    left_prev1 = left[np.maximum(indices - 1, 0)]
    right_prev5 = right[np.maximum(indices - 5, 0)]
    left_prev5 = left[np.maximum(indices - 5, 0)]

    features = np.stack((
        capped_distance(right_raw),
        capped_distance(left_raw),
        normalized_delta(right_raw, right_prev1),
        normalized_delta(left_raw, left_prev1),
        normalized_delta(right_raw, right_prev5),
        normalized_delta(left_raw, left_prev5),
        (right_raw <= 0.003).astype(np.float64),
        (left_raw <= 0.003).astype(np.float64),
        (right_raw <= 0.010).astype(np.float64),
        (left_raw <= 0.010).astype(np.float64),
    ), axis=1)
    return features.astype(np.float32)


def null_baseline(labels):
    prevalence = float(np.mean(labels))
    all_positive_f1 = (
        2.0 * prevalence / (1.0 + prevalence)
        if prevalence
        else 0.0
    )
    return {
        "prevalence": prevalence,
        "all_positive_f1": all_positive_f1,
    }


def main():
    args = parse_args()
    trajectories = load_trajectories(args.trajectories)
    raw = [
        row for row in read_jsonl_gzip(args.raw_candidates)
        if abs(float(row["threshold_m"]) - args.threshold_m) < 1e-12
    ]
    tier_a = {
        candidate_key(row)
        for row in read_jsonl_gzip(args.tier_a_candidates)
        if abs(float(row["threshold_m"]) - args.threshold_m) < 1e-12
    }
    samples = []
    for row in raw:
        sequence = row["sequence"]
        if sequence not in trajectories:
            raise KeyError(f"missing trajectory for {sequence}")
        samples.append({
            "features": extract_window(
                trajectories[sequence],
                int(row["outgoing_release"]),
                args.history,
            ),
            "label": int(candidate_key(row) in tier_a),
            "split": row["split"],
            "sequence": sequence,
            "participant_id": row["participant_id"],
            "object_name": row["object_name"],
            "direction": row["direction"],
            "outgoing_release": int(row["outgoing_release"]),
            "event_id": "|".join(map(str, candidate_key(row))),
        })
    features = np.stack([row["features"] for row in samples])
    labels = np.asarray([row["label"] for row in samples], dtype=np.int64)
    result = {
        "threshold_m": args.threshold_m,
        "history": args.history,
        "feature_count": int(features.shape[-1]),
        "samples": len(samples),
        "label_counts": {
            split: {
                "samples": int(sum(row["split"] == split for row in samples)),
                "positives": int(sum(
                    row["split"] == split and row["label"] == 1
                    for row in samples
                )),
                "negatives": int(sum(
                    row["split"] == split and row["label"] == 0
                    for row in samples
                )),
                "participants": len({
                    row["participant_id"]
                    for row in samples
                    if row["split"] == split
                }),
            }
            for split in ("train", "val")
        },
        "null": {
            split: null_baseline(labels[[
                index for index, row in enumerate(samples)
                if row["split"] == split
            ]])
            for split in ("train", "val")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        labels=labels,
        splits=np.asarray([row["split"] for row in samples]),
        sequences=np.asarray([row["sequence"] for row in samples]),
        participants=np.asarray([
            row["participant_id"] for row in samples
        ]),
        objects=np.asarray([row["object_name"] for row in samples]),
        directions=np.asarray([row["direction"] for row in samples]),
        outgoing_release=np.asarray([
            row["outgoing_release"] for row in samples
        ]),
        event_ids=np.asarray([row["event_id"] for row in samples]),
    )
    args.summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
