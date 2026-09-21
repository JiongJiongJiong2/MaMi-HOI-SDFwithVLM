#!/usr/bin/env python3
"""Build event-centered handover windows from the OakInk manifest."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


HISTORY = 30
POSITIVE_HORIZON = 15
AMBIGUOUS_MARGIN = 30
DISTANCE_CAP_M = 0.020
DELTA_SCALE_M = 0.010
PRIMARY_THRESHOLD_M = 0.005


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--threshold-m", type=float, default=PRIMARY_THRESHOLD_M)
    parser.add_argument("--history", type=int, default=HISTORY)
    return parser.parse_args()


def read_candidates(path, threshold_m):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if abs(float(row["threshold_m"]) - threshold_m) < 1e-12:
                rows.append(row)
    return rows


def select_primary_events(candidates):
    grouped = {}
    for row in candidates:
        grouped.setdefault(row["sequence"], []).append(row)
    result = {}
    for sequence, rows in grouped.items():
        result[sequence] = min(
            rows,
            key=lambda row: (
                abs(int(row["onset_minus_release"])),
                int(row["giver_release"]),
            ),
        )
    return result


def capped_distance(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, DISTANCE_CAP_M)
    return np.clip(values, 0.0, DISTANCE_CAP_M) / DISTANCE_CAP_M


def normalized_delta(current, previous):
    delta = current - previous
    return np.clip(delta / DELTA_SCALE_M, -5.0, 5.0)


def extract_window(
    giver_distance_m,
    receiver_distance_m,
    frame,
    history,
):
    giver = np.asarray(giver_distance_m, dtype=np.float64)
    receiver = np.asarray(receiver_distance_m, dtype=np.float64)
    if frame <= 0:
        return np.zeros((history, 10), dtype=np.float32)
    start = max(0, frame - history)
    indices = np.arange(start, frame)
    if len(indices) < history:
        padding = np.repeat(indices[:1], history - len(indices))
        indices = np.concatenate((padding, indices))
    giver_current = giver[indices]
    receiver_current = receiver[indices]
    giver_prev1 = giver[np.maximum(indices - 1, 0)]
    receiver_prev1 = receiver[np.maximum(indices - 1, 0)]
    giver_prev5 = giver[np.maximum(indices - 5, 0)]
    receiver_prev5 = receiver[np.maximum(indices - 5, 0)]
    return np.stack((
        capped_distance(giver_current),
        capped_distance(receiver_current),
        normalized_delta(giver_current, giver_prev1),
        normalized_delta(receiver_current, receiver_prev1),
        normalized_delta(giver_current, giver_prev5),
        normalized_delta(receiver_current, receiver_prev5),
        (giver_current <= 0.005).astype(np.float64),
        (receiver_current <= 0.005).astype(np.float64),
        (giver_current <= 0.010).astype(np.float64),
        (receiver_current <= 0.010).astype(np.float64),
    ), axis=1).astype(np.float32)


def null_baseline(labels):
    prevalence = float(np.mean(labels)) if len(labels) else 0.0
    return {
        "prevalence": prevalence,
        "all_positive_f1": (
            2.0 * prevalence / (1.0 + prevalence)
            if prevalence
            else 0.0
        ),
    }


def main():
    args = parse_args()
    arrays = np.load(args.trajectories, allow_pickle=True)
    primary_events = select_primary_events(
        read_candidates(args.candidates, args.threshold_m)
    )
    samples = []
    for index, sequence in enumerate(arrays["sequences"]):
        sequence = str(sequence)
        split = str(arrays["splits"][index])
        participant_pair = str(arrays["participant_pairs"][index])
        object_name = str(arrays["object_names"][index])
        giver = np.asarray(arrays["giver_distance_m"][index])
        receiver = np.asarray(arrays["receiver_distance_m"][index])
        event = primary_events.get(sequence)
        release = int(event["giver_release"]) if event else None
        for frame in range(1, len(giver)):
            if release is not None and release - POSITIVE_HORIZON <= frame < release:
                label = 1
            elif release is not None and abs(frame - release) <= AMBIGUOUS_MARGIN:
                continue
            else:
                label = 0
            samples.append({
                "features": extract_window(
                    giver,
                    receiver,
                    frame,
                    args.history,
                ),
                "label": label,
                "split": split,
                "sequence": sequence,
                "participant_pair": participant_pair,
                "object_name": object_name,
                "frame": frame,
                "release": release if release is not None else -1,
            })
    if not samples:
        raise RuntimeError("no OakInk windows")
    features = np.stack([row["features"] for row in samples])
    labels = np.asarray([row["label"] for row in samples], dtype=np.int64)
    summary = {}
    for split in ("train", "val", "test"):
        mask = np.asarray([row["split"] == split for row in samples])
        summary[split] = {
            "samples": int(mask.sum()),
            "positives": int(labels[mask].sum()),
            "negatives": int(mask.sum() - labels[mask].sum()),
            "sequences": len({
                row["sequence"]
                for row in samples
                if row["split"] == split
            }),
            "participants": len({
                row["participant_pair"]
                for row in samples
                if row["split"] == split
            }),
            "null": null_baseline(labels[mask]),
        }
    result = {
        "threshold_m": args.threshold_m,
        "history": args.history,
        "feature_count": int(features.shape[-1]),
        "primary_events": len(primary_events),
        "split_summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=features,
        labels=labels,
        splits=np.asarray([row["split"] for row in samples]),
        sequences=np.asarray([row["sequence"] for row in samples]),
        participants=np.asarray([
            row["participant_pair"] for row in samples
        ]),
        objects=np.asarray([row["object_name"] for row in samples]),
        frames=np.asarray([row["frame"] for row in samples]),
        releases=np.asarray([row["release"] for row in samples]),
    )
    args.summary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
