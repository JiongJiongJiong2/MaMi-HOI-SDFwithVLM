#!/usr/bin/env python3
"""Build online release-detector windows for ARCTIC."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scripts.audit_arctic_handover_gate import (
        binary_segments,
        stable_contact,
    )
    from scripts.build_arctic_role_switch_windows import extract_window
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_arctic_handover_gate import (  # noqa: E402
        binary_segments,
        stable_contact,
    )
    from build_arctic_role_switch_windows import (  # noqa: E402
        extract_window,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--tier-a-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--threshold-m", type=float, default=0.003)
    parser.add_argument("--minimum-contact-frames", type=int, default=15)
    parser.add_argument("--release-confirmation-frames", type=int, default=5)
    parser.add_argument("--history", type=int, default=30)
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
            "split": str(arrays["splits"][index]),
            "participant_id": str(arrays["participants"][index]),
            "object_name": str(arrays["objects"][index]),
            "right": arrays["right_distance_m"][index, :length],
            "left": arrays["left_distance_m"][index, :length],
        }
    return result


def raw_contact(distance, threshold_m):
    distance = np.asarray(distance, dtype=np.float64)
    return np.isfinite(distance) & (distance <= threshold_m)


def confirmed_releases(
    contact,
    minimum_contact_frames,
    confirmation_frames,
):
    stable = stable_contact(
        contact,
        minimum_contact_frames,
        maximum_gap=0,
    )
    releases = []
    for _, end in binary_segments(stable):
        confirmation_end = end + confirmation_frames
        if confirmation_end > len(contact):
            continue
        if contact[end:confirmation_end].any():
            continue
        releases.append(int(end))
    return releases


def null_baseline(labels):
    prevalence = float(np.mean(labels))
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
    trajectories = load_trajectories(args.trajectories)
    tier_a = read_jsonl_gzip(args.tier_a_candidates)
    tier_a_keys = {
        (
            row["sequence"],
            row["outgoing_hand"],
            int(row["outgoing_release"]),
        )
        for row in tier_a
        if abs(float(row["threshold_m"]) - args.threshold_m) < 1e-12
    }
    tier_a_by_split = {
        split: {
            key for key in tier_a_keys
            if key[0] in trajectories
            and trajectories[key[0]]["split"] == split
        }
        for split in ("train", "val")
    }

    samples = []
    trigger_by_split = {"train": set(), "val": set()}
    for sequence, trajectory in sorted(trajectories.items()):
        split = trajectory["split"]
        if split not in ("train", "val"):
            continue
        for hand in ("right", "left"):
            contact = raw_contact(
                trajectory[hand],
                args.threshold_m,
            )
            releases = confirmed_releases(
                contact,
                args.minimum_contact_frames,
                args.release_confirmation_frames,
            )
            for release in releases:
                key = (sequence, hand, release)
                trigger_by_split[split].add(key)
                samples.append({
                    "features": extract_window(
                        trajectory,
                        release,
                        args.history,
                    ),
                    "label": int(key in tier_a_keys),
                    "split": split,
                    "sequence": sequence,
                    "participant_id": trajectory["participant_id"],
                    "object_name": trajectory["object_name"],
                    "outgoing_hand": hand,
                    "outgoing_release": release,
                    "event_id": "|".join(map(str, key)),
                })
    if not samples:
        raise RuntimeError("no confirmed release events")

    features = np.stack([row["features"] for row in samples])
    labels = np.asarray([row["label"] for row in samples], dtype=np.int64)
    split_rows = {
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
    }
    recall = {}
    for split in ("train", "val"):
        expected = tier_a_by_split[split]
        found = trigger_by_split[split]
        recall[split] = {
            "tier_a_events": len(expected),
            "triggered": len(expected & found),
            "recall": (
                len(expected & found) / len(expected)
                if expected
                else 1.0
            ),
        }
    split_labels = {
        split: labels[np.asarray([
            row["split"] == split for row in samples
        ])]
        for split in ("train", "val")
    }
    result = {
        "threshold_m": args.threshold_m,
        "minimum_contact_frames": args.minimum_contact_frames,
        "release_confirmation_frames": args.release_confirmation_frames,
        "history": args.history,
        "feature_count": int(features.shape[-1]),
        "samples": len(samples),
        "label_counts": split_rows,
        "trigger_recall": recall,
        "null": {
            split: null_baseline(values)
            for split, values in split_labels.items()
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
        hands=np.asarray([row["outgoing_hand"] for row in samples]),
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
