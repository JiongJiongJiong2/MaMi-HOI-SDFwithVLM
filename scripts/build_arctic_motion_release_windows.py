#!/usr/bin/env python3
"""Build motion-aware instant-release windows for ARCTIC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scripts.audit_arctic_handover_gate import stable_contact
    from scripts.build_arctic_online_release_windows import (
        confirmed_releases,
        load_trajectories,
        raw_contact,
        read_jsonl_gzip,
    )
    from scripts.build_arctic_role_switch_windows import extract_window
    from scripts.review_arctic_role_switch_events import (
        rotation_delta_degrees,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_arctic_handover_gate import stable_contact  # noqa: E402
    from build_arctic_online_release_windows import (  # noqa: E402
        confirmed_releases,
        load_trajectories,
        raw_contact,
        read_jsonl_gzip,
    )
    from build_arctic_role_switch_windows import (  # noqa: E402
        extract_window,
    )
    from review_arctic_role_switch_events import (  # noqa: E402
        rotation_delta_degrees,
    )


VELOCITY_CAP_MM_PER_FRAME = 200.0
ANGULAR_CAP_DEG_PER_FRAME = 30.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic-root", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--tier-a-candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--threshold-m", type=float, default=0.003)
    parser.add_argument("--minimum-contact-frames", type=int, default=15)
    parser.add_argument("--history", type=int, default=30)
    return parser.parse_args()


def velocity(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.zeros_like(values)
    if len(values) > 1:
        result[1:] = values[1:] - values[:-1]
    return result


def speed(values):
    return np.linalg.norm(velocity(values), axis=1)


def load_sequence_motion(
    raw_root,
    sequence,
    trajectory,
):
    subject, sequence_name = sequence.split("/", 1)
    mano_path = raw_root / subject / f"{sequence_name}.mano.npy"
    object_path = raw_root / subject / f"{sequence_name}.object.npy"
    mano = np.load(mano_path, allow_pickle=True).item()
    object_parameters = np.load(object_path, allow_pickle=True)
    right = np.asarray(mano["right"]["trans"], dtype=np.float64)
    left = np.asarray(mano["left"]["trans"], dtype=np.float64)
    object_translation = (
        np.asarray(object_parameters[:, 4:7], dtype=np.float64)
        / 1000.0
    )
    object_rotation = np.asarray(
        object_parameters[:, 1:4],
        dtype=np.float64,
    )
    object_speed = np.clip(
        speed(object_translation) * 1000.0,
        0.0,
        VELOCITY_CAP_MM_PER_FRAME,
    )
    angular_speed = np.zeros(len(object_rotation), dtype=np.float64)
    for index in range(1, len(object_rotation)):
        angular_speed[index] = rotation_delta_degrees(
            object_rotation[index - 1],
            object_rotation[index],
        )
    angular_speed = np.clip(
        angular_speed,
        0.0,
        ANGULAR_CAP_DEG_PER_FRAME,
    )
    right_velocity = velocity(right)
    left_velocity = velocity(left)
    object_velocity = velocity(object_translation)
    result = {}
    for outgoing_hand in ("right", "left"):
        outgoing_velocity = (
            right_velocity
            if outgoing_hand == "right"
            else left_velocity
        )
        other_velocity = (
            left_velocity if outgoing_hand == "right" else right_velocity
        )
        outgoing_relative = outgoing_velocity - object_velocity
        other_relative = other_velocity - object_velocity
        result[outgoing_hand] = np.stack((
            np.clip(
                np.linalg.norm(outgoing_velocity, axis=1) * 1000.0,
                0.0,
                VELOCITY_CAP_MM_PER_FRAME,
            ),
            np.clip(
                np.linalg.norm(other_velocity, axis=1) * 1000.0,
                0.0,
                VELOCITY_CAP_MM_PER_FRAME,
            ),
            np.clip(
                np.linalg.norm(outgoing_relative, axis=1) * 1000.0,
                0.0,
                VELOCITY_CAP_MM_PER_FRAME,
            ),
            np.clip(
                np.linalg.norm(other_relative, axis=1) * 1000.0,
                0.0,
                VELOCITY_CAP_MM_PER_FRAME,
            ),
            object_speed,
            angular_speed,
        ), axis=1)
    return result


def extract_motion_window(values, release, history):
    values = np.asarray(values, dtype=np.float64)
    release = int(release)
    if release <= 0:
        return np.zeros((history, 6), dtype=np.float32)
    start = max(0, release - history)
    indices = np.arange(start, release)
    if len(indices) < history:
        padding = np.repeat(indices[:1], history - len(indices))
        indices = np.concatenate((padding, indices))
    return values[indices].astype(np.float32)


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
    tier_a_rows = read_jsonl_gzip(args.tier_a_candidates)
    tier_a_keys = {
        (
            row["sequence"],
            row["outgoing_hand"],
            int(row["outgoing_release"]),
        )
        for row in tier_a_rows
        if abs(float(row["threshold_m"]) - args.threshold_m) < 1e-12
    }
    raw_root = args.arctic_root / "raw_seqs"
    motion_cache = {}
    samples = []
    trigger_keys = {"train": set(), "val": set()}
    for sequence, trajectory in sorted(trajectories.items()):
        split = trajectory["split"]
        if split not in ("train", "val"):
            continue
        motion_by_hand = None
        for hand in ("right", "left"):
            contact = raw_contact(
                trajectory[hand],
                args.threshold_m,
            )
            releases = confirmed_releases(
                contact,
                args.minimum_contact_frames,
                0,
            )
            for release in releases:
                key = (sequence, hand, release)
                trigger_keys[split].add(key)
                if motion_by_hand is None:
                    motion_by_hand = load_sequence_motion(
                        raw_root,
                        sequence,
                        trajectory,
                    )
                distance_window = extract_window(
                    trajectory,
                    release,
                    args.history,
                )
                motion_window = extract_motion_window(
                    motion_by_hand[hand],
                    release,
                    args.history,
                )
                samples.append({
                    "features": np.concatenate(
                        (distance_window, motion_window),
                        axis=1,
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
        raise RuntimeError("no release windows")
    features = np.stack([row["features"] for row in samples])
    labels = np.asarray([row["label"] for row in samples], dtype=np.int64)
    expected = {
        split: {
            key for key in tier_a_keys
            if key[0] in trajectories
            and trajectories[key[0]]["split"] == split
        }
        for split in ("train", "val")
    }
    recall = {
        split: {
            "tier_a_events": len(expected[split]),
            "triggered": len(expected[split] & trigger_keys[split]),
            "recall": (
                len(expected[split] & trigger_keys[split])
                / len(expected[split])
                if expected[split]
                else 1.0
            ),
        }
        for split in ("train", "val")
    }
    label_counts = {
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
    result = {
        "threshold_m": args.threshold_m,
        "minimum_contact_frames": args.minimum_contact_frames,
        "history": args.history,
        "feature_count": int(features.shape[-1]),
        "samples": len(samples),
        "label_counts": label_counts,
        "trigger_recall": recall,
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
