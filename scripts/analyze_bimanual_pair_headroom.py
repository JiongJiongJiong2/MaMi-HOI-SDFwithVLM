#!/usr/bin/env python3
"""Measure bimanual candidate-pair headroom on aligned L1 events."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from manip.world_model.contact_action.episodes import (
    contact_episode_metrics,
)


COHORTS = ("10_30", "31_50", "51_70")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--penetration_weight", type=float, default=20.0)
    parser.add_argument("--stable_min_frames", type=int, default=3)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260918)
    return parser.parse_args()


def load_events(root):
    events = []
    for cohort in COHORTS:
        data = json.loads(
            (root / cohort / "result.json").read_text(encoding="utf-8")
        )
        events.extend(data["events"])
    return events


def group_events(events):
    grouped = defaultdict(dict)
    for event in events:
        grouped[event["sequence"]][event["hand"]] = event
    return grouped


def aligned_contact_sequence(event, candidate_index, frames):
    frame_to_value = dict(zip(
        event["frame_indices"],
        event["candidate_contact_sequences"][candidate_index],
    ))
    return np.asarray(
        [bool(frame_to_value.get(int(frame), False)) for frame in frames],
        dtype=bool,
    )


def aligned_ground_truth(event, frames):
    frame_to_value = dict(zip(
        event["frame_indices"],
        event["ground_truth_contact"],
    ))
    return np.asarray(
        [bool(frame_to_value.get(int(frame), False)) for frame in frames],
        dtype=bool,
    )


def mean_defined(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def pair_metrics(
    left_prediction,
    right_prediction,
    left_truth,
    right_truth,
    stable_min_frames,
):
    left = contact_episode_metrics(
        left_prediction,
        left_truth,
        stable_min_frames=stable_min_frames,
    )
    right = contact_episode_metrics(
        right_prediction,
        right_truth,
        stable_min_frames=stable_min_frames,
    )
    return {
        "joint_f1": mean_defined([
            left["contact_f1"],
            right["contact_f1"],
        ]),
        "joint_precision": mean_defined([
            left["contact_precision"],
            right["contact_precision"],
        ]),
        "joint_recall": mean_defined([
            left["contact_recall"],
            right["contact_recall"],
        ]),
        "both_stable_contact_success": int(
            left["stable_contact_success"]
            and right["stable_contact_success"]
        ),
        "max_stable_contact_false_positive": max(
            left["stable_contact_false_positive"],
            right["stable_contact_false_positive"],
        ),
        "mean_early_contact_frames": float(
            (
                left["early_contact_frames"]
                + right["early_contact_frames"]
            )
            / 2.0
        ),
    }


def evaluate_pair(left, right, penetration_weight, stable_min_frames):
    left_frames = np.asarray(left["frame_indices"], dtype=np.int64)
    right_frames = np.asarray(right["frame_indices"], dtype=np.int64)
    common_start = max(int(left_frames[0]), int(right_frames[0]))
    common_end = min(int(left_frames[-1]), int(right_frames[-1]))
    common_frames = np.arange(common_start, common_end + 1)
    if len(common_frames) < 2:
        return None

    left_predictions = np.asarray([
        aligned_contact_sequence(left, index, common_frames)
        for index in range(len(left["candidate_contact_sequences"]))
    ])
    right_predictions = np.asarray([
        aligned_contact_sequence(right, index, common_frames)
        for index in range(len(right["candidate_contact_sequences"]))
    ])
    left_truth = aligned_ground_truth(left, common_frames)
    right_truth = aligned_ground_truth(right, common_frames)
    if not left_truth.any() and not right_truth.any():
        return None

    left_scores = (
        left_predictions.mean(axis=1)
        - penetration_weight
        * np.asarray(left["candidate_mean_penetration_distance"])
    )
    right_scores = (
        right_predictions.mean(axis=1)
        - penetration_weight
        * np.asarray(right["candidate_mean_penetration_distance"])
    )

    independent_index = (
        int(np.argmax(left_scores)),
        int(np.argmax(right_scores)),
    )
    pair_minimum = np.minimum(
        left_scores[:, None],
        right_scores[None, :],
    )
    maximin_index = np.unravel_index(
        int(np.argmax(pair_minimum)),
        pair_minimum.shape,
    )

    realized_f1 = np.empty(
        (len(left_predictions), len(right_predictions)),
        dtype=np.float64,
    )
    for left_index in range(len(left_predictions)):
        for right_index in range(len(right_predictions)):
            realized_f1[left_index, right_index] = pair_metrics(
                left_predictions[left_index],
                right_predictions[right_index],
                left_truth,
                right_truth,
                stable_min_frames,
            )["joint_f1"]
    oracle_index = np.unravel_index(
        int(np.argmax(realized_f1)),
        realized_f1.shape,
    )

    def metrics_for(index):
        return pair_metrics(
            left_predictions[index[0]],
            right_predictions[index[1]],
            left_truth,
            right_truth,
            stable_min_frames,
        )

    return {
        "sequence": left["sequence"],
        "common_frame_start": int(common_frames[0]),
        "common_frame_end": int(common_frames[-1]),
        "common_frame_count": int(len(common_frames)),
        "independent": metrics_for(independent_index),
        "maximin": metrics_for(maximin_index),
        "oracle": metrics_for(oracle_index),
        "independent_indices": list(independent_index),
        "maximin_indices": [int(value) for value in maximin_index],
        "oracle_indices": [int(value) for value in oracle_index],
    }


def mean_metric(rows, policy, metric):
    return mean_defined([
        row[policy][metric]
        for row in rows
    ])


def bootstrap_mean(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values,
        size=(samples, len(values)),
        replace=True,
    ).mean(axis=1)
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    grouped = group_events(load_events(root))
    rows = []
    for _, hands in sorted(grouped.items()):
        if "left" not in hands or "right" not in hands:
            continue
        row = evaluate_pair(
            hands["left"],
            hands["right"],
            args.penetration_weight,
            args.stable_min_frames,
        )
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("no aligned bimanual pairs were found")

    oracle_minus_independent = np.asarray([
        row["oracle"]["joint_f1"] - row["independent"]["joint_f1"]
        for row in rows
        if (
            row["oracle"]["joint_f1"] is not None
            and row["independent"]["joint_f1"] is not None
        )
    ])
    result = {
        "root": str(root),
        "protocol": {
            "penetration_weight": float(args.penetration_weight),
            "stable_min_frames": int(args.stable_min_frames),
        },
        "pair_sequence_count": len(rows),
        "policies": {
            policy: {
                metric: mean_metric(rows, policy, metric)
                for metric in (
                    "joint_f1",
                    "joint_precision",
                    "joint_recall",
                    "both_stable_contact_success",
                    "max_stable_contact_false_positive",
                    "mean_early_contact_frames",
                )
            }
            for policy in ("independent", "maximin", "oracle")
        },
        "oracle_minus_independent_f1": {
            "mean": float(oracle_minus_independent.mean()),
            "ci95": bootstrap_mean(
                oracle_minus_independent,
                args.bootstrap_samples,
                args.seed,
            ),
        },
        "maximin_changed_pair_count": int(sum(
            row["maximin_indices"] != row["independent_indices"]
            for row in rows
        )),
        "pairs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "pair_sequence_count": result["pair_sequence_count"],
        "policies": result["policies"],
        "oracle_minus_independent_f1": result[
            "oracle_minus_independent_f1"
        ],
        "maximin_changed_pair_count": result[
            "maximin_changed_pair_count"
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
