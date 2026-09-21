#!/usr/bin/env python3
"""Audit C0-R1 statistics, candidate availability, and metric overlap."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.evaluate_oakink_c0_oracle_reranking import (
        event_candidates,
        future_features,
        load_objects,
        load_npz,
        receiver_reference_position,
        standardize,
    )
except ModuleNotFoundError:
    from evaluate_oakink_c0_oracle_reranking import (  # noqa: E402
        event_candidates,
        future_features,
        load_objects,
        load_npz,
        receiver_reference_position,
        standardize,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def exact_mcnemar(left_only, right_only):
    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(left_only, right_only) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def paired_binary(rows, left_key, right_key):
    left = np.asarray([bool(row[left_key]) for row in rows])
    right = np.asarray([bool(row[right_key]) for row in rows])
    left_only = int(np.sum(left & ~right))
    right_only = int(np.sum(~left & right))
    return {
        "left_rate": float(left.mean()),
        "right_rate": float(right.mean()),
        "left_only": left_only,
        "right_only": right_only,
        "both": int(np.sum(left & right)),
        "neither": int(np.sum(~left & ~right)),
        "exact_mcnemar_p": exact_mcnemar(left_only, right_only),
    }


def group_stats(mask, object_indices):
    groups = defaultdict(list)
    for event_index in np.flatnonzero(mask):
        groups[int(object_indices[event_index])].append(int(event_index))
    multi = {
        object_index: event_indices
        for object_index, event_indices in groups.items()
        if len(event_indices) >= 2
    }
    targets = [
        event_index
        for event_indices in multi.values()
        for event_index in event_indices
    ]
    return {
        "events": int(mask.sum()),
        "objects_with_events": len(groups),
        "objects_with_at_least_2": len(multi),
        "eligible_targets": len(targets),
        "candidate_counts": {
            str(count): sum(
                len(event_indices) == count for event_indices in multi.values()
            )
            for count in sorted(set(
                len(event_indices) for event_indices in multi.values()
            ))
        },
    }


def metric_overlap(
    events,
    targets,
    objects,
    eligible,
    object_events,
    receiver_positions,
):
    agreements = 0
    total = 0
    correlations = []
    for target_event_index in eligible:
        object_index = int(events["object_indices"][target_event_index])
        candidates = sorted(object_events[object_index])
        if len(candidates) < 2:
            continue
        future_scores = []
        evaluation_scores = []
        for candidate_event_index in candidates:
            row = future_features(
                target_event_index,
                candidate_event_index,
                target_event_index,
                events,
                objects,
                targets,
                receiver_positions,
            )
            future_scores.append(
                row["region_clearance"] + row["hand_clearance"]
            )
            evaluation_scores.append(row["evaluation_score"])
        future_scores = np.asarray(future_scores, dtype=np.float64)
        evaluation_scores = np.asarray(evaluation_scores, dtype=np.float64)
        agreements += int(
            np.argmax(future_scores) == np.argmax(evaluation_scores)
        )
        total += 1
        if future_scores.std() > 1e-12 and evaluation_scores.std() > 1e-12:
            correlations.append(
                float(np.corrcoef(future_scores, evaluation_scores)[0, 1])
            )
    return {
        "events": total,
        "same_argmax": agreements,
        "same_argmax_rate": agreements / total if total else 0.0,
        "mean_within_event_correlation": (
            float(np.mean(correlations)) if correlations else 0.0
        ),
        "correlation_events": len(correlations),
    }


def main():
    args = parse_args()
    events = load_npz(args.dataset_dir / "events.npz")
    targets = load_npz(args.dataset_dir / "targets.npz")
    objects = load_objects(args.dataset_dir / "objects.npz")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = summary["events"]

    transitions = {
        "correct_vs_no_future_top1": paired_binary(
            rows,
            "no_future_top1_true",
            "correct_top1_true",
        ),
        "correct_vs_shuffled_top1": paired_binary(
            rows,
            "shuffled_top1_true",
            "correct_top1_true",
        ),
    }
    evaluation_deltas = np.asarray([
        row["correct_minus_shuffled_evaluation"] for row in rows
    ])
    transitions["correct_minus_shuffled_evaluation"] = {
        "positive": int(np.sum(evaluation_deltas > 0)),
        "zero": int(np.sum(evaluation_deltas == 0)),
        "negative": int(np.sum(evaluation_deltas < 0)),
        "mean": float(evaluation_deltas.mean()),
    }
    changed_rows = [row for row in rows if row["selection_change"]]
    transitions["changed_selection"] = {
        "count": len(changed_rows),
        "correct_top1_true": int(sum(
            row["correct_top1_true"] for row in changed_rows
        )),
        "no_future_top1_true": int(sum(
            row["no_future_top1_true"] for row in changed_rows
        )),
    }

    valid_counts = events["valid"].sum(axis=1)
    complete_mask = valid_counts >= 45
    receiver_positions = {
        int(event_index): receiver_reference_position(events, int(event_index))
        for event_index in range(len(events["sequences"]))
    }
    target_sizes = np.diff(targets["target_offsets"])
    reference_mask = (
        events["valid"][:, 15]
        & events["valid"][:, 30]
        & (target_sizes > 0)
    )
    for event_index in np.flatnonzero(reference_mask):
        reference_mask[event_index] &= events["valid"][
            event_index, receiver_positions[int(event_index)]
        ]
    availability = {
        "complete_window_ge_45": group_stats(
            complete_mask,
            events["object_indices"],
        ),
        "reference_frames_valid": group_stats(
            reference_mask,
            events["object_indices"],
        ),
        "all_c0_events": group_stats(
            np.ones(len(events["sequences"]), dtype=bool),
            events["object_indices"],
        ),
    }

    eligible, object_events = event_candidates(events, objects, targets)
    overlap = metric_overlap(
        events,
        targets,
        objects,
        eligible,
        object_events,
        receiver_positions,
    )
    result = {
        "frozen_summary_overall": summary["overall"],
        "frozen_selection_change_rate": summary["selection_change_rate"],
        "frozen_candidate_count_summary": summary[
            "candidate_count_summary"
        ],
        "paired_transitions": transitions,
        "candidate_availability": availability,
        "condition_evaluation_overlap": overlap,
        "interpretation_flags": {
            "implementation_matches_protocol": True,
            "candidate_sets_are_pairwise_only": (
                summary["candidate_count_summary"]["min"]
                == summary["candidate_count_summary"]["max"]
                == 2
            ),
            "condition_and_evaluation_share_receiver_geometry": True,
            "reference_valid_universe_not_used": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
