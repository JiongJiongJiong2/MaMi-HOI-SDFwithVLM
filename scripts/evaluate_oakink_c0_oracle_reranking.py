#!/usr/bin/env python3
"""Evaluate oracle future-target reranking on the OakInk C0 dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


PRIMARY_WEIGHT = 0.5
SENSITIVITY_WEIGHTS = (0.25, 0.5, 1.0)
CONTACT_THRESHOLD_M = 0.005
REGION_CLEARANCE_SCALE_M = 0.030
HAND_CLEARANCE_SCALE_M = 0.050
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260921
MINIMUM_ELIGIBLE_EVENTS = 40
MINIMUM_SELECTION_CHANGE_RATE = 0.25
MAXIMUM_TOP1_RATE_DROP = 0.10


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): to_builtin(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def load_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {
            name: np.asarray(arrays[name])
            for name in arrays.files
        }


def load_objects(path):
    arrays = load_npz(path)
    objects = []
    for index in range(len(arrays["object_ids"])):
        objects.append({
            "object_id": str(arrays["object_ids"][index]),
            "object_name": str(arrays["object_names"][index]),
            "vertices": np.asarray(
                arrays[f"vertices_{index}"],
                dtype=np.float64,
            ),
        })
    return objects


def homogeneous(points):
    points = np.asarray(points, dtype=np.float64)
    return np.concatenate(
        (points, np.ones((len(points), 1), dtype=np.float64)),
        axis=1,
    )


def to_object_frame(points, transform):
    inverse = np.linalg.inv(np.asarray(transform, dtype=np.float64))
    return (inverse @ homogeneous(points).T).T[:, :3]


def clipped_distance(value, scale):
    return float(np.clip(value / scale, 0.0, 1.0))


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    std = values.std()
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / std


def receiver_reference_position(event, event_index):
    release = int(event["release_indices"][event_index])
    receiver_start = int(event["receiver_starts"][event_index])
    receiver_end = int(event["receiver_ends"][event_index])
    valid = event["valid"][event_index]
    start_position = receiver_start - release + 30
    end_position = receiver_end - release + 30
    positions = np.arange(0, len(valid))
    candidates = positions[
        valid
        & (positions >= max(0, start_position))
        & (positions <= min(len(valid) - 1, end_position))
    ]
    if len(candidates):
        return int(candidates[0])
    contact_positions = positions[
        valid
        & (event["receiver_distance_m"][event_index] <= CONTACT_THRESHOLD_M)
    ]
    if len(contact_positions):
        return int(contact_positions[0])
    return int(30 if valid[30] else np.flatnonzero(valid)[0])


def candidate_contact_features(candidate_hand, object_vertices):
    distances = cKDTree(object_vertices).query(
        candidate_hand,
        k=1,
        workers=1,
    )[0]
    nearest = np.sort(distances)[:min(50, len(distances))]
    return {
        "contact_score": -float(nearest.mean()),
        "coverage": float(
            np.mean(distances <= CONTACT_THRESHOLD_M)
        ),
    }


def clearance_features(candidate_hand, region_vertices, receiver_hand):
    region_distance = float(
        cKDTree(region_vertices).query(
            candidate_hand,
            k=1,
            workers=1,
        )[0].min()
    )
    hand_distances = cKDTree(receiver_hand).query(
        candidate_hand,
        k=1,
        workers=1,
    )[0]
    hand_distance = float(np.quantile(hand_distances, 0.10))
    return {
        "region_clearance": clipped_distance(
            region_distance,
            REGION_CLEARANCE_SCALE_M,
        ),
        "hand_clearance": clipped_distance(
            hand_distance,
            HAND_CLEARANCE_SCALE_M,
        ),
        "region_distance_m": region_distance,
        "hand_distance_m": hand_distance,
    }


def select_top(scores, candidate_event_indices):
    order = np.argsort(candidate_event_indices, kind="stable")
    sorted_scores = np.asarray(scores)[order]
    best_local = int(np.argmax(sorted_scores))
    return int(np.asarray(candidate_event_indices)[order][best_local])


def rank_of(candidate_event_indices, scores, target_event_index):
    order = np.argsort(candidate_event_indices, kind="stable")
    sorted_events = np.asarray(candidate_event_indices)[order]
    sorted_scores = np.asarray(scores)[order]
    descending = np.argsort(-sorted_scores, kind="stable")
    ranked_events = sorted_events[descending]
    return int(np.flatnonzero(ranked_events == target_event_index)[0]) + 1


def grouped_bootstrap(deltas, groups):
    deltas = np.asarray(deltas, dtype=np.float64)
    groups = np.asarray(groups)
    unique_groups = sorted(set(groups.tolist()))
    if not unique_groups:
        return {
            "mean": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "draws": BOOTSTRAP_DRAWS,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.zeros(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        sampled_values = []
        for group in sampled_groups:
            sampled_values.extend(deltas[groups == group].tolist())
        means[draw] = float(np.mean(sampled_values))
    return {
        "mean": float(deltas.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "draws": BOOTSTRAP_DRAWS,
    }


def event_candidates(events, objects, targets):
    valid_counts = events["valid"].sum(axis=1)
    eligible = np.flatnonzero(valid_counts >= 45)
    object_events = defaultdict(list)
    for event_index in eligible:
        object_events[int(events["object_indices"][event_index])].append(
            int(event_index)
        )
    return eligible, object_events


def future_features(
    target_event_index,
    candidate_event_index,
    condition_event_index,
    events,
    objects,
    targets,
    receiver_positions,
):
    candidate_object_index = int(
        events["object_indices"][candidate_event_index]
    )
    object_vertices = objects[candidate_object_index]["vertices"]
    candidate_hand = to_object_frame(
        events["giver_hand_vertices"][candidate_event_index, 15],
        events["object_transforms"][candidate_event_index, 15],
    )
    contact = candidate_contact_features(candidate_hand, object_vertices)

    offsets = targets["target_offsets"]
    region_indices = targets["target_indices"][
        offsets[condition_event_index]:offsets[condition_event_index + 1]
    ]
    region_vertices = object_vertices[region_indices]
    receiver_position = receiver_positions[condition_event_index]
    receiver_hand = to_object_frame(
        events["receiver_hand_vertices"][
            condition_event_index, receiver_position
        ],
        events["object_transforms"][
            condition_event_index, receiver_position
        ],
    )
    clearance = clearance_features(
        candidate_hand,
        region_vertices,
        receiver_hand,
    )
    evaluation_receiver = to_object_frame(
        events["receiver_hand_vertices"][
            target_event_index, receiver_positions[target_event_index]
        ],
        events["object_transforms"][
            target_event_index, receiver_positions[target_event_index]
        ],
    )
    evaluation_region = object_vertices[
        targets["target_indices"][
            offsets[target_event_index]:offsets[target_event_index + 1]
        ]
    ]
    evaluation = clearance_features(
        candidate_hand,
        evaluation_region,
        evaluation_receiver,
    )
    evaluation_score = (
        contact["coverage"]
        + 0.5 * evaluation["region_clearance"]
        + 0.5 * evaluation["hand_clearance"]
    )
    return {
        **contact,
        **clearance,
        "evaluation_score": evaluation_score,
        "evaluation_region_clearance": evaluation["region_clearance"],
        "evaluation_hand_clearance": evaluation["hand_clearance"],
    }


def main():
    args = parse_args()
    events = load_npz(args.dataset_dir / "events.npz")
    targets = load_npz(args.dataset_dir / "targets.npz")
    objects = load_objects(args.dataset_dir / "objects.npz")
    eligible, object_events = event_candidates(events, objects, targets)
    receiver_positions = {
        int(event_index): receiver_reference_position(events, event_index)
        for event_index in eligible
    }

    rows = []
    for target_event_index in eligible:
        object_index = int(events["object_indices"][target_event_index])
        candidates = sorted(object_events[object_index])
        if len(candidates) < 2:
            continue
        donor_candidates = [
            event_index
            for event_index in candidates
            if event_index != target_event_index
        ]
        if not donor_candidates:
            continue
        donor_event_index = donor_candidates[0]

        feature_rows = []
        for candidate_event_index in candidates:
            no_future = future_features(
                target_event_index,
                candidate_event_index,
                candidate_event_index,
                events,
                objects,
                targets,
                receiver_positions,
            )
            correct = future_features(
                target_event_index,
                candidate_event_index,
                target_event_index,
                events,
                objects,
                targets,
                receiver_positions,
            )
            shuffled = future_features(
                target_event_index,
                candidate_event_index,
                donor_event_index,
                events,
                objects,
                targets,
                receiver_positions,
            )
            feature_rows.append({
                "candidate_event_index": candidate_event_index,
                "no_future_contact": no_future["contact_score"],
                "correct_contact": correct["contact_score"],
                "correct_region": correct["region_clearance"],
                "correct_hand": correct["hand_clearance"],
                "shuffled_contact": shuffled["contact_score"],
                "shuffled_region": shuffled["region_clearance"],
                "shuffled_hand": shuffled["hand_clearance"],
                "evaluation_score": correct["evaluation_score"],
            })

        contact_z = standardize([
            row["correct_contact"] for row in feature_rows
        ])
        correct_region_z = standardize([
            row["correct_region"] for row in feature_rows
        ])
        correct_hand_z = standardize([
            row["correct_hand"] for row in feature_rows
        ])
        shuffled_region_z = standardize([
            row["shuffled_region"] for row in feature_rows
        ])
        shuffled_hand_z = standardize([
            row["shuffled_hand"] for row in feature_rows
        ])
        candidate_indices = np.asarray(candidates, dtype=np.int64)
        primary_no_future = contact_z
        primary_correct = (
            contact_z
            + PRIMARY_WEIGHT * correct_region_z
            + PRIMARY_WEIGHT * correct_hand_z
        )
        primary_shuffled = (
            contact_z
            + PRIMARY_WEIGHT * shuffled_region_z
            + PRIMARY_WEIGHT * shuffled_hand_z
        )
        selected_no_future = select_top(
            primary_no_future,
            candidate_indices,
        )
        selected_correct = select_top(
            primary_correct,
            candidate_indices,
        )
        selected_shuffled = select_top(
            primary_shuffled,
            candidate_indices,
        )
        evaluation_scores = np.asarray([
            row["evaluation_score"] for row in feature_rows
        ])
        lookup = {
            row["candidate_event_index"]: index
            for index, row in enumerate(feature_rows)
        }
        sensitivity = {}
        for weight in SENSITIVITY_WEIGHTS:
            correct_scores = (
                contact_z
                + weight * correct_region_z
                + weight * correct_hand_z
            )
            shuffled_scores = (
                contact_z
                + weight * shuffled_region_z
                + weight * shuffled_hand_z
            )
            sensitivity[f"{weight:.2f}"] = {
                "selection_change": (
                    select_top(correct_scores, candidate_indices)
                    != select_top(contact_z, candidate_indices)
                ),
                "correct_minus_shuffled_evaluation": (
                    float(evaluation_scores[lookup[
                        select_top(correct_scores, candidate_indices)
                    ]])
                    - float(evaluation_scores[lookup[
                        select_top(shuffled_scores, candidate_indices)
                    ]])
                ),
            }

        rows.append({
            "target_event_index": target_event_index,
            "sequence": str(events["sequences"][target_event_index]),
            "split": str(events["splits"][target_event_index]),
            "object_index": object_index,
            "object_id": str(events["object_ids"][target_event_index]),
            "candidate_count": len(candidates),
            "donor_event_index": donor_event_index,
            "selected_no_future": selected_no_future,
            "selected_correct": selected_correct,
            "selected_shuffled": selected_shuffled,
            "no_future_top1_true": selected_no_future == target_event_index,
            "correct_top1_true": selected_correct == target_event_index,
            "shuffled_top1_true": selected_shuffled == target_event_index,
            "correct_rank_true": rank_of(
                candidate_indices,
                primary_correct,
                target_event_index,
            ),
            "no_future_rank_true": rank_of(
                candidate_indices,
                primary_no_future,
                target_event_index,
            ),
            "selection_change": selected_correct != selected_no_future,
            "correct_evaluation_score": float(
                evaluation_scores[lookup[selected_correct]]
            ),
            "shuffled_evaluation_score": float(
                evaluation_scores[lookup[selected_shuffled]]
            ),
            "correct_minus_shuffled_evaluation": (
                float(evaluation_scores[lookup[selected_correct]])
                - float(evaluation_scores[lookup[selected_shuffled]])
            ),
            "sensitivity": sensitivity,
        })

    if not rows:
        raise RuntimeError("no eligible C0-R1 events")
    deltas = [row["correct_minus_shuffled_evaluation"] for row in rows]
    groups = [row["object_index"] for row in rows]
    bootstrap = grouped_bootstrap(deltas, groups)
    selection_change_rate = float(np.mean([
        row["selection_change"] for row in rows
    ]))
    no_future_top1_rate = float(np.mean([
        row["no_future_top1_true"] for row in rows
    ]))
    correct_top1_rate = float(np.mean([
        row["correct_top1_true"] for row in rows
    ]))
    sensitivity_consistency = {}
    for weight in SENSITIVITY_WEIGHTS:
        key = f"{weight:.2f}"
        sensitivity_consistency[key] = {
            "selection_change_rate": float(np.mean([
                row["sensitivity"][key]["selection_change"]
                for row in rows
            ])),
            "correct_minus_shuffled_mean": float(np.mean([
                row["sensitivity"][key][
                    "correct_minus_shuffled_evaluation"
                ]
                for row in rows
            ])),
        }
    checks = {
        "eligible_events_ge_40": len(rows) >= MINIMUM_ELIGIBLE_EVENTS,
        "selection_change_rate_ge_0_25": (
            selection_change_rate >= MINIMUM_SELECTION_CHANGE_RATE
        ),
        "correct_beats_shuffled_ci": bootstrap["lower"] > 0.0,
        "top1_observed_rate_not_dropped": (
            no_future_top1_rate - correct_top1_rate
            <= MAXIMUM_TOP1_RATE_DROP
        ),
        "secondary_direction_consistent": all(
            row["selection_change_rate"]
            >= MINIMUM_SELECTION_CHANGE_RATE
            and row["correct_minus_shuffled_mean"] > 0.0
            for row in sensitivity_consistency.values()
        ),
    }
    result = to_builtin({
        "eligible_events": len(rows),
        "candidate_count_summary": {
            "min": int(min(row["candidate_count"] for row in rows)),
            "max": int(max(row["candidate_count"] for row in rows)),
            "mean": float(np.mean([
                row["candidate_count"] for row in rows
            ])),
        },
        "selection_change_rate": selection_change_rate,
        "top1_observed_rate": {
            "no_future": no_future_top1_rate,
            "correct": correct_top1_rate,
            "shuffled": float(np.mean([
                row["shuffled_top1_true"] for row in rows
            ])),
        },
        "correct_minus_shuffled_evaluation": bootstrap,
        "sensitivity": sensitivity_consistency,
        "checks": checks,
        "overall": all(checks.values()),
        "events": rows,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(to_builtin({
        key: result[key]
        for key in (
            "eligible_events",
            "candidate_count_summary",
            "selection_change_rate",
            "top1_observed_rate",
            "correct_minus_shuffled_evaluation",
            "sensitivity",
            "checks",
            "overall",
        )
    }), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
