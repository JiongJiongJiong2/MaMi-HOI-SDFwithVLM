#!/usr/bin/env python3
"""Evaluate C0-R2 with decoupled conditioning and evaluation geometry."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from scripts.evaluate_oakink_c0_oracle_reranking import (
        candidate_contact_features,
        grouped_bootstrap,
        load_npz,
        load_objects,
        rank_of,
        receiver_reference_position,
        select_top,
        standardize,
        to_builtin,
        to_object_frame,
    )
except ModuleNotFoundError:
    from evaluate_oakink_c0_oracle_reranking import (  # noqa: E402
        candidate_contact_features,
        grouped_bootstrap,
        load_npz,
        load_objects,
        rank_of,
        receiver_reference_position,
        select_top,
        standardize,
        to_builtin,
        to_object_frame,
    )


PRIMARY_WEIGHT = 0.5
SENSITIVITY_WEIGHTS = (0.25, 0.5, 1.0)
CONTACT_THRESHOLD_M = 0.005
REGION_CLEARANCE_SCALE_M = 0.030
JOINT_CLEARANCE_SCALE_M = 0.050
JOINT_COLLISION_M = 0.005
MINIMUM_ELIGIBLE_EVENTS = 60
MINIMUM_ORACLE_TOP1_RATE = 0.55
MAXIMUM_TOP1_DROP = 0.05


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
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


def reference_valid(events, targets, receiver_positions):
    target_sizes = np.diff(targets["target_offsets"])
    mask = (
        events["valid"][:, 15]
        & events["valid"][:, 30]
        & (target_sizes > 0)
    )
    for event_index in np.flatnonzero(mask):
        mask[event_index] &= events["valid"][
            event_index, receiver_positions[int(event_index)]
        ]
    return mask


def region_features(candidate_hand, region_vertices):
    distance = float(
        cKDTree(region_vertices).query(
            candidate_hand,
            k=1,
            workers=1,
        )[0].min()
    )
    return {
        "region_clearance": float(np.clip(
            distance / REGION_CLEARANCE_SCALE_M,
            0.0,
            1.0,
        )),
        "region_distance_m": distance,
    }


def joint_features(candidate_hand, receiver_joints):
    distances = cKDTree(receiver_joints).query(
        candidate_hand,
        k=1,
        workers=1,
    )[0]
    joint_distance = float(np.quantile(distances, 0.10))
    return {
        "joint_clearance": float(np.clip(
            joint_distance / JOINT_CLEARANCE_SCALE_M,
            0.0,
            1.0,
        )),
        "joint_distance_m": joint_distance,
        "collision_fraction": float(
            np.mean(distances <= JOINT_COLLISION_M)
        ),
    }


def build_candidate_row(
    target_event_index,
    candidate_event_index,
    condition_event_index,
    events,
    objects,
    targets,
    receiver_positions,
):
    object_index = int(events["object_indices"][candidate_event_index])
    object_vertices = objects[object_index]["vertices"]
    candidate_hand = to_object_frame(
        events["giver_hand_vertices"][candidate_event_index, 15],
        events["object_transforms"][candidate_event_index, 15],
    )
    contact = candidate_contact_features(candidate_hand, object_vertices)
    offsets = targets["target_offsets"]
    condition_region = object_vertices[
        targets["target_indices"][
            offsets[condition_event_index]:offsets[condition_event_index + 1]
        ]
    ]
    condition = region_features(candidate_hand, condition_region)
    receiver_position = receiver_positions[target_event_index]
    receiver_joints = to_object_frame(
        events["receiver_hand_joints"][
            target_event_index, receiver_position
        ],
        events["object_transforms"][
            target_event_index, receiver_position
        ],
    )
    evaluation = joint_features(candidate_hand, receiver_joints)
    evaluation_score = (
        contact["coverage"]
        + 0.5 * evaluation["joint_clearance"]
        - evaluation["collision_fraction"]
    )
    return {
        "candidate_event_index": candidate_event_index,
        "contact_score": contact["contact_score"],
        "contact_coverage": contact["coverage"],
        "region_clearance": condition["region_clearance"],
        "joint_clearance": evaluation["joint_clearance"],
        "collision_fraction": evaluation["collision_fraction"],
        "evaluation_score": evaluation_score,
    }


def paired_mean_interval(left_scores, right_scores):
    deltas = np.asarray(left_scores) - np.asarray(right_scores)
    return deltas


def main():
    args = parse_args()
    events = load_npz(args.dataset_dir / "events.npz")
    targets = load_npz(args.dataset_dir / "targets.npz")
    objects = load_objects(args.dataset_dir / "objects.npz")
    receiver_positions = {
        event_index: receiver_reference_position(events, event_index)
        for event_index in range(len(events["sequences"]))
    }
    valid_mask = reference_valid(events, targets, receiver_positions)
    eligible = np.flatnonzero(valid_mask)
    object_events = defaultdict(list)
    for event_index in eligible:
        object_events[int(events["object_indices"][event_index])].append(
            int(event_index)
        )

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
        donor_event_index = donor_candidates[0]
        feature_rows = []
        for candidate_event_index in candidates:
            correct = build_candidate_row(
                target_event_index,
                candidate_event_index,
                target_event_index,
                events,
                objects,
                targets,
                receiver_positions,
            )
            shuffled = build_candidate_row(
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
                "contact_score": correct["contact_score"],
                "contact_coverage": correct["contact_coverage"],
                "correct_region": correct["region_clearance"],
                "shuffled_region": shuffled["region_clearance"],
                "joint_clearance": correct["joint_clearance"],
                "collision_fraction": correct["collision_fraction"],
                "evaluation_score": correct["evaluation_score"],
            })

        contact_z = standardize([
            row["contact_score"] for row in feature_rows
        ])
        correct_region_z = standardize([
            row["correct_region"] for row in feature_rows
        ])
        shuffled_region_z = standardize([
            row["shuffled_region"] for row in feature_rows
        ])
        candidate_indices = np.asarray(candidates, dtype=np.int64)
        evaluation_scores = np.asarray([
            row["evaluation_score"] for row in feature_rows
        ])
        collision_fractions = np.asarray([
            row["collision_fraction"] for row in feature_rows
        ])
        lookup = {
            row["candidate_event_index"]: index
            for index, row in enumerate(feature_rows)
        }
        no_future_scores = contact_z
        correct_scores = (
            contact_z + PRIMARY_WEIGHT * correct_region_z
        )
        shuffled_scores = (
            contact_z + PRIMARY_WEIGHT * shuffled_region_z
        )
        selected_no_future = select_top(no_future_scores, candidate_indices)
        selected_correct = select_top(correct_scores, candidate_indices)
        selected_shuffled = select_top(shuffled_scores, candidate_indices)
        selected_oracle = select_top(evaluation_scores, candidate_indices)
        delta_no_future = float(
            evaluation_scores[lookup[selected_correct]]
            - evaluation_scores[lookup[selected_no_future]]
        )
        delta_shuffled = float(
            evaluation_scores[lookup[selected_correct]]
            - evaluation_scores[lookup[selected_shuffled]]
        )
        collision_delta_shuffled = float(
            collision_fractions[lookup[selected_correct]]
            - collision_fractions[lookup[selected_shuffled]]
        )
        sensitivity = {}
        for weight in SENSITIVITY_WEIGHTS:
            correct_weight_scores = (
                contact_z + weight * correct_region_z
            )
            shuffled_weight_scores = (
                contact_z + weight * shuffled_region_z
            )
            selected_correct_weight = select_top(
                correct_weight_scores,
                candidate_indices,
            )
            selected_shuffled_weight = select_top(
                shuffled_weight_scores,
                candidate_indices,
            )
            sensitivity[f"{weight:.2f}"] = {
                "correct_minus_no_future": float(
                    evaluation_scores[lookup[selected_correct_weight]]
                    - evaluation_scores[lookup[selected_no_future]]
                ),
                "correct_minus_shuffled": float(
                    evaluation_scores[lookup[selected_correct_weight]]
                    - evaluation_scores[lookup[selected_shuffled_weight]]
                ),
            }

        rows.append({
            "target_event_index": target_event_index,
            "sequence": str(events["sequences"][target_event_index]),
            "split": str(events["splits"][target_event_index]),
            "object_index": object_index,
            "object_id": str(events["object_ids"][target_event_index]),
            "candidate_count": len(candidates),
            "no_future_top1_true": selected_no_future == target_event_index,
            "correct_top1_true": selected_correct == target_event_index,
            "shuffled_top1_true": selected_shuffled == target_event_index,
            "oracle_top1_true": selected_oracle == target_event_index,
            "correct_rank_true": rank_of(
                candidate_indices,
                correct_scores,
                target_event_index,
            ),
            "no_future_rank_true": rank_of(
                candidate_indices,
                no_future_scores,
                target_event_index,
            ),
            "correct_minus_no_future_evaluation": delta_no_future,
            "correct_minus_shuffled_evaluation": delta_shuffled,
            "correct_minus_shuffled_collision": collision_delta_shuffled,
            "sensitivity": sensitivity,
        })

    if not rows:
        raise RuntimeError("no C0-R2 target events")
    groups = [row["object_index"] for row in rows]
    eval_no_future = grouped_bootstrap(
        [row["correct_minus_no_future_evaluation"] for row in rows],
        groups,
    )
    eval_shuffled = grouped_bootstrap(
        [row["correct_minus_shuffled_evaluation"] for row in rows],
        groups,
    )
    collision_shuffled = grouped_bootstrap(
        [row["correct_minus_shuffled_collision"] for row in rows],
        groups,
    )
    no_future_top1_rate = float(np.mean([
        row["no_future_top1_true"] for row in rows
    ]))
    correct_top1_rate = float(np.mean([
        row["correct_top1_true"] for row in rows
    ]))
    oracle_top1_rate = float(np.mean([
        row["oracle_top1_true"] for row in rows
    ]))
    correct_no_future = np.asarray([
        row["no_future_top1_true"] for row in rows
    ])
    correct_top1 = np.asarray([
        row["correct_top1_true"] for row in rows
    ])
    no_future_only = int(np.sum(correct_no_future & ~correct_top1))
    correct_only = int(np.sum(~correct_no_future & correct_top1))
    sensitivity = {}
    for weight in SENSITIVITY_WEIGHTS:
        key = f"{weight:.2f}"
        sensitivity[key] = {
            "correct_minus_no_future_mean": float(np.mean([
                row["sensitivity"][key]["correct_minus_no_future"]
                for row in rows
            ])),
            "correct_minus_shuffled_mean": float(np.mean([
                row["sensitivity"][key]["correct_minus_shuffled"]
                for row in rows
            ])),
        }
    sensitivity_ok = all(
        sensitivity[key]["correct_minus_no_future_mean"] > 0.0
        and sensitivity[key]["correct_minus_shuffled_mean"] > 0.0
        for key in ("0.50", "1.00")
    )
    checks = {
        "eligible_events_ge_60": len(rows) >= MINIMUM_ELIGIBLE_EVENTS,
        "oracle_top1_rate_ge_0_55": (
            oracle_top1_rate >= MINIMUM_ORACLE_TOP1_RATE
        ),
        "correct_beats_no_future_ci": eval_no_future["lower"] > 0.0,
        "correct_beats_shuffled_ci": eval_shuffled["lower"] > 0.0,
        "top1_observed_rate_not_dropped": (
            no_future_top1_rate - correct_top1_rate
            <= MAXIMUM_TOP1_DROP
        ),
        "collision_not_worse_than_shuffled": (
            collision_shuffled["upper"] < 0.0
        ),
        "sensitivity_direction": sensitivity_ok,
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
        "top1_observed_rate": {
            "no_future": no_future_top1_rate,
            "correct": correct_top1_rate,
            "shuffled": float(np.mean([
                row["shuffled_top1_true"] for row in rows
            ])),
            "oracle": oracle_top1_rate,
        },
        "correct_vs_no_future_rank_transition": {
            "no_future_only": no_future_only,
            "correct_only": correct_only,
            "exact_mcnemar_p": exact_mcnemar(
                no_future_only,
                correct_only,
            ),
        },
        "correct_minus_no_future_evaluation": eval_no_future,
        "correct_minus_shuffled_evaluation": eval_shuffled,
        "correct_minus_shuffled_collision": collision_shuffled,
        "sensitivity": sensitivity,
        "checks": checks,
        "overall": all(checks.values()),
        "events": rows,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        __import__("json").dumps(
            result,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(__import__("json").dumps({
        key: result[key]
        for key in (
            "eligible_events",
            "candidate_count_summary",
            "top1_observed_rate",
            "correct_vs_no_future_rank_transition",
            "correct_minus_no_future_evaluation",
            "correct_minus_shuffled_evaluation",
            "correct_minus_shuffled_collision",
            "sensitivity",
            "checks",
            "overall",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
