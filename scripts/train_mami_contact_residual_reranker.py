#!/usr/bin/env python3
"""Sequence-held-out residual reranker for saved MaMi contact candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "action_magnitude",
    "action_smoothness",
    "clearance_abs",
    "contact_probability",
    "frame_contact_fraction",
    "vertex_contact_fraction",
    "min_distance",
    "mean_penetration_distance",
    "penetration",
)
HAND_NAMES = ("left", "right")
DEFAULT_CANDIDATE_NAMES = (
    "base",
    "inward",
    "outward",
    "zero_palm",
    "random_0",
    "random_1",
    "random_2",
    "random_3",
    "random_4",
    "random_5",
    "random_6",
    "random_7",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--penetration-weight", type=float, default=20.0)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def load_events(paths):
    events = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for event in payload["events"]:
            event = dict(event)
            event["_source"] = str(path)
            events.append(event)
    return events


def candidate_feature(event, candidate_index, penetration_weight):
    names = event.get("candidate_names", DEFAULT_CANDIDATE_NAMES)
    name = names[candidate_index]
    hand = event["hand"]
    vector = [
        event["candidate_action_magnitude"][candidate_index],
        event["candidate_action_smoothness"][candidate_index],
        event["candidate_clearance_abs"][candidate_index],
        event["candidate_contact_probability"][candidate_index],
        event["candidate_frame_contact_fraction"][candidate_index],
        event["candidate_vertex_contact_fraction"][candidate_index],
        event["candidate_min_distance"][candidate_index],
        event["candidate_mean_penetration_distance"][candidate_index],
        event["candidate_penetration"][candidate_index],
    ]
    vector.extend(
        1.0 if hand == hand_name else 0.0
        for hand_name in HAND_NAMES
    )
    name_categories = (
        "base",
        "inward",
        "outward",
        "zero_palm",
        "random_0",
        "random_1",
        "random_2",
        "random_3",
        "random_4",
        "random_5",
        "random_6",
        "random_7",
    )
    vector.extend(1.0 if name == value else 0.0 for value in name_categories)
    geometry = (
        event["candidate_frame_contact_fraction"][candidate_index]
        - penetration_weight
        * event["candidate_mean_penetration_distance"][candidate_index]
    )
    utility = (
        event["candidate_f1"][candidate_index]
        - penetration_weight
        * event["candidate_mean_penetration_distance"][candidate_index]
    )
    return np.asarray(vector, dtype=np.float64), geometry, utility


def build_rows(events, penetration_weight):
    rows = []
    for event_index, event in enumerate(events):
        names = event.get("candidate_names", DEFAULT_CANDIDATE_NAMES)
        if len(names) != len(event["candidate_f1"]):
            raise ValueError("candidate names do not match candidate arrays")
        for candidate_index, name in enumerate(names):
            features, geometry, utility = candidate_feature(
                event,
                candidate_index,
                penetration_weight,
            )
            rows.append({
                "event_index": event_index,
                "sequence": event["sequence"],
                "hand": event["hand"],
                "candidate_index": candidate_index,
                "candidate_name": name,
                "features": features,
                "geometry": geometry,
                "utility": utility,
                "f1": float(event["candidate_f1"][candidate_index]),
                "penetration": float(
                    event["candidate_mean_penetration_distance"][
                        candidate_index
                    ]
                ),
            })
    return rows


def fit_ridge(train_x, train_y, ridge):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    normalized = (train_x - mean) / scale
    design = np.concatenate([
        np.ones((normalized.shape[0], 1)),
        normalized,
    ], axis=1)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ train_y,
    )
    return mean, scale, coefficients


def predict_ridge(model, features):
    mean, scale, coefficients = model
    normalized = (features - mean) / scale
    design = np.concatenate([
        np.ones((normalized.shape[0], 1)),
        normalized,
    ], axis=1)
    return design @ coefficients


def select_per_event(rows, event_count, score_key):
    selected = []
    for event_index in range(event_count):
        indices = [
            index
            for index, row in enumerate(rows)
            if row["event_index"] == event_index
        ]
        best = max(indices, key=lambda index: rows[index][score_key])
        selected.append(rows[best])
    return selected


def summarize(selected):
    return {
        "events": len(selected),
        "mean_f1": float(np.mean([row["f1"] for row in selected])),
        "mean_penetration_m": float(
            np.mean([row["penetration"] for row in selected])
        ),
        "selected_name_counts": {
            name: sum(
                row["candidate_name"] == name
                for row in selected
            )
            for name in sorted(set(
                row["candidate_name"] for row in selected
            ))
        },
    }


def bootstrap_difference(
    selected_a,
    selected_b,
    metric,
    samples,
    seed,
):
    groups = sorted(set(row["sequence"] for row in selected_a))
    values_a = {
        sequence: np.mean([
            row[metric]
            for row in selected_a
            if row["sequence"] == sequence
        ])
        for sequence in groups
    }
    values_b = {
        sequence: np.mean([
            row[metric]
            for row in selected_b
            if row["sequence"] == sequence
        ])
        for sequence in groups
    }
    differences = np.asarray([
        values_a[sequence] - values_b[sequence]
        for sequence in groups
    ])
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        differences,
        size=(samples, len(differences)),
        replace=True,
    ).mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def main():
    args = parse_args()
    events = load_events(args.events)
    rows = build_rows(events, args.penetration_weight)
    feature_matrix = np.stack([row["features"] for row in rows])
    residual_target = np.asarray([
        row["utility"] - row["geometry"]
        for row in rows
    ])
    sequences = sorted(set(event["sequence"] for event in events))
    predictions = np.zeros(len(rows), dtype=np.float64)
    for sequence in sequences:
        test_mask = np.asarray([
            row["sequence"] == sequence
            for row in rows
        ])
        train_mask = ~test_mask
        model = fit_ridge(
            feature_matrix[train_mask],
            residual_target[train_mask],
            args.ridge,
        )
        predictions[test_mask] = predict_ridge(
            model,
            feature_matrix[test_mask],
        )
    for row, residual in zip(rows, predictions):
        row["residual_score"] = row["geometry"] + residual
    selected_base = select_per_event(rows, len(events), "geometry") 
    # Replace base by the actual candidate named base for the no-model arm.
    selected_base = [
        next(
            row for row in rows
            if row["event_index"] == event_index
            and row["candidate_name"] == "base"
        )
        for event_index in range(len(events))
    ]
    selected_geometry = select_per_event(
        rows,
        len(events),
        "geometry",
    )
    selected_residual = select_per_event(
        rows,
        len(events),
        "residual_score",
    )
    selected_oracle = select_per_event(
        rows,
        len(events),
        "f1",
    )
    random_rows = [
        row for row in rows
        if row["candidate_name"].startswith("random_")
    ]
    random_mean_f1 = float(np.mean([row["f1"] for row in random_rows]))
    random_mean_penetration = float(np.mean([
        row["penetration"] for row in random_rows
    ]))
    result = {
        "event_count": len(events),
        "sequence_count": len(sequences),
        "base": summarize(selected_base),
        "geometry": summarize(selected_geometry),
        "random_mean": {
            "mean_f1": random_mean_f1,
            "mean_penetration_m": random_mean_penetration,
        },
        "residual": summarize(selected_residual),
        "oracle": summarize(selected_oracle),
        "residual_minus_geometry_f1": bootstrap_difference(
            selected_residual,
            selected_geometry,
            "f1",
            args.bootstrap,
            args.seed,
        ),
        "residual_minus_geometry_penetration": bootstrap_difference(
            selected_residual,
            selected_geometry,
            "penetration",
            args.bootstrap,
            args.seed + 1,
        ),
        "residual_minus_base_f1": bootstrap_difference(
            selected_residual,
            selected_base,
            "f1",
            args.bootstrap,
            args.seed + 2,
        ),
    }
    result["checks"] = {
        "residual_f1_gt_geometry": (
            result["residual_minus_geometry_f1"]["mean"] > 0.0
        ),
        "residual_f1_ci_positive": (
            result["residual_minus_geometry_f1"]["ci95"][0] > 0.0
        ),
        "residual_penetration_not_worse": (
            result["residual_minus_geometry_penetration"]["mean"] <= 0.0
        ),
        "residual_f1_gt_base": (
            result["residual_minus_base_f1"]["mean"] > 0.0
        ),
    }
    result["overall"] = all(result["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
