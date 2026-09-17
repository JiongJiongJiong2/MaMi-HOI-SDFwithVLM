#!/usr/bin/env python3
"""Aggregate canonical analytic contact baseline cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


COHORTS = ("10_30", "31_50", "51_70")
CANDIDATE_NAMES = (
    "base",
    "inward",
    "outward",
    "zero_palm",
    *[f"random_{index}" for index in range(8)],
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    return parser.parse_args()


def load_events(root, cohort):
    path = root / cohort / "result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data["scorer_mode"] != "geom_only":
        raise ValueError(f"expected geom_only result: {path}")
    if data["rollout_mode"] != "analytic":
        raise ValueError(f"expected analytic rollout: {path}")
    if data["events"] and "candidate_episode_metrics" not in data["events"][0]:
        raise ValueError(
            "episode metrics are missing; rerun with the updated scorer"
        )
    return data


def selected_penetration(event):
    return float(
        event["candidate_mean_penetration_distance"][
            CANDIDATE_NAMES.index(event["selected_name"])
        ]
    )


def episode_records(event, arm):
    if arm == "base":
        return [event["candidate_episode_metrics"][0]]
    if arm == "selected":
        return [
            event["candidate_episode_metrics"][event["selected_index"]]
        ]
    if arm == "random":
        return [
            metric
            for name, metric in zip(
                event["candidate_names"],
                event["candidate_episode_metrics"],
            )
            if name.startswith("random_")
        ]
    raise ValueError(f"unsupported arm: {arm}")


def mean_defined(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize_episode_arm(events, arm):
    records = [
        record
        for event in events
        for record in episode_records(event, arm)
    ]
    return {
        "record_count": len(records),
        "stable_contact_success_rate": mean_defined([
            record.get("stable_contact_success")
            for record in records
        ]),
        "stable_contact_false_positive_rate": mean_defined([
            record.get("stable_contact_false_positive")
            for record in records
        ]),
        "contact_precision": mean_defined([
            record.get("contact_precision")
            for record in records
        ]),
        "contact_recall": mean_defined([
            record.get("contact_recall")
            for record in records
        ]),
        "contact_f1": mean_defined([
            record.get("contact_f1")
            for record in records
        ]),
        "dropout_count": mean_defined([
            record.get("dropout_count")
            for record in records
        ]),
        "onset_delay_frames": mean_defined([
            record.get("onset_delay_frames")
            for record in records
        ]),
        "early_contact_frames": mean_defined([
            record.get("early_contact_frames")
            for record in records
        ]),
        "late_contact_frames": mean_defined([
            record.get("late_contact_frames")
            for record in records
        ]),
        "release_delay_frames": mean_defined([
            record.get("release_delay_frames")
            for record in records
        ]),
        "predicted_contact_episode_count": mean_defined([
            record.get("predicted_contact_episode_count")
            for record in records
        ]),
        "longest_predicted_episode_frames": mean_defined([
            record.get("longest_predicted_episode_frames")
            for record in records
        ]),
    }


def bootstrap(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values,
        size=(samples, len(values)),
        replace=True,
    ).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def summarize_cohort(data, cohort, samples, seed):
    events = data["events"]
    base_f1 = [float(event["base_f1"]) for event in events]
    selected_f1 = [float(event["selected_f1"]) for event in events]
    random_f1 = [float(event["random_mean_f1"]) for event in events]
    oracle_f1 = [float(event["oracle_f1"]) for event in events]
    base_penetration = [
        float(event["candidate_mean_penetration_distance"][0])
        for event in events
    ]
    selected_penetration_values = [
        selected_penetration(event)
        for event in events
    ]
    random_penetration = [
        float(
            np.mean(
                event["candidate_mean_penetration_distance"][
                    4 : 4 + 8
                ]
            )
        )
        for event in events
    ]
    return {
        "cohort": cohort,
        "sequence_count": data["sequence_count"],
        "event_count": len(events),
        "mean_f1": {
            "base": float(np.mean(base_f1)),
            "random": float(np.mean(random_f1)),
            "selected": float(np.mean(selected_f1)),
            "oracle": float(np.mean(oracle_f1)),
        },
        "selected_minus_base": bootstrap(
            np.asarray(selected_f1) - np.asarray(base_f1),
            samples,
            seed,
        ),
        "selected_minus_random": bootstrap(
            np.asarray(selected_f1) - np.asarray(random_f1),
            samples,
            seed + 1,
        ),
        "oracle_minus_base": bootstrap(
            np.asarray(oracle_f1) - np.asarray(base_f1),
            samples,
            seed + 2,
        ),
        "mean_penetration": {
            "base": float(np.mean(base_penetration)),
            "random": float(np.mean(random_penetration)),
            "selected": float(np.mean(selected_penetration_values)),
        },
        "selected_minus_base_penetration": bootstrap(
            np.asarray(selected_penetration_values)
            - np.asarray(base_penetration),
            samples,
            seed + 3,
        ),
        "selected_minus_random_penetration": bootstrap(
            np.asarray(selected_penetration_values)
            - np.asarray(random_penetration),
            samples,
            seed + 4,
        ),
        "episode_metrics": {
            arm: summarize_episode_arm(events, arm)
            for arm in ("base", "selected", "random")
        },
    }


def summarize_combined(cohorts, samples):
    events = [
        event
        for data in cohorts.values()
        for event in data["events"]
    ]
    base_f1 = np.asarray([event["base_f1"] for event in events])
    selected_f1 = np.asarray([event["selected_f1"] for event in events])
    random_f1 = np.asarray([
        event["random_mean_f1"] for event in events
    ])
    oracle_f1 = np.asarray([event["oracle_f1"] for event in events])
    base_penetration = np.asarray([
        event["candidate_mean_penetration_distance"][0]
        for event in events
    ])
    selected_penetration_values = np.asarray([
        selected_penetration(event)
        for event in events
    ])
    random_penetration = np.asarray([
        np.mean(
            event["candidate_mean_penetration_distance"][4 : 4 + 8]
        )
        for event in events
    ])
    return {
        "sequence_count": len({
            event["sequence"] for event in events
        }),
        "event_count": len(events),
        "mean_f1": {
            "base": float(base_f1.mean()),
            "random": float(random_f1.mean()),
            "selected": float(selected_f1.mean()),
            "oracle": float(oracle_f1.mean()),
        },
        "selected_minus_base": bootstrap(
            selected_f1 - base_f1,
            samples,
            1000,
        ),
        "selected_minus_random": bootstrap(
            selected_f1 - random_f1,
            samples,
            1001,
        ),
        "oracle_minus_base": bootstrap(
            oracle_f1 - base_f1,
            samples,
            1002,
        ),
        "mean_penetration": {
            "base": float(base_penetration.mean()),
            "random": float(random_penetration.mean()),
            "selected": float(selected_penetration_values.mean()),
        },
        "selected_minus_base_penetration": bootstrap(
            selected_penetration_values - base_penetration,
            samples,
            1003,
        ),
        "selected_minus_random_penetration": bootstrap(
            selected_penetration_values - random_penetration,
            samples,
            1004,
        ),
        "episode_metrics": {
            arm: summarize_episode_arm(events, arm)
            for arm in ("base", "selected", "random")
        },
    }


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    data = {
        cohort: load_events(root, cohort)
        for cohort in COHORTS
    }
    result = {
        "root": str(root),
        "protocol": {
            "scorer_mode": "geom_only",
            "rollout_mode": "analytic",
            "penetration_weight": 20.0,
            "horizon": 8,
            "candidate_seed": 1,
        },
        "combined": summarize_combined(
            data,
            args.bootstrap_samples,
        ),
        "cohorts": [
            summarize_cohort(
                data[cohort],
                cohort,
                args.bootstrap_samples,
                100 + index * 10,
            )
            for index, cohort in enumerate(COHORTS)
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["combined"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
