#!/usr/bin/env python3
"""Summarize Stage 1 learned-residual training and scorer A/B results."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


ARMS = ("geom_only", "learned_state", "hybrid_event")
SEEDS = (21, 22, 23)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def load_events(root, seed, arm):
    path = (
        root
        / "learned_residual_scorer_ab_20260916"
        / "formal"
        / f"seed_{seed}"
        / arm
        / "result.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def event_key(event):
    return event["sequence"], event["hand"]


def selected_penetration(event):
    names = [
        "base",
        "inward",
        "outward",
        "zero_palm",
        *[f"random_{index}" for index in range(8)],
    ]
    return float(
        event["candidate_mean_penetration_distance"][
            names.index(event["selected_name"])
        ]
    )


def paired_bootstrap(values, samples=5000, seed=0):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        values,
        size=(samples, len(values)),
        replace=True,
    ).mean(axis=1)
    return float(draws.mean()), [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def summarize_training(root):
    pattern = (
        root
        / "learned_residual_20260916"
        / "formal"
        / "seed_*"
        / "B_learned_residual"
        / "metrics.json"
    )
    rows = []
    for path in sorted(glob.glob(str(pattern))):
        seed = int(
            next(
                part.removeprefix("seed_")
                for part in Path(path).parts
                if part.startswith("seed_")
            )
        )
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        row = {
            "seed": seed,
            "best_epoch": data["best_epoch"],
            "residual_mean_abs": data["residual_stats"]["mean_abs"],
            "residual_max_abs": data["residual_stats"]["max_abs"],
        }
        for mode in ("free_running", "analytic_free_running"):
            for horizon in ("h1", "h8"):
                for metric in (
                    "contact_f1",
                    "contact_bce",
                    "palm_l1",
                ):
                    row[f"{mode}_{horizon}_{metric}"] = data[mode][
                        horizon
                    ][metric]
        rows.append(row)
    return rows


def summarize_scorer(root):
    scorer = {
        seed: {
            arm: load_events(root, seed, arm)
            for arm in ARMS
        }
        for seed in SEEDS
    }
    summaries = {}
    for seed in SEEDS:
        summaries[seed] = {}
        for arm in ARMS:
            data = scorer[seed][arm]
            summaries[seed][arm] = {
                "selected_mean_f1": data["selected_mean_f1"],
                "base_mean_f1": data["base_mean_f1"],
                "random_mean_f1": data["random_mean_f1"],
                "oracle_mean_f1": data["oracle_mean_f1"],
                "selected_minus_base": data["selected_minus_base"],
                "selected_minus_random": data["selected_minus_random"],
                "selected_penetration_mean": float(
                    np.mean(
                        [
                            selected_penetration(event)
                            for event in data["events"]
                        ]
                    )
                ),
            }
    return scorer, summaries


def paired_arm_comparison(scorer, right):
    f1_deltas = []
    penetration_deltas = []
    wins = ties = losses = 0
    for seed in SEEDS:
        baseline = {
            event_key(event): event
            for event in scorer[seed]["geom_only"]["events"]
        }
        candidate = {
            event_key(event): event
            for event in scorer[seed][right]["events"]
        }
        for key in sorted(set(baseline) & set(candidate)):
            left_event = baseline[key]
            right_event = candidate[key]
            delta = (
                right_event["selected_f1"]
                - left_event["selected_f1"]
            )
            f1_deltas.append(delta)
            penetration_deltas.append(
                selected_penetration(right_event)
                - selected_penetration(left_event)
            )
            if delta > 1e-9:
                wins += 1
            elif delta < -1e-9:
                losses += 1
            else:
                ties += 1
    f1_mean, f1_ci = paired_bootstrap(
        f1_deltas,
        seed=100 + len(right),
    )
    penetration_mean, penetration_ci = paired_bootstrap(
        penetration_deltas,
        seed=200 + len(right),
    )
    return {
        "comparison": f"{right} - geom_only",
        "mean_f1_delta": f1_mean,
        "f1_ci95": f1_ci,
        "mean_penetration_delta": penetration_mean,
        "penetration_ci95": penetration_ci,
        "win_tie_loss": [wins, ties, losses],
    }


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    training = summarize_training(root)
    scorer, scorer_summary = summarize_scorer(root)
    result = {
        "training": training,
        "training_means": {
            mode: {
                horizon: {
                    metric: float(
                        np.mean(
                            [
                                row[
                                    f"{mode}_{horizon}_{metric}"
                                ]
                                for row in training
                            ]
                        )
                    )
                    for metric in (
                        "contact_f1",
                        "contact_bce",
                        "palm_l1",
                    )
                }
                for horizon in ("h1", "h8")
            }
            for mode in ("free_running", "analytic_free_running")
        },
        "scorer": scorer_summary,
        "paired_comparison": [
            paired_arm_comparison(scorer, arm)
            for arm in ("learned_state", "hybrid_event")
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
