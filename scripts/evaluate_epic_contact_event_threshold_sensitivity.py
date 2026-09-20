#!/usr/bin/env python3
"""Evaluate strict-contact event models across threshold sensitivity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_epic_contact_event_baselines import (
    TARGETS,
    build_samples,
    copy_scores,
    distance_linear_scores,
    evaluate_scores,
    learned_scores,
    load_jsonl_gzip,
    split_arrays,
    standardize,
    train_logistic,
    tune_thresholds,
)


DEFAULT_THRESHOLDS = (0.0005, 0.001, 0.0015, 0.002, 0.003)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
    )
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


def object_names_from_frames(frames):
    return sorted({
        row[hand]["object_name"]
        for row in frames
        for hand in ("left", "right")
        if row[hand]["valid"] and row[hand]["object_name"] is not None
    })


def best_trivial_auprc(evaluation, target):
    return max(
        evaluation["copy_current"][target]["auprc"],
        evaluation["distance_linear"][target]["auprc"],
    )


def run_threshold(
    frames,
    object_names,
    threshold_m,
    epochs,
    learning_rate,
    l2,
    seed,
):
    samples = build_samples(
        frames,
        object_names,
        threshold_m,
    )
    split = split_arrays(samples)
    train_x, dev_x, test_x, mean, std = standardize(
        split["train"]["x"],
        split["dev"]["x"],
        split["test"]["x"],
    )
    fitted = {}
    for offset, target in enumerate(TARGETS):
        fitted[target] = train_logistic(
            train_x,
            split["train"]["y"][target],
            dev_x,
            split["dev"]["y"][target],
            epochs,
            learning_rate,
            l2,
            seed + offset,
        )
        fitted[target]["x_mean"] = mean
        fitted[target]["x_std"] = std

    scores = {
        "copy_current": {
            "dev": copy_scores(split["dev"]["rows"]),
            "test": copy_scores(split["test"]["rows"]),
        },
        "distance_linear": {
            "dev": distance_linear_scores(split["dev"]["rows"]),
            "test": distance_linear_scores(split["test"]["rows"]),
        },
        "learned_logistic": {
            "dev": learned_scores(split["dev"], fitted),
            "test": learned_scores(split["test"], fitted),
        },
    }
    thresholds = {
        model: tune_thresholds(split["dev"]["y"], values["dev"])
        for model, values in scores.items()
    }
    evaluation = {
        split_name: {
            model: evaluate_scores(
                split[split_name],
                values[split_name],
                thresholds[model],
            )
            for model, values in scores.items()
        }
        for split_name in ("dev", "test")
    }
    counts = {
        split_name: {
            "samples": len(values["rows"]),
            "participants": len({
                row["participant_id"] for row in values["rows"]
            }),
            "labels": {
                target: int(values["y"][target].sum())
                for target in TARGETS
            },
        }
        for split_name, values in split.items()
    }
    return {
        "threshold_m": float(threshold_m),
        "counts": counts,
        "thresholds": thresholds,
        "evaluation": evaluation,
    }


def sensitivity_gate(curve):
    rows = {row["threshold_m"]: row for row in curve}
    primary = rows.get(0.001)
    checks = {
        "primary_threshold_present": primary is not None,
    }
    for threshold_m in (0.0005, 0.0015):
        row = rows.get(threshold_m)
        present = row is not None
        checks[f"{threshold_m:.4f}_present"] = present
        if not present:
            checks[f"{threshold_m:.4f}_onset_beats_trivial"] = False
            checks[f"{threshold_m:.4f}_release_beats_trivial"] = False
            continue
        evaluation = row["evaluation"]["test"]
        checks[f"{threshold_m:.4f}_onset_beats_trivial"] = (
            evaluation["learned_logistic"]["onset"]["auprc"]
            > best_trivial_auprc(evaluation, "onset")
        )
        checks[f"{threshold_m:.4f}_release_beats_trivial"] = (
            evaluation["learned_logistic"]["release"]["auprc"]
            > best_trivial_auprc(evaluation, "release")
        )
    checks["overall"] = all(checks.values())
    return checks


def main():
    args = parse_args()
    frames = load_jsonl_gzip(args.frames)
    object_names = object_names_from_frames(frames)
    curve = [
        run_threshold(
            frames,
            object_names,
            threshold_m,
            args.epochs,
            args.learning_rate,
            args.l2,
            args.seed,
        )
        for threshold_m in args.thresholds
    ]
    curve.sort(key=lambda row: row["threshold_m"])
    result = {
        "thresholds_m": sorted(float(value) for value in args.thresholds),
        "curve": curve,
        "gate": sensitivity_gate(curve),
        "versions": {
            "numpy": np.__version__,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
