#!/usr/bin/env python3
"""Evaluate selected MaMi candidates using the official per-sequence metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = {
    "contact_f1": ("mean_contact_f1_score", "higher"),
    "contact_precision": ("mean_contact_precision", "higher"),
    "contact_recall": ("mean_contact_recall", "higher"),
    "hand_jpe": ("mean_hand_jpe", "lower"),
    "mpjpe": ("mean_mpjpe", "lower"),
    "hand_penetration": ("mean_hand_penetration_score", "lower"),
    "object_translation_error": ("mean_obj_com_pos_err", "lower"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--candidate_root", required=True)
    parser.add_argument("--candidate_prefix", default="candidate_seed_")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def load_candidate_metrics(candidate_dir, sequence_names):
    root = Path(candidate_dir) / "evaluation_metrics_json"
    files = sorted(root.rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No evaluation JSON under {root}")
    sequence_to_metrics = {}
    for path in files:
        matched = [
            sequence
            for sequence in sequence_names
            if path.name.startswith(sequence + "_")
        ]
        if not matched:
            continue
        if not matched:
            continue
        if len(matched) > 1:
            raise ValueError(f"Could not map metric file: {path}")
        sequence_to_metrics[matched[0]] = json.loads(
            path.read_text(encoding="utf-8")
        )
    missing = set(sequence_names).difference(sequence_to_metrics)
    if missing:
        raise FileNotFoundError(
            f"Missing metrics for sequences: {sorted(missing)}"
        )
    return sequence_to_metrics


def bootstrap_mean_interval(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def main():
    args = parse_args()
    selection = json.loads(
        Path(args.selection).read_text(encoding="utf-8")
    )
    selected_by_sequence = {
        row["sequence_name"]: int(row["selected_candidate_index"])
        for row in selection["selected"]
    }
    candidate_dirs = sorted(
        Path(args.candidate_root).glob(f"{args.candidate_prefix}*")
    )
    if not candidate_dirs:
        raise FileNotFoundError(
            f"No candidates matching {args.candidate_prefix}*"
        )
    candidate_metrics = [
        load_candidate_metrics(path, set(selected_by_sequence))
        for path in candidate_dirs
    ]
    sequences = sorted(
        set(candidate_metrics[0]).intersection(selected_by_sequence)
    )
    rows = []
    for sequence in sequences:
        selected_index = selected_by_sequence[sequence]
        baseline = candidate_metrics[0][sequence]
        selected = candidate_metrics[selected_index][sequence]
        row = {"sequence_name": sequence, "selected_index": selected_index}
        for name, (metric_key, direction) in METRICS.items():
            candidates = [
                metrics[sequence][metric_key]
                for metrics in candidate_metrics
            ]
            row[name] = {
                "baseline": float(baseline[metric_key]),
                "selected": float(selected[metric_key]),
                "random_mean": float(np.mean(candidates)),
                "oracle": (
                    float(np.max(candidates))
                    if direction == "higher"
                    else float(np.min(candidates))
                ),
            }
        rows.append(row)

    summary = {}
    for metric_index, (name, (_, direction)) in enumerate(METRICS.items()):
        baseline = np.asarray([row[name]["baseline"] for row in rows])
        selected = np.asarray([row[name]["selected"] for row in rows])
        random_mean = np.asarray([row[name]["random_mean"] for row in rows])
        sign = 1.0 if direction == "higher" else -1.0
        selected_baseline_diff = sign * (selected - baseline)
        selected_random_diff = sign * (selected - random_mean)
        summary[name] = {
            "direction": direction,
            "baseline_mean": float(baseline.mean()),
            "selected_mean": float(selected.mean()),
            "random_mean": float(random_mean.mean()),
            "oracle_mean": float(np.mean([
                row[name]["oracle"] for row in rows
            ])),
            "selected_minus_baseline_improvement": float(
                selected_baseline_diff.mean()
            ),
            "selected_minus_baseline_ci95": bootstrap_mean_interval(
                selected_baseline_diff,
                args.bootstrap_samples,
                args.seed + metric_index,
            ),
            "selected_minus_random_improvement": float(
                selected_random_diff.mean()
            ),
            "selected_minus_random_ci95": bootstrap_mean_interval(
                selected_random_diff,
                args.bootstrap_samples,
                args.seed + 100 + metric_index,
            ),
            "selected_improved_count": int(
                (selected_baseline_diff > 0).sum()
            ),
            "selected_equal_count": int(
                (selected_baseline_diff == 0).sum()
            ),
            "selected_worse_count": int(
                (selected_baseline_diff < 0).sum()
            ),
        }

    output = {
        "selection": str(Path(args.selection)),
        "candidate_dirs": [str(path) for path in candidate_dirs],
        "summary": summary,
        "sequences": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
