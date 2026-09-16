#!/usr/bin/env python3
"""Evaluate MaMi candidates with the existing hand-contact metric v3.

This is an external-first, CPU-only preflight. It reuses MaMi's saved
hand-mesh clearance and contact counts instead of introducing another
clearance threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HAND_KEYS = ("left", "right")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--candidate_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def load_candidate_contact(candidate_dir, sequence_names):
    path = (
        Path(candidate_dir)
        / "evaluation_metrics_json"
        / "chois_wo_guidance"
        / "hand_contact_metrics_v3.jsonl"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sequence = row["sequence"]
        for source_sequence in sequence_names:
            if sequence.startswith(source_sequence + "_"):
                break
        else:
            raise ValueError(f"Could not map sequence: {sequence}")
        records[source_sequence] = {
            hand: hand_metrics(row, hand) for hand in HAND_KEYS
        }
    return records


def hand_metrics(row, hand):
    prefix = hand
    precision = row.get(f"{prefix}_precision")
    recall = row.get(f"{prefix}_recall")
    f1 = row.get(f"{prefix}_f1")
    return {
        "precision": None if precision is None else float(precision),
        "recall": None if recall is None else float(recall),
        "f1": None if f1 is None else float(f1),
        "tp": int(row.get(f"{prefix}_tp", 0)),
        "fp": int(row.get(f"{prefix}_fp", 0)),
        "fn": int(row.get(f"{prefix}_fn", 0)),
        "predicted_contact_frames": int(
            row.get(f"{prefix}_pred_contact_frame_count", 0)
        ),
        "gt_contact_frames": int(
            row.get(f"{prefix}_gt_contact_frame_count", 0)
        ),
        "pred_hand_mesh_mm": row.get(f"pred_hand_mesh_{prefix}_mm"),
        "pred_proxy_mm": row.get(f"pred_proxy_{prefix}_mm"),
        "gt_hand_mesh_mm": row.get(f"gt_hand_mesh_{prefix}_mm"),
        "gt_proxy_mm": row.get(f"gt_proxy_{prefix}_mm"),
    }


def pool_counts(rows):
    result = {}
    for hand in HAND_KEYS:
        tp = sum(row[hand]["tp"] for row in rows)
        fp = sum(row[hand]["fp"] for row in rows)
        fn = sum(row[hand]["fn"] for row in rows)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * tp / max(2 * tp + fp + fn, 1)
        result[hand] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return result


def mean_valid(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def summarize_arm(rows, arm_name):
    pooled = pool_counts(rows)
    summary = {
        "arm": arm_name,
        "sequence_count": len(rows),
        "pooled": pooled,
    }
    for hand in HAND_KEYS:
        summary[f"{hand}_macro_f1"] = mean_valid([
            row[hand]["f1"] for row in rows
        ])
        summary[f"{hand}_mean_pred_hand_mesh_mm"] = mean_valid([
            row[hand]["pred_hand_mesh_mm"] for row in rows
        ])
        summary[f"{hand}_mean_pred_proxy_mm"] = mean_valid([
            row[hand]["pred_proxy_mm"] for row in rows
        ])
        summary[f"{hand}_mean_gt_hand_mesh_mm"] = mean_valid([
            row[hand]["gt_hand_mesh_mm"] for row in rows
        ])
        summary[f"{hand}_mean_gt_proxy_mm"] = mean_valid([
            row[hand]["gt_proxy_mm"] for row in rows
        ])
    return summary


def bootstrap_interval(values, samples, seed):
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
    selected_index = {
        row["sequence_name"]: int(row["selected_candidate_index"])
        for row in selection["selected"]
    }
    candidate_dirs = sorted(
        Path(args.candidate_root).glob("candidate_seed_*")
    )
    if not candidate_dirs:
        raise FileNotFoundError("No candidate_seed_* directories found")
    all_metrics = [
        load_candidate_contact(candidate_dir, set(selected_index))
        for candidate_dir in candidate_dirs
    ]
    sequences = sorted(
        set(selected_index).intersection(all_metrics[0])
    )
    if not sequences:
        raise ValueError("No common sequence names")

    baseline_rows = [all_metrics[0][sequence] for sequence in sequences]
    selected_rows = [
        all_metrics[selected_index[sequence]][sequence]
        for sequence in sequences
    ]
    random_rows = []
    oracle_rows = []
    for sequence in sequences:
        rows = [metrics[sequence] for metrics in all_metrics]
        random_rows.append({
            hand: {
                key: mean_valid([row[hand][key] for row in rows])
                for key in rows[0][hand]
            }
            for hand in HAND_KEYS
        })
        oracle_rows.append({
            hand: max(
                (row[hand] for row in rows),
                key=lambda item: (
                    -1.0 if item["f1"] is None else item["f1"]
                ),
            )
            for hand in HAND_KEYS
        })

    selected_random_diff = {}
    selected_baseline_diff = {}
    for hand in HAND_KEYS:
        selected_random_diff[hand] = np.asarray([
            selected[hand]["f1"] - random[hand]["f1"]
            for selected, random in zip(selected_rows, random_rows)
        ])
        selected_baseline_diff[hand] = np.asarray([
            selected[hand]["f1"] - baseline[hand]["f1"]
            for selected, baseline in zip(selected_rows, baseline_rows)
        ])

    summary = {
        "baseline": summarize_arm(baseline_rows, "baseline"),
        "random": summarize_arm(random_rows, "random"),
        "selected": summarize_arm(selected_rows, "selected"),
        "oracle": summarize_arm(oracle_rows, "oracle"),
        "selected_minus_baseline": {},
        "selected_minus_random": {},
    }
    for hand_index, hand in enumerate(HAND_KEYS):
        summary["selected_minus_baseline"][hand] = {
            "mean": float(selected_baseline_diff[hand].mean()),
            "ci95": bootstrap_interval(
                selected_baseline_diff[hand],
                args.bootstrap_samples,
                args.seed + hand_index,
            ),
        }
        summary["selected_minus_random"][hand] = {
            "mean": float(selected_random_diff[hand].mean()),
            "ci95": bootstrap_interval(
                selected_random_diff[hand],
                args.bootstrap_samples,
                args.seed + 10 + hand_index,
            ),
        }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "selection": str(Path(args.selection)),
                "candidate_dirs": [
                    str(path) for path in candidate_dirs
                ],
                "sequences": sequences,
                "summary": summary,
                "baseline_rows": baseline_rows,
                "selected_rows": selected_rows,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
