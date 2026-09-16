#!/usr/bin/env python3
"""Summarize Stage 2 A/A-plus/B-shuffle/B runs across seeds."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


ARMS = ("A", "A_plus", "B_shuffle", "B")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(11, 12, 13),
    )
    parser.add_argument("--output", default="")
    return parser.parse_args()


def read_arm(root, seed, arm):
    path = Path(root) / f"seed_{seed}" / arm / "metrics.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data["contact_events"]
    onset = [
        value
        for key, value in events.items()
        if key.endswith("onset_auc") and value is not None
    ]
    release = [
        value
        for key, value in events.items()
        if key.endswith("release_auc") and value is not None
    ]
    free = data["free_running"]
    contact_f1 = [
        row["contact_f1"]
        for row in free.values()
        if row.get("contact_f1") is not None
    ]
    return {
        "onset_auc": statistics.mean(onset),
        "release_auc": statistics.mean(release),
        "contact_f1": statistics.mean(contact_f1),
        "best_epoch": int(data["best_epoch"]),
    }


def delta(left, right, key):
    return left[key] - right[key]


def main():
    args = parse_args()
    seed_rows = {}
    for seed in args.seeds:
        row = {}
        for arm in ARMS:
            metrics = read_arm(args.root, seed, arm)
            if metrics is not None:
                row[arm] = metrics
        seed_rows[seed] = row

    complete_rows = {
        seed: row
        for seed, row in seed_rows.items()
        if all(arm in row for arm in ARMS)
    }
    summary = {
        "root": str(args.root),
        "seeds": list(args.seeds),
        "complete_seeds": sorted(complete_rows),
        "per_seed": seed_rows,
        "mean": {},
        "comparisons": {},
        "gate_by_seed": {},
    }
    for arm in ARMS:
        available = [
            row[arm]
            for row in complete_rows.values()
        ]
        if not available:
            continue
        summary["mean"][arm] = {
            key: statistics.mean(item[key] for item in available)
            for key in ("onset_auc", "release_auc", "contact_f1")
        }

    for left, right in (
        ("B", "A"),
        ("B", "A_plus"),
        ("B", "B_shuffle"),
    ):
        per_seed = {}
        for seed, row in complete_rows.items():
            per_seed[str(seed)] = {
                key: delta(row[left], row[right], key)
                for key in ("onset_auc", "release_auc", "contact_f1")
            }
        summary["comparisons"][f"{left}-{right}"] = per_seed

    for seed, row in complete_rows.items():
        b = row["B"]
        gate = (
            b["onset_auc"] - row["A"]["onset_auc"] >= 0.02
            and b["onset_auc"] > row["A_plus"]["onset_auc"]
            and b["onset_auc"] > row["B_shuffle"]["onset_auc"]
        )
        summary["gate_by_seed"][str(seed)] = bool(gate)
    summary["gate_all_complete_seeds"] = bool(
        summary["gate_by_seed"]
        and all(summary["gate_by_seed"].values())
    )

    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
