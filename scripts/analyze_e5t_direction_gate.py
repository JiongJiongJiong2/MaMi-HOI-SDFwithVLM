#!/usr/bin/env python3
"""Evaluate a train-thresholded direction gate between E5 and E5-T."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


FEATURE = "near_outward_fraction"
SELECTION_PENALTY = 0.05
MAX_TRAIN_SELECTION_FRACTION = 0.08
BOOTSTRAP_SEED = 20260920


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e5-result", type=Path, required=True)
    parser.add_argument("--e5t-result", type=Path, required=True)
    parser.add_argument("--direction-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_key(row):
    return (
        row["chunk_id"],
        tuple(int(frame) for frame in row["frames"]),
    )


def load_rows(e5_result, e5t_result, direction_summary):
    e5 = json.loads(e5_result.read_text(encoding="utf-8"))
    e5t = json.loads(e5t_result.read_text(encoding="utf-8"))
    direction = json.loads(
        direction_summary.read_text(encoding="utf-8")
    )
    e5_windows = {window_key(row): row for row in e5["windows"]}
    e5t_windows = {window_key(row): row for row in e5t["windows"]}
    direction_windows = {
        window_key(row): row for row in direction["windows"]
    }
    if set(e5_windows) != set(e5t_windows):
        raise ValueError("E5 and E5-T window sets differ")
    if set(e5_windows) != set(direction_windows):
        raise ValueError("direction and E5 window sets differ")

    rows = []
    for key in e5_windows:
        base = e5_windows[key]
        candidate = e5t_windows[key]
        metrics = direction_windows[key]
        rows.append({
            "chunk_id": key[0],
            "sequence": base["sequence"],
            "split": base["split"],
            "object_name": base["object_name"],
            "frames": list(key[1]),
            "e5_contact_pass": bool(
                base["optimized_contact_gate"]["passed"]
            ),
            "e5t_contact_pass": bool(
                candidate["optimized_contact_gate"]["passed"]
            ),
            "e5_combined_pass": bool(
                base["optimized_combined_pass"]
            ),
            "e5t_combined_pass": bool(
                candidate["optimized_combined_pass"]
            ),
            FEATURE: metrics[FEATURE],
        })
    return rows


def threshold_candidates(train, feature):
    values = sorted({
        row[feature]
        for row in train
        if row["e5_contact_pass"]
        and row[feature] is not None
        and np.isfinite(row[feature])
    })
    return [
        math.inf,
        *[
            (lower + upper) / 2.0
            for lower, upper in zip(values, values[1:])
        ],
    ]


def choose_e5(row, feature, threshold):
    value = row[feature]
    if not row["e5_contact_pass"]:
        return False
    if value is None or not np.isfinite(value):
        return False
    return bool(value > threshold)


def selection_stats(rows, feature, threshold):
    decisions = [
        choose_e5(row, feature, threshold) for row in rows
    ]
    return {
        "window_count": len(rows),
        "selected_e5_count": int(sum(decisions)),
        "contact_pass_count": int(sum(
            row["e5_contact_pass"] if select_e5
            else row["e5t_contact_pass"]
            for row, select_e5 in zip(rows, decisions)
        )),
        "combined_pass_count": int(sum(
            row["e5_combined_pass"] if select_e5
            else row["e5t_combined_pass"]
            for row, select_e5 in zip(rows, decisions)
        )),
    }


def fit_threshold(train, feature):
    maximum = int(
        MAX_TRAIN_SELECTION_FRACTION * len(train)
    )
    scored = []
    for threshold in threshold_candidates(train, feature):
        stats = selection_stats(train, feature, threshold)
        if stats["selected_e5_count"] > maximum:
            continue
        objective = (
            stats["combined_pass_count"]
            - SELECTION_PENALTY * stats["selected_e5_count"]
        )
        scored.append((
            objective,
            threshold,
            stats,
        ))
    if not scored:
        raise ValueError("no threshold satisfies the selection budget")
    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )
    objective, threshold, stats = scored[0]
    return {
        "threshold": threshold,
        "objective": objective,
        "train_stats": stats,
        "maximum_train_selection_count": maximum,
    }


def paired_counts(rows, feature, threshold):
    selected = []
    baseline = []
    groups = []
    for row in rows:
        select_e5 = choose_e5(row, feature, threshold)
        selected.append(
            row["e5_combined_pass"] if select_e5
            else row["e5t_combined_pass"]
        )
        baseline.append(row["e5t_combined_pass"])
        groups.append(row["sequence"])
    return np.asarray(selected), np.asarray(baseline), np.asarray(groups)


def cluster_bootstrap_count_difference(
    selected,
    baseline,
    groups,
    samples,
    seed,
):
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    indices = [
        np.flatnonzero(inverse == index)
        for index in range(len(unique_groups))
    ]
    observed = int(selected.sum() - baseline.sum())
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.int64)
    for draw in range(samples):
        sample = rng.integers(
            0,
            len(unique_groups),
            size=len(unique_groups),
        )
        chosen = np.concatenate(
            [indices[index] for index in sample]
        )
        draws[draw] = int(
            selected[chosen].sum() - baseline[chosen].sum()
        )
    return {
        "mean": observed,
        "ci95": [
            int(np.quantile(draws, 0.025)),
            int(np.quantile(draws, 0.975)),
        ],
    }


def evaluate_split(rows, feature, threshold, samples, seed):
    stats = selection_stats(rows, feature, threshold)
    selected, baseline, groups = paired_counts(
        rows,
        feature,
        threshold,
    )
    return {
        **stats,
        "e5_contact_pass_count": int(sum(
            row["e5_contact_pass"] for row in rows
        )),
        "e5t_contact_pass_count": int(sum(
            row["e5t_contact_pass"] for row in rows
        )),
        "e5_combined_pass_count": int(sum(
            row["e5_combined_pass"] for row in rows
        )),
        "e5t_combined_pass_count": int(sum(
            row["e5t_combined_pass"] for row in rows
        )),
        "oracle_combined_pass_count": int(sum(
            row["e5_combined_pass"] or row["e5t_combined_pass"]
            for row in rows
        )),
        "selected_minus_e5t_combined": (
            cluster_bootstrap_count_difference(
                selected,
                baseline,
                groups,
                samples,
                seed,
            )
        ),
    }


def leave_one_object_out(train, feature):
    folds = []
    for object_name in sorted({
        row["object_name"] for row in train
    }):
        fit_rows = [
            row for row in train
            if row["object_name"] != object_name
        ]
        test_rows = [
            row for row in train
            if row["object_name"] == object_name
        ]
        fitted = fit_threshold(fit_rows, feature)
        folds.append({
            "object_name": object_name,
            "threshold": fitted["threshold"],
            "stats": selection_stats(
                test_rows,
                feature,
                fitted["threshold"],
            ),
        })
    return {
        "fold_count": len(folds),
        "selected_e5_count": sum(
            fold["stats"]["selected_e5_count"] for fold in folds
        ),
        "contact_pass_count": sum(
            fold["stats"]["contact_pass_count"] for fold in folds
        ),
        "combined_pass_count": sum(
            fold["stats"]["combined_pass_count"] for fold in folds
        ),
        "window_count": sum(
            fold["stats"]["window_count"] for fold in folds
        ),
        "folds": folds,
    }


def main():
    args = parse_args()
    rows = load_rows(
        args.e5_result,
        args.e5t_result,
        args.direction_summary,
    )
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    if not train or not dev:
        raise ValueError("both train and dev windows are required")

    fitted = fit_threshold(train, FEATURE)
    threshold = fitted["threshold"]
    train_eval = evaluate_split(
        train,
        FEATURE,
        threshold,
        args.bootstrap_samples,
        BOOTSTRAP_SEED,
    )
    dev_eval = evaluate_split(
        dev,
        FEATURE,
        threshold,
        args.bootstrap_samples,
        BOOTSTRAP_SEED + 1,
    )
    leave_one_out = leave_one_object_out(train, FEATURE)
    gate = {
        "train_combined_gain_over_e5t": (
            train_eval["combined_pass_count"]
            > train_eval["e5t_combined_pass_count"]
        ),
        "dev_combined_gain_over_e5t": (
            dev_eval["combined_pass_count"]
            > dev_eval["e5t_combined_pass_count"]
        ),
        "dev_contact_not_below_e5t": (
            dev_eval["contact_pass_count"]
            >= dev_eval["e5t_contact_pass_count"]
        ),
        "dev_combined_ci_lower_positive": (
            dev_eval["selected_minus_e5t_combined"]["ci95"][0] > 0
        ),
    }
    gate["pass"] = all(gate.values())
    result = {
        "method": (
            "use E5 contact-pass windows; select E5 when the train-fitted "
            "near outward-vertex fraction exceeds a threshold, otherwise "
            "select E5-T"
        ),
        "feature": FEATURE,
        "selection_penalty": SELECTION_PENALTY,
        "maximum_train_selection_fraction": (
            MAX_TRAIN_SELECTION_FRACTION
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "inputs": {
            "e5_result": str(args.e5_result),
            "e5_result_sha256": sha256_file(args.e5_result),
            "e5t_result": str(args.e5t_result),
            "e5t_result_sha256": sha256_file(args.e5t_result),
            "direction_summary": str(args.direction_summary),
            "direction_summary_sha256": sha256_file(
                args.direction_summary
            ),
        },
        "fitted": fitted,
        "train": train_eval,
        "dev": dev_eval,
        "leave_one_object_out_train": leave_one_out,
        "gate": gate,
        "decision": "GO" if gate["pass"] else "NO-GO",
        "testing_note": (
            "the dev split was inspected during the direction pilot; this "
            "is a locked-threshold confirmation, not a pristine final test"
        ),
        "test_split": "untouched",
        "selected_windows": [
            {
                **row,
                "select_e5": choose_e5(row, FEATURE, threshold),
            }
            for row in rows
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "decision": result["decision"],
        "threshold": threshold,
        "train": train_eval,
        "dev": dev_eval,
        "leave_one_object_out_train": leave_one_out,
        "gate": gate,
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
