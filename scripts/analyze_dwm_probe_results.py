#!/usr/bin/env python3
"""Aggregate P0 arms and evaluate the frozen decision-utility gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260922)
    return parser.parse_args()


def load_npz(path):
    return {
        key: np.asarray(value)
        for key, value in np.load(path, allow_pickle=False).items()
    }


def keys_from_rows(object_id, reset_id):
    return np.asarray([
        f"{str(object_value)}:{int(reset_value)}"
        for object_value, reset_value in zip(object_id, reset_id)
    ])


def metric_rows(keys, utility, chosen, contact_mode):
    oracle = utility.argmax(axis=1)
    rows = np.arange(utility.shape[0])
    chosen_utility = utility[rows, chosen]
    oracle_utility = utility[rows, oracle]
    chosen_contact = contact_mode[rows, chosen]
    return {
        "keys": keys,
        "top1": (chosen == oracle).astype(np.float64),
        "regret": oracle_utility - chosen_utility,
        "slip": (chosen_contact[:, 1:] == 2).any(axis=1).astype(np.float64),
        "release": (chosen_contact[:, 1:] == 3).any(axis=1).astype(np.float64),
    }


def condition_mask(data, mode, length):
    return (
        (data["probe_mode"] == mode)
        & (data["probe_length"] == int(length))
    )


def parse_condition(text):
    if text == "all":
        return None, None
    if text == "none":
        return "none", 0
    mode, length = text.split(":", 1)
    return mode, int(length)


def rows_for_arm(arm, test, eval_mode, eval_length):
    predictions = arm["predictions"]
    condition = arm["metrics"].get("condition", "all")
    mode, length = parse_condition(condition)
    if mode is None:
        prediction_mask = condition_mask(
            predictions,
            eval_mode,
            eval_length,
        )
        test_mask = condition_mask(test, eval_mode, eval_length)
    else:
        prediction_mask = np.ones(predictions["utility"].shape[0], dtype=bool)
        test_mask = condition_mask(test, mode, length)
    if int(prediction_mask.sum()) != int(test_mask.sum()):
        raise ValueError("prediction count does not match test condition")
    keys = keys_from_rows(
        predictions["object_id"][prediction_mask],
        predictions["reset_id"][prediction_mask],
    )
    expected_keys = keys_from_rows(
        test["object_id"][test_mask],
        test["reset_id"][test_mask],
    )
    if not np.array_equal(keys, expected_keys):
        raise ValueError("prediction row order does not match dataset")
    return metric_rows(
        keys,
        predictions["utility"][prediction_mask].astype(np.float64),
        predictions["chosen"][prediction_mask],
        test["candidate_contact_mode"][test_mask],
    )


def geometry_rows(test):
    mask = condition_mask(test, "fixed", 2)
    action = test["candidate_action"][mask]
    hold = action[:, 0, -1, :3]
    predicted = action[:, :, -1, :3] - hold[:, None, :]
    target = test["target_translation"][mask]
    score = -np.sum(
        np.abs(predicted - target[:, None, :]),
        axis=-1,
    )
    return metric_rows(
        keys_from_rows(test["object_id"][mask], test["reset_id"][mask]),
        test["utility"][mask].astype(np.float64),
        score.argmax(axis=1),
        test["candidate_contact_mode"][mask],
    )


def random_rows(test, seed):
    mask = condition_mask(test, "fixed", 2)
    utility = test["utility"][mask].astype(np.float64)
    rng = np.random.default_rng(seed)
    chosen = rng.integers(
        0,
        utility.shape[1],
        size=utility.shape[0],
    )
    return metric_rows(
        keys_from_rows(test["object_id"][mask], test["reset_id"][mask]),
        utility,
        chosen,
        test["candidate_contact_mode"][mask],
    )


def align_rows(rows):
    key_sets = [set(row["keys"].tolist()) for row in rows]
    keys = sorted(set.intersection(*key_sets))
    if not keys:
        raise ValueError("no aligned rows")
    aligned = []
    for row in rows:
        index = {key: value for value, key in enumerate(row["keys"])}
        aligned.append({
            name: row[name][[index[key] for key in keys]]
            for name in ("top1", "regret", "slip", "release")
        })
    return keys, aligned


def hierarchical_bootstrap_diff(
    left,
    right,
    keys,
    samples,
    seed,
):
    object_ids = np.asarray([key.split(":", 1)[0] for key in keys])
    unique_objects = sorted(set(object_ids.tolist()))
    by_object = {
        object_id: np.flatnonzero(object_ids == object_id)
        for object_id in unique_objects
    }
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        selected_objects = rng.choice(
            unique_objects,
            size=len(unique_objects),
            replace=True,
        )
        selected_rows = []
        for object_id in selected_objects:
            candidates = by_object[object_id]
            selected_rows.append(rng.choice(
                candidates,
                size=candidates.size,
                replace=True,
            ))
        rows = np.concatenate(selected_rows)
        values.append(float(np.mean(left[rows] - right[rows])))
    return {
        "mean": float(np.mean(values)),
        "ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ],
    }


def mean_metric(aligned, name):
    return np.mean([row[name] for row in aligned], axis=0)


def aggregate_rows(rows):
    return {
        "top1": float(np.mean([row["top1"].mean() for row in rows])),
        "regret": float(np.mean([row["regret"].mean() for row in rows])),
        "slip": float(np.mean([row["slip"].mean() for row in rows])),
        "release": float(np.mean([row["release"].mean() for row in rows])),
        "groups": int(np.mean([row["top1"].size for row in rows])),
    }


def oracle_utility_rows(test):
    mask = condition_mask(test, "fixed", 2)
    utility = test["utility"][mask].astype(np.float64)
    return metric_rows(
        keys_from_rows(test["object_id"][mask], test["reset_id"][mask]),
        utility,
        utility.argmax(axis=1),
        test["candidate_contact_mode"][mask],
    )


def compare_rows(
    main_rows,
    baseline_rows,
    bootstrap,
    bootstrap_seed,
):
    if len(baseline_rows) == 1:
        baseline_rows = baseline_rows * len(main_rows)
    if len(main_rows) != len(baseline_rows):
        raise ValueError("main and baseline seed counts differ")
    keys, aligned = align_rows(main_rows + baseline_rows)
    aligned_main = aligned[:len(main_rows)]
    aligned_baseline = aligned[len(main_rows):]
    main_mean = {
        name: mean_metric(aligned_main, name)
        for name in ("top1", "regret", "slip", "release")
    }
    baseline_mean = {
        name: mean_metric(aligned_baseline, name)
        for name in ("top1", "regret", "slip", "release")
    }
    top1_diff = main_mean["top1"] - baseline_mean["top1"]
    regret_diff = baseline_mean["regret"] - main_mean["regret"]
    slip_diff = main_mean["slip"] - baseline_mean["slip"]
    release_diff = main_mean["release"] - baseline_mean["release"]
    seed_top1 = [
        float(np.mean(main["top1"] - baseline["top1"]))
        for main, baseline in zip(aligned_main, aligned_baseline)
    ]
    seed_regret = [
        float(np.mean(baseline["regret"] - main["regret"]))
        for main, baseline in zip(aligned_main, aligned_baseline)
    ]
    return {
        "groups": len(keys),
        "main": {
            "top1": float(main_mean["top1"].mean()),
            "regret": float(main_mean["regret"].mean()),
            "slip": float(main_mean["slip"].mean()),
            "release": float(main_mean["release"].mean()),
        },
        "baseline": {
            "top1": float(baseline_mean["top1"].mean()),
            "regret": float(baseline_mean["regret"].mean()),
            "slip": float(baseline_mean["slip"].mean()),
            "release": float(baseline_mean["release"].mean()),
        },
        "top1_gain_pp": float(100.0 * top1_diff.mean()),
        "regret_reduction_pct": float(
            100.0 * regret_diff.mean()
            / max(float(baseline_mean["regret"].mean()), 1e-12)
        ),
        "slip_diff_pp": float(100.0 * slip_diff.mean()),
        "release_diff_pp": float(100.0 * release_diff.mean()),
        "seed_top1_gain_pp": [float(100.0 * value) for value in seed_top1],
        "seed_regret_reduction_pct": [
            float(
                100.0 * value
                / max(float(baseline["regret"].mean()), 1e-12)
            )
            for value, baseline in zip(seed_regret, aligned_baseline)
        ],
        "top1_bootstrap": hierarchical_bootstrap_diff(
            main_mean["top1"],
            baseline_mean["top1"],
            keys,
            bootstrap,
            bootstrap_seed,
        ),
        "regret_bootstrap": hierarchical_bootstrap_diff(
            baseline_mean["regret"],
            main_mean["regret"],
            keys,
            bootstrap,
            bootstrap_seed + 1,
        ),
        "directions_positive": (
            all(value > 0 for value in seed_top1)
            and all(value > 0 for value in seed_regret)
        ),
        "safety_ok": (
            max(0.0, float(100.0 * slip_diff.mean())) <= 2.0
            and max(0.0, float(100.0 * release_diff.mean())) <= 2.0
        ),
    }


def load_arms(results_dir):
    arms = []
    for path in sorted(results_dir.glob("*/metrics.json")):
        predictions_path = path.parent / "test_predictions.npz"
        if not predictions_path.exists():
            continue
        arms.append({
            "metrics": json.loads(path.read_text(encoding="utf-8")),
            "predictions": load_npz(predictions_path),
        })
    return arms


def select_arms(arms, objective, condition):
    return sorted(
        [
            arm for arm in arms
            if arm["metrics"]["objective"] == objective
            and arm["metrics"].get("condition", "all") == condition
        ],
        key=lambda arm: arm["metrics"]["seed"],
    )


def main():
    args = parse_args()
    test = load_npz(args.dataset_dir / "test.npz")
    arms = load_arms(args.results_dir)
    main_arms = select_arms(arms, "listwise", "all")
    no_probe_arms = select_arms(arms, "listwise", "none")
    if not main_arms or not no_probe_arms:
        raise ValueError("required listwise test predictions are missing")
    main_rows = [
        rows_for_arm(arm, test, "fixed", 2)
        for arm in main_arms
    ]
    no_probe_rows = [
        rows_for_arm(arm, test, "none", 0)
        for arm in no_probe_arms
    ]
    comparisons = {
        "geometry": compare_rows(
            main_rows,
            [geometry_rows(test)],
            args.bootstrap,
            args.bootstrap_seed,
        ),
        "random": compare_rows(
            main_rows,
            [random_rows(test, args.bootstrap_seed)],
            args.bootstrap,
            args.bootstrap_seed,
        ),
        "no_probe": compare_rows(
            main_rows,
            no_probe_rows,
            args.bootstrap,
            args.bootstrap_seed,
        ),
    }
    geometry = comparisons["geometry"]
    no_probe = comparisons["no_probe"]
    checks = {
        "top1_geometry_ge_10pp": geometry["top1_gain_pp"] >= 10.0,
        "top1_no_probe_ge_5pp": no_probe["top1_gain_pp"] >= 5.0,
        "regret_geometry_ge_20pct": (
            geometry["regret_reduction_pct"] >= 20.0
        ),
        "regret_no_probe_ge_10pct": (
            no_probe["regret_reduction_pct"] >= 10.0
        ),
        "top1_ci_positive": (
            geometry["top1_bootstrap"]["ci95"][0] > 0.0
            and no_probe["top1_bootstrap"]["ci95"][0] > 0.0
        ),
        "regret_ci_positive": (
            geometry["regret_bootstrap"]["ci95"][0] > 0.0
            and no_probe["regret_bootstrap"]["ci95"][0] > 0.0
        ),
        "seed_directions_positive": (
            geometry["directions_positive"]
            and no_probe["directions_positive"]
        ),
        "safety_noninferior": (
            geometry["safety_ok"] and no_probe["safety_ok"]
        ),
    }
    result = {
        "main_arms": [
            {
                "seed": arm["metrics"]["seed"],
                "metrics": arm["metrics"].get("test"),
            }
            for arm in main_arms
        ],
        "no_probe_arms": [
            {
                "seed": arm["metrics"]["seed"],
                "metrics": arm["metrics"].get("test"),
            }
            for arm in no_probe_arms
        ],
        "comparisons": comparisons,
        "budget_sensitivity": {},
        "checks": checks,
        "overall": all(checks.values()),
    }
    for mode, length in (
        ("none", 0),
        ("fixed", 1),
        ("fixed", 2),
        ("fixed", 4),
        ("random", 1),
        ("random", 2),
        ("random", 4),
    ):
        result["budget_sensitivity"][f"{mode}:{length}"] = aggregate_rows([
            rows_for_arm(arm, test, mode, length)
            for arm in main_arms
        ])
    for objective in ("absolute", "quotient", "oracle"):
        reference_arms = select_arms(arms, objective, "all")
        if reference_arms:
            result[f"{objective}_fixed2"] = aggregate_rows([
                rows_for_arm(arm, test, "fixed", 2)
                for arm in reference_arms
            ])
    result["oracle_utility_fixed2"] = aggregate_rows([
        oracle_utility_rows(test),
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
