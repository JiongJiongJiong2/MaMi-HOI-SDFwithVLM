#!/usr/bin/env python3
"""Run feature-group and seed stability audits for ARCTIC motion windows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.train_arctic_role_switch_candidates import (
        best_f1_threshold,
        fit_predict,
        metric_bundle,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_arctic_role_switch_candidates import (  # noqa: E402
        best_f1_threshold,
        fit_predict,
        metric_bundle,
    )


FEATURE_GROUPS = {
    "distance": list(range(0, 10)),
    "hand_speed": [10, 11],
    "relative_speed": [12, 13],
    "object_motion": [14, 15],
    "motion_only": list(range(10, 16)),
    "all": list(range(0, 16)),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[20260921, 20260922, 20260923, 20260924, 20260925],
    )
    parser.add_argument("--folds", type=int, default=4)
    return parser.parse_args()


def feature_indices(name):
    groups = dict(FEATURE_GROUPS)
    groups["all_minus_hand_speed"] = [
        index for index in range(16) if index not in {10, 11}
    ]
    groups["all_minus_relative_speed"] = [
        index for index in range(16) if index not in {12, 13}
    ]
    groups["all_minus_object_motion"] = [
        index for index in range(16) if index not in {14, 15}
    ]
    return groups[name]


def select_features(features, name):
    selected = features[:, :, feature_indices(name)]
    return selected.reshape(len(selected), -1)


def balanced_group_folds(participants, labels, folds, seed):
    rng = np.random.default_rng(seed)
    groups = np.asarray(sorted(set(participants)))
    rng.shuffle(groups)
    random_order = {
        group: index for index, group in enumerate(groups)
    }
    positive_count = {
        group: int(labels[participants == group].sum())
        for group in groups
    }
    sample_count = {
        group: int((participants == group).sum())
        for group in groups
    }
    order = sorted(
        groups,
        key=lambda group: (
            -positive_count[group],
            -sample_count[group],
            random_order[group],
        ),
    )
    fold_positive = [0] * folds
    fold_samples = [0] * folds
    assignment = {}
    for group in order:
        fold = min(
            range(folds),
            key=lambda index: (
                fold_positive[index],
                fold_samples[index],
                index,
            ),
        )
        assignment[group] = fold
        fold_positive[fold] += positive_count[group]
        fold_samples[fold] += sample_count[group]
    result = []
    group_fold = np.asarray([assignment[group] for group in participants])
    for fold in range(folds):
        result.append((
            np.flatnonzero(group_fold != fold),
            np.flatnonzero(group_fold == fold),
        ))
    return result


def out_of_fold_boosting(
    train_x,
    train_y,
    groups,
    folds,
    seed,
):
    scores = np.full(len(train_y), np.nan, dtype=np.float64)
    for fold_index, (fit_index, score_index) in enumerate(
        balanced_group_folds(groups, train_y, folds, seed)
    ):
        scores[score_index] = fit_predict(
            "boosting",
            train_x[fit_index],
            train_y[fit_index],
            train_x[score_index],
            seed + fold_index,
        )
    if np.isnan(scores).any():
        raise RuntimeError("incomplete out-of-fold scores")
    return scores


def evaluate_feature_set(
    raw_features,
    labels,
    splits,
    participants,
    feature_name,
    folds,
    seed,
):
    features = select_features(raw_features, feature_name)
    train = splits == "train"
    val = splits == "val"
    oof = out_of_fold_boosting(
        features[train],
        labels[train],
        participants[train],
        folds,
        seed,
    )
    threshold = best_f1_threshold(labels[train], oof)
    val_scores = fit_predict(
        "boosting",
        features[train],
        labels[train],
        features[val],
        seed,
    )
    return {
        "feature_name": feature_name,
        "feature_count": int(features.shape[-1]),
        "threshold": float(threshold),
        "train_oof": metric_bundle(
            labels[train],
            oof,
            threshold,
        ),
        "val": metric_bundle(
            labels[val],
            val_scores,
            threshold,
        ),
        "oof_scores": oof,
        "val_scores": val_scores,
    }


def expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=np.float64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (
                (probabilities >= edges[index])
                & (probabilities <= edges[index + 1])
            )
        else:
            mask = (
                (probabilities >= edges[index])
                & (probabilities < edges[index + 1])
            )
        if not mask.any():
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(labels[mask].mean())
        result += float(mask.mean()) * abs(confidence - accuracy)
    return float(result)


def calibration_report(labels, probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    return {
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "ece": expected_calibration_error(labels, probabilities),
    }


def fit_calibrators(oof_labels, oof_scores, val_scores):
    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(oof_scores, oof_labels)
    isotonic_scores = isotonic.predict(val_scores)
    platt = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="liblinear",
            random_state=20260921,
        ),
    )
    platt.fit(np.asarray(oof_scores).reshape(-1, 1), oof_labels)
    platt_scores = platt.predict_proba(
        np.asarray(val_scores).reshape(-1, 1)
    )[:, 1]
    return isotonic_scores, platt_scores


def summary_row(result):
    return {
        "feature_count": result["feature_count"],
        "threshold": result["threshold"],
        "train_oof_auprc": result["train_oof"]["auprc"],
        "train_oof_f1": result["train_oof"]["f1"],
        "val_auprc": result["val"]["auprc"],
        "val_auroc": result["val"]["auroc"],
        "val_f1": result["val"]["f1"],
        "val_precision": result["val"]["precision"],
        "val_recall": result["val"]["recall"],
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.windows, allow_pickle=True)
    features = arrays["features"]
    labels = arrays["labels"]
    splits = arrays["splits"]
    participants = arrays["participants"]
    train = splits == "train"
    val = splits == "val"

    ablation_names = [
        "distance",
        "hand_speed",
        "relative_speed",
        "object_motion",
        "motion_only",
        "all",
        "all_minus_hand_speed",
        "all_minus_relative_speed",
        "all_minus_object_motion",
    ]
    ablation = {}
    primary_results = {}
    for name in ablation_names:
        result = evaluate_feature_set(
            features,
            labels,
            splits,
            participants,
            name,
            args.folds,
            args.seeds[0],
        )
        ablation[name] = summary_row(result)
        primary_results[name] = result

    seed_results = {
        name: []
        for name in ("distance", "motion_only", "all")
    }
    for seed in args.seeds:
        for name in seed_results:
            result = evaluate_feature_set(
                features,
                labels,
                splits,
                participants,
                name,
                args.folds,
                seed,
            )
            seed_results[name].append({
                "seed": seed,
                **summary_row(result),
                "oof_scores": result["oof_scores"],
                "val_scores": result["val_scores"],
            })

    all_primary = primary_results["all"]
    isotonic_scores, platt_scores = fit_calibrators(
        labels[train],
        all_primary["oof_scores"],
        all_primary["val_scores"],
    )
    calibration = {
        "raw": calibration_report(
            labels[val],
            all_primary["val_scores"],
        ),
        "isotonic": calibration_report(
            labels[val],
            isotonic_scores,
        ),
        "platt": calibration_report(
            labels[val],
            platt_scores,
        ),
    }

    all_oof = np.asarray([
        row["train_oof_auprc"] for row in seed_results["all"]
    ])
    distance_oof = np.asarray([
        row["train_oof_auprc"] for row in seed_results["distance"]
    ])
    all_val_auprc = np.asarray([
        row["val_auprc"] for row in seed_results["all"]
    ])
    all_val_f1 = np.asarray([
        row["val_f1"] for row in seed_results["all"]
    ])
    gate = {
        "mean_all_oof_above_distance": (
            float(all_oof.mean()) > float(distance_oof.mean())
        ),
        "all_oof_better_in_at_least_4_of_5": (
            int(np.sum(all_oof > distance_oof)) >= 4
        ),
        "mean_val_auprc_ge_0_6484": (
            float(all_val_auprc.mean()) >= 0.6484
        ),
        "mean_val_f1_ge_0_6154": (
            float(all_val_f1.mean()) >= 0.6154
        ),
        "official_test_available": False,
    }
    gate["stable_motion_overall"] = all((
        gate["mean_all_oof_above_distance"],
        gate["all_oof_better_in_at_least_4_of_5"],
        gate["mean_val_auprc_ge_0_6484"],
        gate["mean_val_f1_ge_0_6154"],
    ))
    gate["final_benchmark_ready"] = False
    seed_summary = {
        name: {
            "oof_auprc_mean": float(np.mean([
                row["train_oof_auprc"] for row in rows
            ])),
            "oof_auprc_std": float(np.std([
                row["train_oof_auprc"] for row in rows
            ])),
            "val_auprc_mean": float(np.mean([
                row["val_auprc"] for row in rows
            ])),
            "val_auprc_std": float(np.std([
                row["val_auprc"] for row in rows
            ])),
            "val_f1_mean": float(np.mean([
                row["val_f1"] for row in rows
            ])),
            "val_f1_std": float(np.std([
                row["val_f1"] for row in rows
            ])),
        }
        for name, rows in seed_results.items()
    }
    result = {
        "seeds": args.seeds,
        "folds": args.folds,
        "ablation_primary_seed": ablation,
        "seed_results": {
            name: [
                {
                    key: value
                    for key, value in row.items()
                    if not key.endswith("scores")
                }
                for row in rows
            ]
            for name, rows in seed_results.items()
        },
        "seed_summary": seed_summary,
        "calibration_primary_all": calibration,
        "gate": gate,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
