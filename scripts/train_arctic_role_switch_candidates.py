#!/usr/bin/env python3
"""Train pilot ARCTIC role-switch candidate-quality classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def binary_f1(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if not positives:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(labels) + 1)
    precision = cumulative / ranks
    return float(
        precision[sorted_labels == 1].sum() / positives
    )


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while (
            end < len(scores)
            and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    positives = int(positive.sum())
    negatives = int(negative.sum())
    rank_sum = ranks[positive].sum()
    return float(
        (
            rank_sum
            - positives * (positives + 1) / 2.0
        )
        / (positives * negatives)
    )


def best_f1_threshold(labels, scores):
    candidates = np.unique(scores)
    best = (-1.0, 0.0)
    for threshold in candidates:
        value = binary_f1(labels, scores >= threshold)
        if value > best[0]:
            best = (value, float(threshold))
    return best[1]


def metric_bundle(labels, scores, threshold):
    predictions = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    return {
        "count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "auroc": roc_auc(labels, scores),
        "auprc": average_precision(labels, scores),
        "f1": binary_f1(labels, predictions),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "threshold": float(threshold),
    }


def positive_weight(labels):
    positive = max(float(labels.sum()), 1.0)
    negative = max(float(len(labels) - labels.sum()), 1.0)
    return np.sqrt(negative / positive)


def fit_predict(model_name, train_x, train_y, test_x, seed):
    if model_name == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                solver="liblinear",
                random_state=seed,
                max_iter=2000,
            ),
        )
        model.fit(train_x, train_y)
        return model.predict_proba(test_x)[:, 1]
    if model_name == "boosting":
        weight = positive_weight(train_y)
        sample_weight = np.where(train_y == 1, weight, 1.0)
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            max_bins=128,
            early_stopping=False,
            random_state=seed,
        )
        model.fit(
            train_x,
            train_y,
            sample_weight=sample_weight,
        )
        return model.predict_proba(test_x)[:, 1]
    raise ValueError(model_name)


def out_of_fold_scores(
    model_name,
    train_x,
    train_y,
    groups,
    folds,
    seed,
):
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=folds)
    scores = np.full(len(train_y), np.nan, dtype=np.float64)
    for fold_index, (fit_index, score_index) in enumerate(
        splitter.split(train_x, train_y, groups)
    ):
        scores[score_index] = fit_predict(
            model_name,
            train_x[fit_index],
            train_y[fit_index],
            train_x[score_index],
            seed + fold_index,
        )
    if np.isnan(scores).any():
        raise RuntimeError("out-of-fold scores are incomplete")
    return scores


def merge_split_scores(
    splits,
    train_scores,
    val_scores,
    test_scores=None,
):
    splits = np.asarray(splits)
    result = np.full(len(splits), np.nan, dtype=np.float64)
    result[splits == "train"] = train_scores
    result[splits == "val"] = val_scores
    if test_scores is not None:
        result[splits == "test"] = test_scores
    if np.isnan(result).any():
        raise ValueError("missing scores for one or more splits")
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.windows, allow_pickle=True)
    features = arrays["features"]
    flat = features.reshape(len(features), -1)
    labels = arrays["labels"]
    splits = arrays["splits"]
    participants = arrays["participants"]
    train = splits == "train"
    val = splits == "val"
    test = splits == "test"

    output = {
        "folds": args.folds,
        "feature_shape": list(features.shape[1:]),
        "train": {
            "samples": int(train.sum()),
            "positives": int(labels[train].sum()),
        },
        "val": {
            "samples": int(val.sum()),
            "positives": int(labels[val].sum()),
        },
        "models": {},
    }
    for offset, model_name in enumerate(("logistic", "boosting")):
        oof = out_of_fold_scores(
            model_name,
            flat[train],
            labels[train],
            participants[train],
            args.folds,
            args.seed + 100 * offset,
        )
        threshold = best_f1_threshold(labels[train], oof)
        val_scores = fit_predict(
            model_name,
            flat[train],
            labels[train],
            flat[val],
            args.seed + 100 * offset,
        )
        output["models"][model_name] = {
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
        }
        test_scores = None
        if test.any():
            test_scores = fit_predict(
                model_name,
                flat[train],
                labels[train],
                flat[test],
                args.seed + 100 * offset,
            )
            output["models"][model_name]["test"] = metric_bundle(
                labels[test],
                test_scores,
                threshold,
            )
        combined_scores = merge_split_scores(
            splits,
            oof,
            val_scores,
            test_scores,
        )
        np.savez_compressed(
            args.output_dir / f"{model_name}_predictions.npz",
            labels=labels,
            scores=combined_scores,
            splits=splits,
            participants=participants,
        )

    prevalence = float(np.mean(labels[val]))
    all_positive_f1 = (
        2.0 * prevalence / (1.0 + prevalence)
        if prevalence
        else 0.0
    )
    best_name = max(
        output["models"],
        key=lambda name: output["models"][name]["val"]["auprc"],
    )
    best = output["models"][best_name]
    output["baseline"] = {
        "prevalence": prevalence,
        "all_positive_f1": all_positive_f1,
    }
    output["gate"] = {
        "best_model": best_name,
        "val_auprc_above_prevalence": (
            best["val"]["auprc"] > prevalence
        ),
        "val_f1_above_all_positive": (
            best["val"]["f1"] > all_positive_f1
        ),
        "train_oof_auprc_above_prevalence": (
            best["train_oof"]["auprc"]
            > float(np.mean(labels[train]))
        ),
        "official_test_available": False,
    }
    if test.any():
        test_prevalence = float(np.mean(labels[test]))
        test_all_positive_f1 = (
            2.0 * test_prevalence / (1.0 + test_prevalence)
            if test_prevalence
            else 0.0
        )
        output["test_baseline"] = {
            "prevalence": test_prevalence,
            "all_positive_f1": test_all_positive_f1,
        }
        output["gate"].update({
            "test_auprc_above_prevalence": (
                best["test"]["auprc"] > test_prevalence
            ),
            "test_f1_above_all_positive": (
                best["test"]["f1"] > test_all_positive_f1
            ),
            "test_recall_ge_0_5": best["test"]["recall"] >= 0.5,
        })
    output["gate"]["pilot_overall"] = all((
        output["gate"]["val_auprc_above_prevalence"],
        output["gate"]["val_f1_above_all_positive"],
        output["gate"]["train_oof_auprc_above_prevalence"],
    ))
    output["gate"]["final_benchmark_ready"] = False
    (args.output_dir / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
