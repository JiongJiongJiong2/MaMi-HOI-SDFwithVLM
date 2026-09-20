#!/usr/bin/env python3
"""Train strict-contact event baselines on the EPIC lifecycle dataset."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


THRESHOLD_M = 0.001
HISTORY = 4
TARGETS = ("next_contact", "onset", "release")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


def load_jsonl_gzip(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def write_jsonl_gzip(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sigmoid(values):
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if not len(positive) or not len(negative):
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
    positive_rank_sum = ranks[positive].sum()
    return float(
        (
            positive_rank_sum
            - len(positive) * (len(positive) + 1) / 2.0
        )
        / (len(positive) * len(negative))
    )


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    cumulative_tp = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(labels) + 1, dtype=np.float64)
    precision = cumulative_tp / ranks
    return float(precision[sorted_labels == 1].sum() / positives)


def binary_f1(labels, predictions):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def best_f1_threshold(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(scores)
    best = (-1.0, float(candidates[0]) if len(candidates) else 0.0)
    for threshold in candidates:
        f1 = binary_f1(labels, scores >= threshold)
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best[1]


def metric_bundle(labels, scores, threshold=None):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    result = {
        "count": len(labels),
        "positive_count": int(labels.sum()),
        "auroc": roc_auc(labels, scores),
        "auprc": average_precision(labels, scores),
        "brier": float(np.mean((scores - labels) ** 2)),
    }
    if threshold is not None:
        predictions = (scores >= threshold).astype(np.int64)
        result.update({
            "threshold": float(threshold),
            "f1": binary_f1(labels, predictions),
            "precision": (
                float(
                    np.sum((labels == 1) & (predictions == 1))
                    / max(np.sum(predictions == 1), 1)
                )
            ),
            "recall": (
                float(
                    np.sum((labels == 1) & (predictions == 1))
                    / max(np.sum(labels == 1), 1)
                )
            ),
        })
    return result


def contiguous_segments(rows):
    ordered = sorted(rows, key=lambda row: row["frame"])
    segments = []
    current = []
    for row in ordered:
        if current and row["frame"] - current[-1]["frame"] > 1:
            segments.append(current)
            current = []
        current.append(row)
    if current:
        segments.append(current)
    return segments


def build_samples(frames, object_names):
    observations = defaultdict(list)
    for row in frames:
        for hand in ("left", "right"):
            hand_row = row[hand]
            if (
                not hand_row["valid"]
                or hand_row["clip_id"] is None
                or hand_row["min_distance_m"] is None
            ):
                continue
            observations[(
                row["split"],
                row["video_id"],
                hand_row["clip_id"],
                hand,
                hand_row["object_name"],
            )].append({
                "frame": int(row["frame"]),
                "distance": float(hand_row["min_distance_m"]),
                "contact": bool(
                    float(hand_row["min_distance_m"]) <= THRESHOLD_M
                ),
            })

    object_index = {
        name: index for index, name in enumerate(object_names)
    }
    samples = []
    for key, raw_rows in observations.items():
        split, video_id, clip_id, hand, object_name = key
        for segment in contiguous_segments(raw_rows):
            if len(segment) < HISTORY + 1:
                continue
            for index in range(HISTORY - 1, len(segment) - 1):
                history = segment[index - HISTORY + 1 : index + 1]
                current = segment[index]
                following = segment[index + 1]
                contact_history = [
                    int(row["contact"]) for row in history
                ]
                distances = [
                    row["distance"] / THRESHOLD_M for row in history
                ]
                delta1 = (
                    history[-1]["distance"]
                    - history[-2]["distance"]
                ) / THRESHOLD_M
                delta2 = (
                    history[-2]["distance"]
                    - history[-3]["distance"]
                ) / THRESHOLD_M
                hold_duration = 1
                for previous in reversed(history[:-1]):
                    if previous["contact"] != current["contact"]:
                        break
                    hold_duration += 1
                since_switch = 0
                for previous in reversed(history[:-1]):
                    if previous["contact"] != current["contact"]:
                        break
                    since_switch += 1
                object_one_hot = [
                    0.0 for _ in object_names
                ]
                object_one_hot[object_index.get(
                    object_name,
                    len(object_names) - 1,
                )] = 1.0
                features = np.asarray(
                    [
                        distances[-1],
                        delta1,
                        delta2,
                        float(current["contact"]),
                        *contact_history,
                        np.log1p(hold_duration),
                        np.log1p(since_switch),
                        float(hand == "right"),
                        *object_one_hot,
                    ],
                    dtype=np.float64,
                )
                samples.append({
                    "split": split,
                    "participant_id": video_id.split("_", 1)[0],
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "hand": hand,
                    "object_name": object_name,
                    "frame": current["frame"],
                    "distance_m": current["distance"],
                    "current_contact": int(current["contact"]),
                    "next_contact": int(following["contact"]),
                    "onset": int(
                        not current["contact"]
                        and following["contact"]
                    ),
                    "release": int(
                        current["contact"]
                        and not following["contact"]
                    ),
                    "features": features,
                })
    return samples


def split_arrays(samples):
    result = {}
    for split in ("train", "dev", "test"):
        rows = [row for row in samples if row["split"] == split]
        result[split] = {
            "rows": rows,
            "x": np.stack([row["features"] for row in rows]),
            "y": {
                target: np.asarray(
                    [row[target] for row in rows],
                    dtype=np.float64,
                )
                for target in TARGETS
            },
        }
    return result


def standardize(train_x, val_x, test_x):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-8] = 1.0
    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        (test_x - mean) / std,
        mean,
        std,
    )


def train_logistic(
    train_x,
    train_y,
    dev_x,
    dev_y,
    epochs,
    learning_rate,
    l2,
    seed,
):
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=train_x.shape[1])
    bias = 0.0
    positive = max(float(train_y.sum()), 1.0)
    negative = max(float(len(train_y) - train_y.sum()), 1.0)
    pos_weight = np.sqrt(negative / positive)
    best = {
        "score": -np.inf,
        "epoch": -1,
        "weights": weights.copy(),
        "bias": bias,
    }
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    first_bias = second_bias = 0.0
    for epoch in range(epochs):
        logits = train_x @ weights + bias
        probabilities = sigmoid(logits)
        sample_weights = np.where(train_y == 1, pos_weight, 1.0)
        gradient = (
            train_x.T @ (
                sample_weights * (probabilities - train_y)
            )
            / len(train_y)
            + 2.0 * l2 * weights
        )
        bias_gradient = float(
            np.mean(sample_weights * (probabilities - train_y))
        )
        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient**2
        first_bias = beta1 * first_bias + (1 - beta1) * bias_gradient
        second_bias = (
            beta2 * second_bias + (1 - beta2) * bias_gradient**2
        )
        corrected_first = first_moment / (1 - beta1 ** (epoch + 1))
        corrected_second = second_moment / (1 - beta2 ** (epoch + 1))
        weights -= learning_rate * corrected_first / (
            np.sqrt(corrected_second) + eps
        )
        corrected_bias_first = first_bias / (1 - beta1 ** (epoch + 1))
        corrected_bias_second = second_bias / (
            1 - beta2 ** (epoch + 1)
        )
        bias -= learning_rate * corrected_bias_first / (
            np.sqrt(corrected_bias_second) + eps
        )
        dev_score = average_precision(
            dev_y,
            sigmoid(dev_x @ weights + bias),
        )
        if dev_score is not None and dev_score > best["score"]:
            best = {
                "score": float(dev_score),
                "epoch": epoch,
                "weights": weights.copy(),
                "bias": float(bias),
            }
    return best


def copy_scores(rows):
    current = np.asarray(
        [row["current_contact"] for row in rows],
        dtype=np.float64,
    )
    return {
        "next_contact": current,
        "onset": np.full(len(rows), 0.5),
        "release": np.full(len(rows), 0.5),
    }


def distance_linear_scores(rows):
    result = {target: [] for target in TARGETS}
    for row in rows:
        features = row["features"]
        current_scaled = features[0]
        delta = features[1]
        predicted = current_scaled + delta
        contact_score = -predicted
        result["next_contact"].append(contact_score)
        result["onset"].append(contact_score)
        result["release"].append(predicted)
    return {
        target: np.asarray(values, dtype=np.float64)
        for target, values in result.items()
    }


def learned_scores(split, fitted):
    result = {}
    for target in TARGETS:
        weights = fitted[target]["weights"]
        bias = fitted[target]["bias"]
        result[target] = sigmoid(split["x"] @ weights + bias)
    return result


def tune_thresholds(labels, scores):
    return {
        target: best_f1_threshold(labels[target], scores[target])
        for target in TARGETS
    }


def evaluate_scores(split, scores, thresholds):
    return {
        target: metric_bundle(
            split["y"][target],
            scores[target],
            thresholds[target],
        )
        for target in TARGETS
    }


def cluster_bootstrap_difference(
    rows,
    labels,
    first_scores,
    second_scores,
    first_thresholds,
    second_thresholds,
    metric,
    samples,
    seed,
):
    groups = np.asarray([row["participant_id"] for row in rows])
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_indices = [
        np.flatnonzero(inverse == index)
        for index in range(len(unique_groups))
    ]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        selected = rng.integers(
            0,
            len(unique_groups),
            size=len(unique_groups),
        )
        indices = np.concatenate(
            [group_indices[index] for index in selected]
        )
        first = metric_bundle(
            labels[indices],
            first_scores[indices],
            first_thresholds,
        )[metric]
        second = metric_bundle(
            labels[indices],
            second_scores[indices],
            second_thresholds,
        )[metric]
        if first is not None and second is not None:
            draws.append(first - second)
    if not draws:
        return None
    draws = np.asarray(draws, dtype=np.float64)
    return {
        "mean": float(draws.mean()),
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def main():
    args = parse_args()
    frames = load_jsonl_gzip(args.frames)
    object_names = sorted({
        row[hand]["object_name"]
        for row in frames
        for hand in ("left", "right")
        if row[hand]["valid"] and row[hand]["object_name"] is not None
    })
    samples = build_samples(frames, object_names)
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
            args.epochs,
            args.learning_rate,
            args.l2,
            args.seed + offset,
        )
        fitted[target]["x_mean"] = mean
        fitted[target]["x_std"] = std

    model_scores = {
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
        model: tune_thresholds(split["dev"]["y"], scores["dev"])
        for model, scores in model_scores.items()
    }
    evaluation = {
        "dev": {
            model: evaluate_scores(
                split["dev"],
                scores["dev"],
                thresholds[model],
            )
            for model, scores in model_scores.items()
        },
        "test": {
            model: evaluate_scores(
                split["test"],
                scores["test"],
                thresholds[model],
            )
            for model, scores in model_scores.items()
        },
    }
    bootstrap = {}
    test_labels = split["test"]["y"]
    test_rows = split["test"]["rows"]
    for target in TARGETS:
        bootstrap[target] = {}
        for baseline in ("copy_current", "distance_linear"):
            bootstrap[target][f"learned_minus_{baseline}"] = {
                metric: cluster_bootstrap_difference(
                    test_rows,
                    test_labels[target],
                    model_scores["learned_logistic"]["test"][target],
                    model_scores[baseline]["test"][target],
                    thresholds["learned_logistic"][target],
                    thresholds[baseline][target],
                    metric,
                    args.bootstrap_samples,
                    args.seed + 1000 + len(target) + len(baseline),
                )
                for metric in ("f1", "auprc")
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "test_predictions.jsonl.gz"
    predictions = []
    for index, row in enumerate(test_rows):
        predictions.append({
            "participant_id": row["participant_id"],
            "video_id": row["video_id"],
            "clip_id": row["clip_id"],
            "hand": row["hand"],
            "object_name": row["object_name"],
            "frame": row["frame"],
            **{
                target: int(test_labels[target][index])
                for target in TARGETS
            },
            **{
                f"{model}_{target}": float(
                    model_scores[model]["test"][target][index]
                )
                for model in model_scores
                for target in TARGETS
            },
        })
    write_jsonl_gzip(prediction_path, predictions)
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
    result = {
        "threshold_m": THRESHOLD_M,
        "history": HISTORY,
        "targets": list(TARGETS),
        "object_names": object_names,
        "counts": counts,
        "fitted": {
            target: {
                "best_epoch": fitted[target]["epoch"],
                "best_dev_auprc": fitted[target]["score"],
            }
            for target in TARGETS
        },
        "thresholds": thresholds,
        "evaluation": evaluation,
        "bootstrap": bootstrap,
        "outputs": {
            "predictions": str(prediction_path),
            "prediction_rows": len(predictions),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
