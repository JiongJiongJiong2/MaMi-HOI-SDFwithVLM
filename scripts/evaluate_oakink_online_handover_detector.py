#!/usr/bin/env python3
"""Evaluate full-sequence online handover event detection on OakInk."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.build_oakink_handover_windows import extract_window
    from scripts.train_arctic_role_switch_candidates import (
        best_f1_threshold,
        fit_predict,
        metric_bundle,
        out_of_fold_scores,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_oakink_handover_windows import extract_window  # noqa: E402
    from train_arctic_role_switch_candidates import (  # noqa: E402
        best_f1_threshold,
        fit_predict,
        metric_bundle,
        out_of_fold_scores,
    )


HISTORY = 30
MATCH_TOLERANCE_FRAMES = 15
NMS_FRAMES = 15
PRIMARY_THRESHOLD_M = 0.005


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handover-windows", type=Path, required=True)
    parser.add_argument("--handover-trajectories", type=Path, required=True)
    parser.add_argument("--nonhandover-trajectories", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def read_primary_events(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            threshold = float(
                row.get("threshold_m", PRIMARY_THRESHOLD_M)
            )
            if abs(threshold - PRIMARY_THRESHOLD_M) < 1e-12:
                rows.append(row)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["sequence"], []).append(row)
    return {
        sequence: int(min(
            items,
            key=lambda row: (
                abs(int(row["onset_minus_release"])),
                int(row["giver_release"]),
            ),
        )["giver_release"])
        for sequence, items in grouped.items()
    }


def fit_logistic(train_features, train_labels, seed):
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
    model.fit(train_features, train_labels)
    return model


def score_windows(model, features):
    flat = features.reshape(len(features), -1)
    return model.predict_proba(flat)[:, 1]


def score_sequence(model, giver, receiver):
    features = [
        extract_window(giver, receiver, frame, HISTORY)
        for frame in range(1, len(giver))
    ]
    if not features:
        return np.empty(0, dtype=np.float64)
    return score_windows(model, np.stack(features))


def peak_frames(scores, threshold, nms_frames):
    mask = scores >= threshold
    candidates = []
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        end = index
        while end < len(mask) and mask[end]:
            end += 1
        local = index + int(np.argmax(scores[index:end]))
        candidates.append((local + 1, float(scores[local])))
        index = end
    candidates.sort(key=lambda row: -row[1])
    kept = []
    for frame, score in candidates:
        if any(abs(frame - other) <= nms_frames for other, _ in kept):
            continue
        kept.append((frame, score))
    return sorted(kept)


def match_events(truth_by_sequence, predictions_by_sequence, tolerance):
    matched_predictions = set()
    matched_truth = 0
    total_truth = 0
    for sequence, truth_frames in truth_by_sequence.items():
        predictions = sorted(predictions_by_sequence.get(sequence, []))
        for truth in truth_frames:
            total_truth += 1
            candidates = [
                (abs(prediction - truth), prediction)
                for prediction in predictions
                if abs(prediction - truth) <= tolerance
                and (sequence, prediction) not in matched_predictions
            ]
            if candidates:
                _, prediction = min(candidates)
                matched_predictions.add((sequence, prediction))
                matched_truth += 1
    total_predictions = sum(
        len(frames) for frames in predictions_by_sequence.values()
    )
    precision = (
        matched_predictions.__len__() / total_predictions
        if total_predictions
        else 0.0
    )
    recall = matched_truth / total_truth if total_truth else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": matched_truth,
        "false_positive": total_predictions - len(matched_predictions),
        "false_negative": total_truth - matched_truth,
        "total_truth": total_truth,
        "total_predictions": total_predictions,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def build_scores(
    model,
    handover_arrays,
    nonhandover_arrays,
):
    sequences = []
    splits = []
    sources = []
    scores = []
    for index, sequence in enumerate(handover_arrays["sequences"]):
        length = int(handover_arrays["lengths"][index])
        giver = handover_arrays["giver_distance_m"][index, :length]
        receiver = handover_arrays["receiver_distance_m"][index, :length]
        sequences.append(str(sequence))
        splits.append(str(handover_arrays["splits"][index]))
        sources.append("handover")
        scores.append(score_sequence(model, giver, receiver))
    for index, sequence in enumerate(nonhandover_arrays["sequences"]):
        length = int(nonhandover_arrays["lengths"][index])
        active = nonhandover_arrays["distance_m"][index, :length]
        absent = np.full(length, 0.020, dtype=np.float64)
        sequences.append(str(sequence))
        splits.append(str(nonhandover_arrays["splits"][index]))
        sources.append("nonhandover")
        scores.append(score_sequence(model, active, absent))
    return {
        "sequences": sequences,
        "splits": np.asarray(splits),
        "sources": np.asarray(sources),
        "scores": scores,
    }


def write_scored_predictions(path, scored):
    maximum_length = max(
        (len(scores) for scores in scored["scores"]),
        default=0,
    )
    scores = np.full(
        (len(scored["sequences"]), maximum_length),
        np.nan,
        dtype=np.float32,
    )
    lengths = np.zeros(len(scored["sequences"]), dtype=np.int32)
    for index, values in enumerate(scored["scores"]):
        length = len(values)
        lengths[index] = length
        scores[index, :length] = values
    np.savez_compressed(
        path,
        sequences=np.asarray(scored["sequences"]),
        splits=scored["splits"],
        sources=scored["sources"],
        lengths=lengths,
        scores=scores,
    )


def evaluate_split(
    split,
    scored,
    truth,
    threshold,
):
    predictions = {}
    for sequence, split_value, scores in zip(
        scored["sequences"],
        scored["splits"],
        scored["scores"],
    ):
        if split_value != split:
            continue
        peaks = peak_frames(scores, threshold, NMS_FRAMES)
        if peaks:
            predictions[sequence] = [frame for frame, _ in peaks]
    split_lookup = {
        sequence: split_value
        for sequence, split_value in zip(
            scored["sequences"],
            scored["splits"],
        )
    }
    split_truth = {
        sequence: [frame]
        for sequence, frame in truth.items()
        if sequence in scored["sequences"]
        and split_lookup[sequence] == split
    }
    result = match_events(
        split_truth,
        predictions,
        MATCH_TOLERANCE_FRAMES,
    )
    result["false_positive_sequences"] = len({
        sequence for sequence in predictions
        if sequence not in split_truth
        or any(
            min(
                [abs(frame - truth) for truth in split_truth[sequence]],
                default=10**9,
            )
            > MATCH_TOLERANCE_FRAMES
            for frame in predictions[sequence]
        )
    })
    return result, predictions


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handover_windows = np.load(
        args.handover_windows,
        allow_pickle=True,
    )
    handover_features = handover_windows["features"]
    handover_flat = handover_features.reshape(
        len(handover_features),
        -1,
    )
    handover_labels = handover_windows["labels"]
    handover_splits = handover_windows["splits"]
    handover_participants = handover_windows["participants"]
    train = handover_splits == "train"

    oof = out_of_fold_scores(
        "logistic",
        handover_flat[train],
        handover_labels[train],
        handover_participants[train],
        min(4, len(set(handover_participants[train]))),
        args.seed,
    )
    threshold = best_f1_threshold(handover_labels[train], oof)
    model = fit_logistic(
        handover_flat[train],
        handover_labels[train],
        args.seed,
    )

    handover_arrays = np.load(
        args.handover_trajectories,
        allow_pickle=False,
    )
    nonhandover_arrays = np.load(
        args.nonhandover_trajectories,
        allow_pickle=False,
    )
    scored = build_scores(
        model,
        handover_arrays,
        nonhandover_arrays,
    )
    truth = read_primary_events(args.candidates)
    split_results = {}
    prediction_sets = {}
    for split in ("train", "val", "test"):
        result, predictions = evaluate_split(
            split,
            scored,
            truth,
            threshold,
        )
        split_results[split] = result
        prediction_sets[split] = predictions
    gate = {
        split: {
            "precision_ge_0_5": row["precision"] >= 0.5,
            "recall_ge_0_5": row["recall"] >= 0.5,
            "f1_ge_0_5": row["f1"] >= 0.5,
        }
        for split, row in split_results.items()
    }
    for split, checks in gate.items():
        checks["overall"] = all(checks.values())
    result = {
        "threshold": float(threshold),
        "train_oof": metric_bundle(
            handover_labels[train],
            oof,
            threshold,
        ),
        "splits": split_results,
        "gate": gate,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_scored_predictions(
        args.output_dir / "online_predictions.npz",
        scored,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
