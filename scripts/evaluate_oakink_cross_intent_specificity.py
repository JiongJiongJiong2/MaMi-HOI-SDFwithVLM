#!/usr/bin/env python3
"""Evaluate OakInk handover specificity on non-handover release windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scripts.audit_arctic_handover_gate import (
        binary_segments,
        stable_contact,
    )
    from scripts.build_oakink_handover_windows import extract_window
    from scripts.train_arctic_role_switch_candidates import (
        best_f1_threshold,
        fit_predict,
        metric_bundle,
        out_of_fold_scores,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_arctic_handover_gate import (  # noqa: E402
        binary_segments,
        stable_contact,
    )
    from build_oakink_handover_windows import extract_window  # noqa: E402
    from train_arctic_role_switch_candidates import (  # noqa: E402
        best_f1_threshold,
        fit_predict,
        metric_bundle,
        out_of_fold_scores,
    )


CONTACT_THRESHOLD_M = 0.005
MIN_CONTACT_FRAMES = 15
HISTORY = 30


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handover-windows", type=Path, required=True)
    parser.add_argument("--nonhandover-trajectories", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260921)
    return parser.parse_args()


def build_cross_intent_windows(path):
    arrays = np.load(path, allow_pickle=False)
    features = []
    splits = []
    participants = []
    sequences = []
    frames = []
    for index, sequence in enumerate(arrays["sequences"]):
        sequence = str(sequence)
        split = str(arrays["splits"][index])
        participant = str(arrays["subjects"][index])
        length = int(arrays["lengths"][index])
        distance = np.asarray(
            arrays["distance_m"][index, :length],
            dtype=np.float64,
        )
        contact = np.isfinite(distance) & (
            distance <= CONTACT_THRESHOLD_M
        )
        stable = stable_contact(
            contact,
            MIN_CONTACT_FRAMES,
            maximum_gap=0,
        )
        seen_frames = set()
        for _, release in binary_segments(stable):
            if release >= length:
                continue
            for frame in range(
                max(1, release - 15),
                release,
            ):
                if frame in seen_frames:
                    continue
                seen_frames.add(frame)
                receiver = np.full(length, 0.020, dtype=np.float64)
                features.append(extract_window(
                    distance,
                    receiver,
                    frame,
                    HISTORY,
                ))
                splits.append(split)
                participants.append(participant)
                sequences.append(sequence)
                frames.append(frame)
    return {
        "features": np.stack(features),
        "splits": np.asarray(splits),
        "participants": np.asarray(participants),
        "sequences": np.asarray(sequences),
        "frames": np.asarray(frames),
    }


def combined_metrics(labels, scores, threshold):
    return metric_bundle(labels, scores, threshold)


def specificity_summary(
    split,
    handover_labels,
    handover_scores,
    cross_scores,
    threshold,
):
    labels = np.concatenate((
        handover_labels,
        np.zeros(len(cross_scores), dtype=np.int64),
    ))
    scores = np.concatenate((handover_scores, cross_scores))
    metrics = combined_metrics(labels, scores, threshold)
    positive = handover_labels == 1
    predictions = scores >= threshold
    positive_recall = (
        float(predictions[:len(handover_labels)][positive].mean())
        if positive.any()
        else 0.0
    )
    cross_predictions = predictions[len(handover_labels):]
    return {
        "split": split,
        "combined": metrics,
        "handover_positive_recall": positive_recall,
        "cross_intent_count": int(len(cross_scores)),
        "cross_intent_false_positive_rate": float(
            cross_predictions.mean()
        ),
        "cross_intent_score_median": float(
            np.median(cross_scores)
        ),
        "cross_intent_score_q75": float(
            np.quantile(cross_scores, 0.75)
        ),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handover = np.load(args.handover_windows, allow_pickle=True)
    handover_features = handover["features"]
    handover_flat = handover_features.reshape(
        len(handover_features),
        -1,
    )
    handover_labels = handover["labels"]
    handover_splits = handover["splits"]
    handover_participants = handover["participants"]
    train = handover_splits == "train"

    oof = out_of_fold_scores(
        "logistic",
        handover_flat[train],
        handover_labels[train],
        handover_participants[train],
        min(args.folds, len(set(handover_participants[train]))),
        args.seed,
    )
    threshold = best_f1_threshold(handover_labels[train], oof)
    handover_scores = fit_predict(
        "logistic",
        handover_flat[train],
        handover_labels[train],
        handover_flat,
        args.seed,
    )

    cross = build_cross_intent_windows(
        args.nonhandover_trajectories
    )
    cross_flat = cross["features"].reshape(len(cross["features"]), -1)
    cross_scores = fit_predict(
        "logistic",
        handover_flat[train],
        handover_labels[train],
        cross_flat,
        args.seed,
    )

    split_results = {}
    for split in ("val", "test"):
        handover_mask = handover_splits == split
        cross_mask = cross["splits"] == split
        if not handover_mask.any() or not cross_mask.any():
            raise RuntimeError(f"missing split data for {split}")
        split_results[split] = specificity_summary(
            split,
            handover_labels[handover_mask],
            handover_scores[handover_mask],
            cross_scores[cross_mask],
            threshold,
        )

    gate = {
        split: {
            "positive_recall_ge_0_5": (
                row["handover_positive_recall"] >= 0.5
            ),
            "cross_intent_fpr_le_0_10": (
                row["cross_intent_false_positive_rate"] <= 0.10
            ),
            "combined_auprc_above_prevalence": (
                row["combined"]["auprc"]
                > row["combined"]["positive_count"]
                / row["combined"]["count"]
            ),
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
        "split_results": split_results,
        "gate": gate,
        "cross_intent_windows": int(len(cross["features"])),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "specificity_predictions.npz",
        handover_scores=handover_scores,
        handover_labels=handover_labels,
        handover_splits=handover_splits,
        cross_scores=cross_scores,
        cross_splits=cross["splits"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
