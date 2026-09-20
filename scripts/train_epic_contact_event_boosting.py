#!/usr/bin/env python3
"""Train nonlinear strict-contact event models on frozen history features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
except ImportError as error:
    raise SystemExit(
        "scikit-learn is required for the nonlinear event model"
    ) from error

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_epic_contact_event_baselines import (  # noqa: E402
    HISTORY,
    TARGETS,
    THRESHOLD_M,
    build_samples,
    cluster_bootstrap_difference,
    copy_scores,
    distance_linear_scores,
    evaluate_scores,
    learned_scores,
    load_jsonl_gzip,
    split_arrays,
    standardize,
    train_logistic,
    tune_thresholds,
    write_jsonl_gzip,
)


BOOSTING_PARAMETERS = {
    "loss": "log_loss",
    "learning_rate": 0.05,
    "max_iter": 300,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "max_bins": 128,
    "early_stopping": False,
}


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


def train_boosting(train_x, train_y, seed):
    positive = max(float(train_y.sum()), 1.0)
    negative = max(float(len(train_y) - train_y.sum()), 1.0)
    positive_weight = np.sqrt(negative / positive)
    sample_weight = np.where(
        train_y == 1,
        positive_weight,
        1.0,
    )
    model = HistGradientBoostingClassifier(
        **BOOSTING_PARAMETERS,
        random_state=seed,
    )
    model.fit(
        train_x,
        train_y,
        sample_weight=sample_weight,
    )
    return model


def boosting_scores(split, fitted):
    return {
        target: fitted[target].predict_proba(split["x"])[:, 1]
        for target in TARGETS
    }


def gate_result(evaluation, bootstrap):
    logistic = evaluation["test"]["learned_logistic"]
    boosting = evaluation["test"]["boosting"]
    release_auprc = bootstrap["release"][
        "boosting_minus_learned_logistic"
    ]["auprc"]
    checks = {
        "release_auprc_improves": (
            boosting["release"]["auprc"]
            > logistic["release"]["auprc"]
        ),
        "release_auprc_ci_lower_positive": (
            release_auprc is not None
            and release_auprc["ci95"][0] > 0.0
        ),
        "onset_auprc_noninferior": (
            boosting["onset"]["auprc"]
            >= logistic["onset"]["auprc"] - 0.01
        ),
        "next_contact_auprc_noninferior": (
            boosting["next_contact"]["auprc"]
            >= logistic["next_contact"]["auprc"] - 0.01
        ),
        "onset_f1_noninferior": (
            boosting["onset"]["f1"]
            >= logistic["onset"]["f1"] - 0.02
        ),
        "release_f1_noninferior": (
            boosting["release"]["f1"]
            >= logistic["release"]["f1"] - 0.02
        ),
    }
    checks["overall"] = all(checks.values())
    return checks


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

    logistic = {}
    for offset, target in enumerate(TARGETS):
        logistic[target] = train_logistic(
            train_x,
            split["train"]["y"][target],
            dev_x,
            split["dev"]["y"][target],
            args.epochs,
            args.learning_rate,
            args.l2,
            args.seed + offset,
        )
        logistic[target]["x_mean"] = mean
        logistic[target]["x_std"] = std

    boosting = {}
    for offset, target in enumerate(TARGETS):
        boosting[target] = train_boosting(
            split["train"]["x"],
            split["train"]["y"][target],
            args.seed + offset,
        )

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
            "dev": learned_scores(split["dev"], logistic),
            "test": learned_scores(split["test"], logistic),
        },
        "boosting": {
            "dev": boosting_scores(split["dev"], boosting),
            "test": boosting_scores(split["test"], boosting),
        },
    }
    thresholds = {
        model: tune_thresholds(split["dev"]["y"], scores["dev"])
        for model, scores in model_scores.items()
    }
    evaluation = {
        split_name: {
            model: evaluate_scores(
                split[split_name],
                scores[split_name],
                thresholds[model],
            )
            for model, scores in model_scores.items()
        }
        for split_name in ("dev", "test")
    }

    test_rows = split["test"]["rows"]
    test_labels = split["test"]["y"]
    bootstrap = {}
    for target in TARGETS:
        bootstrap[target] = {}
        for baseline in (
            "copy_current",
            "distance_linear",
            "learned_logistic",
        ):
            bootstrap[target][f"boosting_minus_{baseline}"] = {
                metric: cluster_bootstrap_difference(
                    test_rows,
                    test_labels[target],
                    model_scores["boosting"]["test"][target],
                    model_scores[baseline]["test"][target],
                    thresholds["boosting"][target],
                    thresholds[baseline][target],
                    metric,
                    args.bootstrap_samples,
                    args.seed + 2000 + len(target) + len(baseline),
                )
                for metric in ("f1", "auprc")
            }

    gate = gate_result(evaluation, bootstrap)
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
        "boosting_parameters": BOOSTING_PARAMETERS,
        "versions": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "thresholds": thresholds,
        "evaluation": evaluation,
        "bootstrap": bootstrap,
        "gate": gate,
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
