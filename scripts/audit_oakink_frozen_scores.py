"""Audit saved OakInk scores without training or importing scikit-learn."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    scores_path = args.workspace / "oakink_online_handover_detector_v1_20260921/server_output/online_predictions.npz"
    candidates_path = args.workspace / "oakink_handover_manifest_v1_20260921/role_switch_candidates.jsonl.gz"
    summary_path = repo / "docs/experiments/oakink_sequence_aware_detector_summary_v1_20260921.json"
    arrays = np.load(scores_path, allow_pickle=False)
    recorded = json.loads(summary_path.read_text(encoding="utf-8"))
    threshold = recorded["threshold"]
    grouped = {}
    with gzip.open(candidates_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if abs(float(row.get("threshold_m", 0.005)) - 0.005) < 1e-12:
                grouped.setdefault(row["sequence"], []).append(row)
    truth = {}
    for sequence, rows in grouped.items():
        primary = min(rows, key=lambda r: (
            abs(int(r["onset_minus_release"])), int(r["giver_release"])
        ))
        truth[sequence] = int(primary["giver_release"])
    result = {
        "scope": "Frozen-score audit; no training or threshold selection. Source-filter ablation is descriptive, not a new held-out experiment.",
        "threshold": threshold,
        "replay": {},
        "without_source_filter": {},
        "timing_errors": {},
        "source_counts": {},
    }
    for split in ("train", "val", "test"):
        result["source_counts"][split] = {
            source: int(np.sum((arrays["splits"] == split) & (arrays["sources"] == source)))
            for source in ("handover", "nonhandover")
        }
        for filter_source in (True, False):
            tp = fp = total_truth = 0
            errors = []
            for i, sequence in enumerate(arrays["sequences"]):
                if arrays["splits"][i] != split:
                    continue
                sequence = str(sequence)
                total_truth += int(sequence in truth)
                if filter_source and arrays["sources"][i] != "handover":
                    continue
                scores = arrays["scores"][i, :int(arrays["lengths"][i])]
                if not len(scores) or float(scores.max()) < threshold:
                    continue
                predicted = int(np.argmax(scores)) + 1
                matched = sequence in truth and abs(predicted - truth[sequence]) <= 15
                tp += int(matched)
                fp += int(not matched)
                if matched:
                    errors.append(predicted - truth[sequence])
            fn = total_truth - tp
            metrics = {
                "true_positive": tp, "false_positive": fp, "false_negative": fn,
                "total_truth": total_truth, "total_predictions": tp + fp,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / total_truth if total_truth else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            }
            if filter_source:
                metrics["matches_recorded"] = all(
                    abs(value - recorded["splits"][split][key]) < 1e-12
                    for key, value in metrics.items()
                )
                result["replay"][split] = metrics
                result["timing_errors"][split] = {
                    "matched": len(errors),
                    "late_matches": sum(e > 0 for e in errors),
                    "on_time_matches": sum(e == 0 for e in errors),
                    "early_matches": sum(e < 0 for e in errors),
                    "median_error_frames": float(np.median(errors)) if errors else None,
                }
            else:
                result["without_source_filter"][split] = metrics
    trajectory_path = args.workspace / "oakink_handover_manifest_v1_20260921/handover_trajectories_portable.npz"
    traj = np.load(trajectory_path, allow_pickle=False)
    result["trajectory_keys"] = list(traj.files)
    result["participant_pairs"] = {
        s: sorted(set(map(str, traj["participant_pairs"][traj["splits"] == s])))
        for s in ("train", "val", "test")
    }
    result["actual_participants"] = {
        split: sorted({person for pair in pairs for person in pair.split("->")})
        for split, pairs in result["participant_pairs"].items()
    }
    negative_path = args.workspace / "oakink_cross_intent_trajectories_v1_20260921/trajectories.npz"
    negatives = np.load(negative_path, allow_pickle=False)
    result["nonhandover_intent_counts"] = {
        split: {
            intent: int(np.sum((negatives["splits"] == split) & (negatives["intent_names"] == intent)))
            for intent in ("use", "hold", "liftup")
        }
        for split in ("train", "val", "test")
    }
    result["object_overlap"] = {
        a + "_" + b: len(set(traj["object_names"][traj["splits"] == a]) & set(traj["object_names"][traj["splits"] == b]))
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    result["sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            scores_path, candidates_path, summary_path, trajectory_path, negative_path,
            repo / "scripts/evaluate_oakink_sequence_aware_handover_detector.py",
            repo / "scripts/evaluate_oakink_online_handover_detector.py",
            repo / "scripts/build_oakink_handover_windows.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    assert all(row["matches_recorded"] for row in result["replay"].values())


if __name__ == "__main__":
    main()
