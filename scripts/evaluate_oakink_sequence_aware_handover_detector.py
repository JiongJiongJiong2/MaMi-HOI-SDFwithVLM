#!/usr/bin/env python3
"""Apply frozen sequence-aware post-processing to OakInk v1 scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from scripts.evaluate_oakink_online_handover_detector import (
        match_events,
        read_primary_events,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate_oakink_online_handover_detector import (  # noqa: E402
        match_events,
        read_primary_events,
    )


MATCH_TOLERANCE_FRAMES = 15


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-predictions", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def primary_predictions(arrays, threshold):
    predictions = {}
    emitted = []
    for index, sequence in enumerate(arrays["sequences"]):
        if str(arrays["sources"][index]) != "handover":
            continue
        length = int(arrays["lengths"][index])
        scores = arrays["scores"][index, :length]
        maximum = float(scores.max()) if len(scores) else -1.0
        if maximum < threshold:
            continue
        frame = int(np.argmax(scores)) + 1
        sequence = str(sequence)
        predictions.setdefault(sequence, []).append(frame)
        emitted.append({
            "sequence": sequence,
            "split": str(arrays["splits"][index]),
            "frame": frame,
            "score": maximum,
        })
    return predictions, emitted


def evaluate_split(split, arrays, truth, threshold):
    predictions, emitted = primary_predictions(arrays, threshold)
    split_lookup = {
        str(sequence): str(arrays["splits"][index])
        for index, sequence in enumerate(arrays["sequences"])
    }
    split_predictions = {
        sequence: frames
        for sequence, frames in predictions.items()
        if split_lookup[sequence] == split
    }
    split_truth = {
        sequence: [frame]
        for sequence, frame in truth.items()
        if split_lookup.get(sequence) == split
    }
    result = match_events(
        split_truth,
        split_predictions,
        MATCH_TOLERANCE_FRAMES,
    )
    result["emitted"] = [
        row for row in emitted if row["split"] == split
    ]
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = np.load(args.online_predictions, allow_pickle=False)
    truth = read_primary_events(args.candidates)
    split_results = {
        split: evaluate_split(
            split,
            arrays,
            truth,
            args.threshold,
        )
        for split in ("train", "val", "test")
    }
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
        "threshold": args.threshold,
        "splits": split_results,
        "gate": gate,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
