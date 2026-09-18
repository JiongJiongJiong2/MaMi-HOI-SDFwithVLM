#!/usr/bin/env python3
"""Select contiguous contact-eligible windows for temporal ContactOpt tests."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--threshold-m", type=float, default=0.02)
    parser.add_argument("--min-run", type=int, default=5)
    parser.add_argument("--max-window", type=int, default=8)
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--exclude-sequence", nargs="*", default=[])
    return parser.parse_args()


def contiguous_runs(mask):
    runs = []
    start = None
    for index, enabled in enumerate(mask):
        if enabled and start is None:
            start = index
        if start is not None and (not enabled or index == len(mask) - 1):
            end = index if enabled and index == len(mask) - 1 else index - 1
            runs.append((start, end))
            start = None
    return runs


def choose_window(run, max_window):
    start, end = run
    length = end - start + 1
    if length <= max_window:
        return list(range(start, end + 1))
    offset = (length - max_window) // 2
    window_start = start + offset
    return list(range(window_start, window_start + max_window))


def frame_eligibility(candidate_path, threshold):
    with np.load(candidate_path, allow_pickle=True) as archive:
        hand = np.asarray(archive["pred_right_hand_verts"])
        obj = np.asarray(archive["pred_object_verts"])
    distances = []
    for frame_index in range(len(hand)):
        nearest = cKDTree(obj[frame_index]).query(hand[frame_index])[0]
        distances.append(float(np.percentile(nearest, 10)))
    return np.asarray(distances, dtype=np.float64)


def main():
    args = parse_args()
    source = json.loads(
        args.source_manifest.read_text(encoding="utf-8")
    )
    selected = []
    rejected = []
    excluded = set(args.exclude_sequence)

    for candidate in source["selected"]:
        if candidate["sequence"] in excluded:
            continue
        p10 = frame_eligibility(Path(candidate["path"]), args.threshold_m)
        runs = contiguous_runs(p10 <= args.threshold_m)
        if not runs:
            rejected.append(
                {
                    "sequence": candidate["sequence"],
                    "reason": "no contiguous eligible run",
                }
            )
            continue
        longest = max(runs, key=lambda run: (run[1] - run[0] + 1, -run[0]))
        run_length = longest[1] - longest[0] + 1
        if run_length < args.min_run:
            rejected.append(
                {
                    "sequence": candidate["sequence"],
                    "reason": "longest contiguous run below minimum",
                    "longest_run": run_length,
                }
            )
            continue
        frames = choose_window(longest, args.max_window)
        selected.append(
            {
                **candidate,
                "frames": frames,
                "contiguous_run": [longest[0], longest[1]],
                "contiguous_run_length": run_length,
                "frame_p10_distance_m": p10.tolist(),
            }
        )
        if len(selected) == args.max_cases:
            break

    result = {
        "source_manifest": str(args.source_manifest),
        "selection_rule": (
            "first manifest candidates with a contiguous p10<=threshold run; "
            "use the longest run, tie by earliest, then take a centered "
            "window capped at max_window frames"
        ),
        "threshold_m": args.threshold_m,
        "min_run": args.min_run,
        "max_window": args.max_window,
        "selected": selected,
        "rejected_before_limit": rejected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
