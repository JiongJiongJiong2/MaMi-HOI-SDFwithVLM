#!/usr/bin/env python3
"""Build a sequence-disjoint dense window dataset for temporal refinement."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "pred_right_hand_verts",
    "pred_object_verts",
    "object_faces",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, nargs="+", required=True)
    parser.add_argument("--object", dest="objects", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--threshold-m", type=float, default=0.02)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--min-run", type=int, default=8)
    parser.add_argument("--max-windows-per-sequence", type=int, default=8)
    parser.add_argument("--min-size-mb", type=float, default=10.0)
    parser.add_argument(
        "--dev-sequence",
        nargs="*",
        default=[],
        help="Sequences already inspected during temporal development.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=20260919)
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


def iter_shifted_windows(start, end, window, stride):
    run_length = end - start + 1
    if run_length < window:
        return
    final_start = end - window + 1
    starts = list(range(start, final_start + 1, stride))
    if starts[-1] != final_start:
        starts.append(final_start)
    for window_start in starts:
        yield window_start, window_start + window - 1


def evenly_spaced_indices(count, maximum):
    if count <= maximum:
        return list(range(count))
    if maximum == 1:
        return [count // 2]
    last = count - 1
    return sorted(
        {
            int(round(index * last / (maximum - 1)))
            for index in range(maximum)
        }
    )


def stable_sequence_bucket(sequence, seed):
    digest = hashlib.sha256(f"{seed}:{sequence}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def infer_object_name(path, archive):
    if "object_name" in archive.files:
        value = archive["object_name"]
        if value.shape == ():
            return str(value.item())
    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer object name from {path}")
    return parts[1]


def infer_sequence_name(path, archive):
    if "seq_name" in archive.files:
        value = archive["seq_name"]
        if value.shape == ():
            return str(value.item())
    return path.stem


def load_p10_distances(path, threshold):
    from scipy.spatial import cKDTree

    with np.load(path, allow_pickle=True) as archive:
        hand = np.asarray(archive["pred_right_hand_verts"])
        obj = np.asarray(archive["pred_object_verts"])
    if hand.ndim != 3 or obj.ndim != 3:
        raise ValueError(
            f"Unexpected geometry shapes in {path}: {hand.shape}, {obj.shape}"
        )
    if len(hand) != len(obj):
        raise ValueError(
            f"Hand/object frame mismatch in {path}: {len(hand)} vs {len(obj)}"
        )
    distances = []
    for frame_index in range(len(hand)):
        nearest = cKDTree(obj[frame_index]).query(hand[frame_index])[0]
        distances.append(float(np.percentile(nearest, 10)))
    distances = np.asarray(distances, dtype=np.float64)
    return distances, distances <= threshold


def scan_candidates(roots, objects, min_size_mb):
    minimum_size = int(min_size_mb * 1024 * 1024)
    allowed = set(objects)
    by_sequence = {}
    ignored = {
        "too_small": 0,
        "missing_keys": 0,
        "wrong_object": 0,
    }

    for root in roots:
        for path in sorted(root.rglob("*.npz")):
            if path.stat().st_size < minimum_size:
                ignored["too_small"] += 1
                continue
            with np.load(path, allow_pickle=True) as archive:
                if not REQUIRED_KEYS.issubset(archive.files):
                    ignored["missing_keys"] += 1
                    continue
                object_name = infer_object_name(path, archive)
                if object_name not in allowed:
                    ignored["wrong_object"] += 1
                    continue
                sequence = infer_sequence_name(path, archive)
            candidate = {
                "path": str(path.resolve()),
                "sequence": sequence,
                "object_name": object_name,
                "size_bytes": path.stat().st_size,
            }
            previous = by_sequence.get(sequence)
            if previous is None or (
                candidate["object_name"],
                candidate["path"],
            ) < (
                previous["object_name"],
                previous["path"],
            ):
                by_sequence[sequence] = candidate

    return [by_sequence[key] for key in sorted(by_sequence)], ignored


def split_sequences(sequences, dev_sequences, test_fraction, split_seed):
    sequence_set = set(sequences)
    explicit_dev = sorted(sequence_set.intersection(dev_sequences))
    remaining = sorted(sequence_set.difference(explicit_dev))
    test = [
        sequence
        for sequence in remaining
        if stable_sequence_bucket(sequence, split_seed) < test_fraction
    ]
    test_set = set(test)
    train = [sequence for sequence in remaining if sequence not in test_set]
    return {
        "train": train,
        "dev": explicit_dev,
        "test": test,
    }


def summarize_windows(windows, split_sequences):
    summary = {}
    for split in ("train", "dev", "test"):
        selected = [
            window
            for window in windows
            if window["sequence"] in set(split_sequences[split])
        ]
        summary[split] = {
            "assigned_sequence_count": len(split_sequences[split]),
            "sequence_count": len(
                {window["sequence"] for window in selected}
            ),
            "window_count": len(selected),
            "frame_count": int(
                sum(len(window["frames"]) for window in selected)
            ),
            "objects": sorted(
                {window["object_name"] for window in selected}
            ),
        }
    return summary


def main():
    args = parse_args()
    if args.window < 2:
        raise ValueError("--window must be at least 2")
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    if args.min_run < args.window:
        raise ValueError("--min-run must be at least --window")
    if args.max_windows_per_sequence < 1:
        raise ValueError("--max-windows-per-sequence must be positive")
    if not 0.0 <= args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be in [0, 1)")

    candidates, ignored = scan_candidates(
        args.root,
        args.objects,
        args.min_size_mb,
    )
    sequence_to_object = {
        candidate["sequence"]: candidate["object_name"]
        for candidate in candidates
    }
    sequence_splits = split_sequences(
        list(sequence_to_object),
        set(args.dev_sequence),
        args.test_fraction,
        args.split_seed,
    )
    split_by_sequence = {
        sequence: split
        for split, sequences in sequence_splits.items()
        for sequence in sequences
    }

    windows = []
    rejected = []
    for candidate in candidates:
        sequence = candidate["sequence"]
        path = Path(candidate["path"])
        try:
            p10, eligible = load_p10_distances(path, args.threshold_m)
        except (OSError, ValueError) as error:
            rejected.append(
                {
                    "sequence": sequence,
                    "object_name": candidate["object_name"],
                    "path": str(path),
                    "reason": str(error),
                }
            )
            continue

        runs = contiguous_runs(eligible)
        sequence_windows = []
        for start, end in runs:
            if end - start + 1 < args.min_run:
                continue
            for window_start, window_end in iter_shifted_windows(
                start,
                end,
                args.window,
                args.stride,
            ):
                frames = list(range(window_start, window_end + 1))
                sequence_windows.append(
                    {
                        "sequence": sequence,
                        "object_name": candidate["object_name"],
                        "path": candidate["path"],
                        "size_bytes": candidate["size_bytes"],
                        "split": split_by_sequence[sequence],
                        "frames": frames,
                        "contiguous_run": [start, end],
                        "contiguous_run_length": end - start + 1,
                        "p10_min_m": float(p10[window_start : window_end + 1].min()),
                        "p10_mean_m": float(
                            p10[window_start : window_end + 1].mean()
                        ),
                        "p10_max_m": float(p10[window_start : window_end + 1].max()),
                    }
                )

        if not sequence_windows:
            rejected.append(
                {
                    "sequence": sequence,
                    "object_name": candidate["object_name"],
                    "path": str(path),
                    "reason": "no eligible contiguous run at minimum length",
                }
            )
            continue

        keep = evenly_spaced_indices(
            len(sequence_windows),
            args.max_windows_per_sequence,
        )
        windows.extend(sequence_windows[index] for index in keep)

    split_summary = summarize_windows(windows, sequence_splits)
    valid = all(
        split_summary[split]["window_count"] > 0
        for split in ("train", "dev", "test")
    )
    result = {
        "roots": [str(root.resolve()) for root in args.root],
        "objects_requested": args.objects,
        "threshold_m": args.threshold_m,
        "window": args.window,
        "stride": args.stride,
        "min_run": args.min_run,
        "max_windows_per_sequence": args.max_windows_per_sequence,
        "split_seed": args.split_seed,
        "test_fraction": args.test_fraction,
        "selection_rule": (
            "deduplicate by sequence; require contiguous p10<=threshold runs; "
            "enumerate shifted windows; evenly sample at most "
            "max_windows_per_sequence per sequence; assign explicit dev "
            "sequences first, then split remaining sequences by stable hash"
        ),
        "candidate_count": len(candidates),
        "ignored_candidate_counts": ignored,
        "sequence_splits": sequence_splits,
        "split_summary": split_summary,
        "window_count": len(windows),
        "windows": sorted(
            windows,
            key=lambda item: (
                item["split"],
                item["object_name"],
                item["sequence"],
                item["frames"][0],
            ),
        ),
        "rejected": rejected,
        "valid_sequence_disjoint_dataset": valid,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_count": result["candidate_count"],
                "window_count": result["window_count"],
                "split_summary": result["split_summary"],
                "valid_sequence_disjoint_dataset": valid,
                "output_json": str(args.output_json),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
