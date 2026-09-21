#!/usr/bin/env python3
"""Refine ARCTIC role-switch candidates into quality tiers."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.audit_arctic_handover_gate import (
        binary_segments,
        stable_contact,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from audit_arctic_handover_gate import (  # noqa: E402
        binary_segments,
        stable_contact,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-contact-frames", type=int, default=15)
    parser.add_argument("--tier-a-offset", type=int, default=5)
    parser.add_argument("--tier-b-offset", type=int, default=10)
    parser.add_argument("--tier-a-overlap", type=int, default=10)
    parser.add_argument("--tier-b-overlap", type=int, default=20)
    parser.add_argument("--outgoing-gap-frames", type=int, default=15)
    return parser.parse_args()


def read_candidates(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def classify_tier(row, args):
    offset = abs(int(row["onset_minus_release"]))
    overlap = int(row["overlap_before_release"])
    if (
        offset <= args.tier_a_offset
        and overlap <= args.tier_a_overlap
        and row["outgoing_gap"] >= args.outgoing_gap_frames
        and row["receiving_stable_after_release"]
        >= args.minimum_contact_frames
    ):
        return "tier_a"
    if (
        offset <= args.tier_b_offset
        and overlap <= args.tier_b_overlap
        and row["outgoing_gap"] >= args.outgoing_gap_frames
        and row["receiving_stable_after_release"]
        >= args.minimum_contact_frames
    ):
        return "tier_b"
    return "raw"


def load_trajectories(path):
    arrays = np.load(path, allow_pickle=True)
    result = {}
    for index, key in enumerate(arrays["keys"]):
        length = int(arrays["lengths"][index])
        result[str(key)] = {
            "right": arrays["right_distance_m"][index, :length],
            "left": arrays["left_distance_m"][index, :length],
            "length": length,
        }
    return result


def sequence_masks(record, threshold_m, minimum_frames):
    right = np.isfinite(record["right"]) & (
        record["right"] <= threshold_m
    )
    left = np.isfinite(record["left"]) & (
        record["left"] <= threshold_m
    )
    return (
        stable_contact(right, minimum_frames, 0),
        stable_contact(left, minimum_frames, 0),
    )


def enrich_candidate(row, trajectory, threshold_m, minimum_frames):
    right_stable, left_stable = sequence_masks(
        trajectory,
        threshold_m,
        minimum_frames,
    )
    if row["outgoing_hand"] == "right":
        outgoing = right_stable
        receiving = left_stable
    else:
        outgoing = left_stable
        receiving = right_stable
    release = int(row["outgoing_release"])
    receiving_start = int(row["receiving_start"])
    receiving_end = int(row["receiving_end"])
    length = trajectory["length"]
    overlap_start = max(0, release - 15)
    overlap_before_release = int(
        (right_stable & left_stable)[overlap_start:release].sum()
    )
    outgoing_segments = binary_segments(outgoing)
    next_starts = [
        start for start, _ in outgoing_segments if start > release
    ]
    next_outgoing_start = min(next_starts) if next_starts else length
    outgoing_gap = int(next_outgoing_start - release)
    after_end = min(length, release + minimum_frames)
    receiving_stable_after_release = int(
        receiving[release:after_end].sum()
    )
    return {
        **row,
        "outgoing_duration": int(
            release - max(
                start for start, end in outgoing_segments
                if start <= release <= end
            )
        ),
        "receiving_duration": int(receiving_end - receiving_start),
        "overlap_before_release": overlap_before_release,
        "outgoing_gap": outgoing_gap,
        "receiving_stable_after_release": (
            receiving_stable_after_release
        ),
    }


def split_summary(rows, split):
    selected = [row for row in rows if row["split"] == split]
    return {
        "candidate_count": len(selected),
        "candidate_sequences": len({
            row["sequence"] for row in selected
        }),
        "candidate_participants": len({
            row["participant_id"] for row in selected
        }),
        "candidate_objects": sorted({
            row["object_name"] for row in selected
        }),
        "candidate_directions": {
            direction: sum(
                row["direction"] == direction
                for row in selected
            )
            for direction in ("left_to_right", "right_to_left")
        },
    }


def tier_gate(tier_a_by_threshold):
    primary = tier_a_by_threshold["0.0030"]
    train = primary["train"]
    val = primary["val"]
    checks = {
        "train_candidates_ge_10": train["candidate_count"] >= 10,
        "train_participants_ge_3": (
            train["candidate_participants"] >= 3
        ),
        "train_both_directions_ge_2": all(
            count >= 2
            for count in train["candidate_directions"].values()
        ),
        "train_objects_ge_3": len(train["candidate_objects"]) >= 3,
        "val_candidates_ge_1": val["candidate_count"] >= 1,
    }
    checks["pilot_tier_a_overall"] = all(checks.values())
    checks["official_test_available"] = False
    checks["final_benchmark_ready"] = False
    return checks


def write_jsonl_gzip(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = load_trajectories(args.trajectories)
    raw_rows = read_candidates(args.candidates)
    enriched = []
    for row in raw_rows:
        sequence = row["sequence"]
        if sequence not in trajectories:
            raise KeyError(f"missing trajectory for {sequence}")
        result = enrich_candidate(
            row,
            trajectories[sequence],
            float(row["threshold_m"]),
            args.minimum_contact_frames,
        )
        result["tier"] = classify_tier(result, args)
        enriched.append(result)

    by_threshold_tier = defaultdict(lambda: defaultdict(list))
    for row in enriched:
        threshold = f"{float(row['threshold_m']):.4f}"
        by_threshold_tier[threshold][row["tier"]].append(row)

    summary = {}
    tier_rows = defaultdict(list)
    for threshold, tier_rows_for_threshold in sorted(
        by_threshold_tier.items()
    ):
        summary[threshold] = {}
        for tier, rows in tier_rows_for_threshold.items():
            summary[threshold][tier] = {
                split: split_summary(rows, split)
                for split in ("train", "val")
            }
            for row in rows:
                if row["tier"] in ("tier_a", "tier_b"):
                    tier_rows[row["tier"]].append(row)

    result = {
        "config": {
            "minimum_contact_frames": args.minimum_contact_frames,
            "tier_a_offset": args.tier_a_offset,
            "tier_b_offset": args.tier_b_offset,
            "tier_a_overlap": args.tier_a_overlap,
            "tier_b_overlap": args.tier_b_overlap,
            "outgoing_gap_frames": args.outgoing_gap_frames,
        },
        "summary": summary,
        "gate": tier_gate(summary),
        "counts": {
            tier: len(rows) for tier, rows in tier_rows.items()
        },
    }
    (args.output_dir / "candidate_quality_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl_gzip(
        args.output_dir / "tier_a_candidates.jsonl.gz",
        tier_rows["tier_a"],
    )
    write_jsonl_gzip(
        args.output_dir / "tier_b_candidates.jsonl.gz",
        tier_rows["tier_b"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
