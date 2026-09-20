#!/usr/bin/env python3
"""Audit EPIC-Contact schemas and event continuity."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


CONTACT_THRESHOLD_M = 3e-3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pickle", type=Path, required=True)
    parser.add_argument("--keys-pickle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--contact-threshold-m",
        type=float,
        default=CONTACT_THRESHOLD_M,
    )
    return parser.parse_args()


def contact_state(
    distance,
    valid,
    threshold_m,
):
    if not bool(np.asarray(valid).reshape(-1)[0]):
        return False, None
    values = np.asarray(distance, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not values.size:
        return False, None
    minimum = float(values.min())
    return bool(minimum <= threshold_m), minimum


def frame_gap_summary(frames):
    frames = np.asarray(sorted(frames), dtype=np.int64)
    if len(frames) < 2:
        return {
            "frame_count": len(frames),
            "span": 0,
            "gap_1_count": 0,
            "gap_gt_1_count": 0,
            "max_gap": 0,
        }
    gaps = np.diff(frames)
    return {
        "frame_count": len(frames),
        "span": int(frames[-1] - frames[0]),
        "gap_1_count": int(np.sum(gaps == 1)),
        "gap_gt_1_count": int(np.sum(gaps > 1)),
        "max_gap": int(gaps.max()),
    }


def contact_run_sequence(rows):
    ordered = sorted(rows, key=lambda row: row["frame"])
    codes = []
    for row in ordered:
        left = row["left_contact"]
        right = row["right_contact"]
        if left and right:
            code = "B"
        elif left:
            code = "L"
        elif right:
            code = "R"
        else:
            code = "0"
        codes.append((row["frame"], code))
    runs = []
    left_to_right_direct = 0
    right_to_left_direct = 0
    for frame, code in codes:
        if runs and runs[-1]["code"] == code:
            runs[-1]["end_frame"] = frame
            runs[-1]["length"] += 1
        else:
            runs.append({
                "code": code,
                "start_frame": frame,
                "end_frame": frame,
                "length": 1,
            })
    for previous, current in zip(runs, runs[1:]):
        if previous["code"] == "L" and current["code"] == "R":
            left_to_right_direct += 1
        if previous["code"] == "R" and current["code"] == "L":
            right_to_left_direct += 1
    return {
        "runs": runs,
        "left_to_right_direct_count": int(left_to_right_direct),
        "right_to_left_direct_count": int(right_to_left_direct),
    }


def contact_transition_counts(rows, hand):
    ordered = sorted(rows, key=lambda row: row["frame"])
    starts = int(bool(ordered[0][f"{hand}_contact"]))
    ends = 0
    previous = None
    for row in ordered:
        current = row[f"{hand}_contact"]
        if previous is not None:
            previous_contact = previous[f"{hand}_contact"]
            if current and not previous_contact:
                starts += 1
            if previous_contact and not current:
                ends += 1
        previous = row
    if ordered and ordered[-1][f"{hand}_contact"]:
        ends += 1
    return {
        "start_count": int(starts),
        "end_count": int(ends),
    }


def parse_clip_metadata(key, sample):
    prefix = key.split("_frame_")[0]
    parts = prefix.split("_")
    if len(parts) < 6:
        raise ValueError(f"unexpected EPIC key: {key}")
    return {
        "video_id": str(sample["video_id"]),
        "clip_id": "_".join(parts[:4]),
        "hand": parts[4],
        "object_name": "_".join(parts[5:]),
    }


def summarize_keys(keys):
    clips = {}
    videos = set()
    for key in keys:
        prefix = str(key).split("_frame_")[0]
        parts = prefix.split("_")
        if len(parts) < 6:
            continue
        video_id = "_".join(parts[:2])
        clip_id = "_".join(parts[:4])
        videos.add(video_id)
        clips.setdefault(clip_id, []).append(key)
    return {
        "key_count": len(keys),
        "video_count": len(videos),
        "clip_count": len(clips),
        "clip_length_counts": dict(sorted(Counter(
            len(values) for values in clips.values()
        ).items())),
    }


def summarize(data):
    rows = []
    sample_keys = set()
    object_classes = Counter()
    for key, sample in data.items():
        if not isinstance(sample, dict):
            continue
        sample_keys.update(sample.keys())
        object_classes[str(sample.get("obj_class"))] += 1
        left_contact, left_min = contact_state(
            sample["dist.lo"],
            sample["left_valid"],
            CONTACT_THRESHOLD_M,
        )
        right_contact, right_min = contact_state(
            sample["dist.ro"],
            sample["right_valid"],
            CONTACT_THRESHOLD_M,
        )
        metadata = parse_clip_metadata(key, sample)
        rows.append({
            "key": key,
            **metadata,
            "frame": int(np.asarray(sample["frame_num"]).reshape(-1)[0]),
            "left_valid": bool(
                np.asarray(sample["left_valid"]).reshape(-1)[0]
            ),
            "right_valid": bool(
                np.asarray(sample["right_valid"]).reshape(-1)[0]
            ),
            "left_contact": left_contact,
            "right_contact": right_contact,
            "left_min_m": left_min,
            "right_min_m": right_min,
        })

    videos = {}
    clips = {}
    for row in rows:
        videos.setdefault(row["video_id"], []).append(row)
        clips.setdefault(row["clip_id"], []).append(row)

    merged_frames = {}
    for row in rows:
        frame_key = (row["video_id"], row["frame"])
        entry = merged_frames.setdefault(frame_key, {
            "video_id": row["video_id"],
            "frame": row["frame"],
            "left_valid": False,
            "right_valid": False,
            "left_contact": False,
            "right_contact": False,
            "left_objects": set(),
            "right_objects": set(),
        })
        if row["hand"] == "left" and row["left_valid"]:
            entry["left_valid"] = True
            entry["left_contact"] = (
                entry["left_contact"] or row["left_contact"]
            )
            entry["left_objects"].add(row["object_name"])
        if row["hand"] == "right" and row["right_valid"]:
            entry["right_valid"] = True
            entry["right_contact"] = (
                entry["right_contact"] or row["right_contact"]
            )
            entry["right_objects"].add(row["object_name"])
    merged = list(merged_frames.values())
    for row in merged:
        row["same_object"] = bool(
            row["left_objects"] & row["right_objects"]
        )
        row["same_object_bimanual"] = bool(
            row["left_valid"]
            and row["right_valid"]
            and row["same_object"]
        )
        row["same_object_both_contact"] = bool(
            row["same_object_bimanual"]
            and row["left_contact"]
            and row["right_contact"]
        )
    video_summaries = {}
    total_gap_1 = 0
    total_gap_gt_1 = 0
    total_max_gap = 0
    total_starts = 0
    total_ends = 0
    for video_id, video_rows in sorted(videos.items()):
        gaps = frame_gap_summary(
            [row["frame"] for row in video_rows]
        )
        total_gap_1 += gaps["gap_1_count"]
        total_gap_gt_1 += gaps["gap_gt_1_count"]
        total_max_gap = max(total_max_gap, gaps["max_gap"])
        video_summaries[video_id] = gaps

    clip_lengths = Counter()
    clip_hand_counts = Counter()
    cross_hand_clips = []
    for clip_id, clip_rows in clips.items():
        clip_lengths[len(clip_rows)] += 1
        valid_hands = tuple(sorted({
            row["hand"]
            for row in clip_rows
            if row["left_valid"] or row["right_valid"]
        }))
        clip_hand_counts[valid_hands] += 1
        if valid_hands == ("left", "right"):
            cross_hand_clips.append({
                "clip_id": clip_id,
                "video_id": clip_rows[0]["video_id"],
                "object_name": clip_rows[0]["object_name"],
                "frames": [
                    {
                        "frame": row["frame"],
                        "hand": row["hand"],
                        "left_valid": row["left_valid"],
                        "right_valid": row["right_valid"],
                        "left_contact": row["left_contact"],
                        "right_contact": row["right_contact"],
                    }
                    for row in sorted(
                        clip_rows,
                        key=lambda row: row["frame"],
                    )
                ],
            })
        for hand in ("left", "right"):
            transitions = contact_transition_counts(clip_rows, hand)
            total_starts += transitions["start_count"]
            total_ends += transitions["end_count"]

    valid_left = [row for row in rows if row["left_valid"]]
    valid_right = [row for row in rows if row["right_valid"]]
    bimanual_groups = {}
    for row in merged:
        if not row["same_object_bimanual"]:
            continue
        object_name = "+".join(
            sorted(row["left_objects"] & row["right_objects"])
        )
        bimanual_groups.setdefault(
            (row["video_id"], object_name),
            [],
        ).append(row)
    bimanual_group_rows = []
    bimanual_contact_start_count = 0
    bimanual_contact_end_count = 0
    for (video_id, object_name), group_rows in sorted(
        bimanual_groups.items()
    ):
        ordered = sorted(group_rows, key=lambda row: row["frame"])
        gaps = frame_gap_summary([
            row["frame"] for row in ordered
        ])
        starts = int(bool(
            ordered[0]["same_object_both_contact"]
        ))
        ends = 0
        previous = None
        for row in ordered:
            current = row["same_object_both_contact"]
            if previous is not None:
                previous_contact = previous["same_object_both_contact"]
                if current and not previous_contact:
                    starts += 1
                if previous_contact and not current:
                    ends += 1
            previous = row
        if ordered[-1]["same_object_both_contact"]:
            ends += 1
        bimanual_contact_start_count += starts
        bimanual_contact_end_count += ends
        contact_sequence = contact_run_sequence(ordered)
        bimanual_group_rows.append({
            "video_id": video_id,
            "object_name": object_name,
            **gaps,
            "both_contact_frame_count": int(sum(
                row["same_object_both_contact"]
                for row in ordered
            )),
            "both_contact_start_count": starts,
            "both_contact_end_count": ends,
            **contact_sequence,
        })
    return {
        "frame_count": len(rows),
        "merged_frame_count": len(merged),
        "video_count": len(videos),
        "clip_count": len(clips),
        "cross_hand_clip_count": len(cross_hand_clips),
        "cross_hand_clips": cross_hand_clips,
        "clip_length_counts": {
            str(key): value
            for key, value in sorted(clip_lengths.items())
        },
        "clip_valid_hand_counts": {
            "+".join(key) if key else "none": value
            for key, value in sorted(clip_hand_counts.items())
        },
        "sample_keys": sorted(sample_keys),
        "object_class_counts": dict(sorted(object_classes.items())),
        "valid": {
            "row_left_fraction": float(np.mean(
                [row["left_valid"] for row in rows]
            )),
            "row_right_fraction": float(np.mean(
                [row["right_valid"] for row in rows]
            )),
            "left_fraction": float(np.mean(
                [row["left_valid"] for row in merged]
            )),
            "right_fraction": float(np.mean(
                [row["right_valid"] for row in merged]
            )),
            "both_fraction": float(np.mean(
                [row["left_valid"] and row["right_valid"] for row in merged]
            )),
            "same_object_both_fraction": float(np.mean(
                [row["same_object_bimanual"] for row in merged]
            )),
        },
        "contact_at_3mm": {
            "left_fraction_all": float(np.mean(
                [row["left_contact"] for row in rows]
            )),
            "right_fraction_all": float(np.mean(
                [row["right_contact"] for row in rows]
            )),
            "left_fraction_valid": float(np.mean(
                [row["left_contact"] for row in valid_left]
            )) if valid_left else None,
            "right_fraction_valid": float(np.mean(
                [row["right_contact"] for row in valid_right]
            )) if valid_right else None,
            "either_fraction": float(np.mean(
                [row["left_contact"] or row["right_contact"] for row in rows]
            )),
            "both_fraction": float(np.mean(
                [
                    row["left_contact"] and row["right_contact"]
                    for row in merged
                ]
            )),
            "same_object_both_fraction": float(np.mean(
                [
                    row["same_object_both_contact"]
                    for row in merged
                ]
            )),
        },
        "same_object_bimanual": {
            "frame_count": int(sum(
                row["same_object_bimanual"] for row in merged
            )),
            "both_contact_frame_count": int(sum(
                row["same_object_both_contact"] for row in merged
            )),
            "group_count": len(bimanual_group_rows),
            "both_contact_start_count": int(
                bimanual_contact_start_count
            ),
            "both_contact_end_count": int(
                bimanual_contact_end_count
            ),
            "direct_role_switch_count": int(sum(
                row["left_to_right_direct_count"]
                + row["right_to_left_direct_count"]
                for row in bimanual_group_rows
            )),
            "groups": bimanual_group_rows,
        },
        "continuity": {
            "adjacent_frame_pairs": int(total_gap_1),
            "nonadjacent_frame_pairs": int(total_gap_gt_1),
            "max_frame_gap": total_max_gap,
            "contact_start_count": int(total_starts),
            "contact_end_count": int(total_ends),
        },
        "videos": video_summaries,
    }


def main():
    args = parse_args()
    global CONTACT_THRESHOLD_M
    CONTACT_THRESHOLD_M = args.contact_threshold_m
    with args.test_pickle.open("rb") as handle:
        data = pickle.load(handle)
    with args.keys_pickle.open("rb") as handle:
        all_keys = pickle.load(handle)
    summary = summarize(data)
    summary["full_key_count"] = len(all_keys)
    summary["keys_summary"] = summarize_keys(all_keys)
    summary["test_key_count"] = len(data)
    summary["contact_threshold_m"] = CONTACT_THRESHOLD_M
    summary["decision"] = {
        "contact_schema": "pass",
        "contact_memory_b_gate": (
            "candidate_ready_pending_sequence_gap_review"
            if summary["continuity"]["contact_start_count"] > 0
            and summary["continuity"]["contact_end_count"] > 0
            else "blocked_no_transitions"
        ),
        "future_handover_c_gate": (
            "candidate_ready_pending_manual_validation"
            if summary["same_object_bimanual"][
                "direct_role_switch_count"
            ] > 0
            else (
                "not_supported_no_direct_role_switch"
                if summary["same_object_bimanual"]["frame_count"] > 0
                else "not_supported_no_bimanual_frames"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: summary[key]
        for key in (
            "full_key_count",
            "test_key_count",
            "frame_count",
            "merged_frame_count",
            "video_count",
            "clip_count",
            "cross_hand_clip_count",
            "valid",
            "contact_at_3mm",
            "same_object_bimanual",
            "continuity",
            "decision",
        )
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
