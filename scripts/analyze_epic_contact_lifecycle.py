#!/usr/bin/env python3
"""Analyze lifecycle statistics in the EPIC-Contact B manifest."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gap-frames", type=int, default=1)
    return parser.parse_args()


def load_jsonl_gzip(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {}
    return {
        "min": float(values.min()),
        "p25": float(np.quantile(values, 0.25)),
        "p50": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def episode_summary(episodes):
    summary = {}
    complete = []
    for split in ("train", "test"):
        rows = [row for row in episodes if row["split"] == split]
        complete_rows = [
            row for row in rows
            if row["onset_observed"] and row["release_observed"]
        ]
        complete.extend(complete_rows)
        summary[split] = {
            "episode_count": len(rows),
            "complete_count": len(complete_rows),
            "onset_only_count": int(sum(
                row["onset_observed"] and not row["release_observed"]
                for row in rows
            )),
            "release_only_count": int(sum(
                not row["onset_observed"] and row["release_observed"]
                for row in rows
            )),
            "truncated_or_unbounded_count": int(sum(
                not (
                    row["onset_observed"]
                    and row["release_observed"]
                )
                for row in rows
            )),
            "by_hand": dict(Counter(row["hand"] for row in rows)),
            "by_object": dict(sorted(Counter(
                row["object_name"] for row in rows
            ).items())),
            "contact_frame_count": quantiles([
                row["contact_frame_count"] for row in rows
            ]),
            "complete_contact_frame_count": quantiles([
                row["contact_frame_count"] for row in complete_rows
            ]),
        }
    summary["all"] = {
        "episode_count": len(episodes),
        "complete_count": len(complete),
    }
    return summary


def transition_summary(frames, max_gap_frames):
    groups = defaultdict(list)
    for row in frames:
        for hand in ("left", "right"):
            hand_row = row[hand]
            if not hand_row["valid"] or hand_row["clip_id"] is None:
                continue
            key = (
                row["split"],
                row["video_id"],
                hand_row["clip_id"],
                hand,
                hand_row["object_name"],
            )
            groups[key].append((
                row["frame"],
                bool(hand_row["contact"]),
            ))

    result = {
        split: {
            "valid_hand_frames": 0,
            "adjacent_pairs": 0,
            "contact_to_contact": 0,
            "contact_to_noncontact": 0,
            "noncontact_to_contact": 0,
            "noncontact_to_noncontact": 0,
        }
        for split in ("train", "test")
    }
    for key, values in groups.items():
        split = key[0]
        ordered = sorted(values)
        result[split]["valid_hand_frames"] += len(ordered)
        for (left_frame, left_contact), (
            right_frame,
            right_contact,
        ) in zip(ordered, ordered[1:]):
            if right_frame - left_frame > max_gap_frames:
                continue
            result[split]["adjacent_pairs"] += 1
            if left_contact and right_contact:
                result[split]["contact_to_contact"] += 1
            elif left_contact and not right_contact:
                result[split]["contact_to_noncontact"] += 1
            elif not left_contact and right_contact:
                result[split]["noncontact_to_contact"] += 1
            else:
                result[split]["noncontact_to_noncontact"] += 1
    return result


def contact_observations(frames):
    groups = defaultdict(list)
    for row in frames:
        for hand in ("left", "right"):
            hand_row = row[hand]
            if (
                not hand_row["valid"]
                or hand_row["clip_id"] is None
                or hand_row["min_distance_m"] is None
            ):
                continue
            key = (
                row["split"],
                row["video_id"],
                hand_row["clip_id"],
                hand,
                hand_row["object_name"],
            )
            groups[key].append((
                row["frame"],
                float(hand_row["min_distance_m"]),
            ))
    return groups


def threshold_sensitivity(
    frames,
    thresholds,
    max_gap_frames,
):
    groups = contact_observations(frames)
    result = {}
    for threshold in thresholds:
        split_result = {
            split: {
                "hold": 0,
                "release": 0,
                "onset": 0,
                "noncontact": 0,
                "complete_episode": 0,
                "onset_only_episode": 0,
                "release_only_episode": 0,
                "truncated_episode": 0,
            }
            for split in ("train", "test")
        }
        for key, raw_values in groups.items():
            split = key[0]
            values = [
                (frame, distance <= threshold)
                for frame, distance in sorted(raw_values)
            ]
            segments = []
            segment = []
            for item in values:
                if (
                    segment
                    and item[0] - segment[-1][0] > max_gap_frames
                ):
                    segments.append(segment)
                    segment = []
                segment.append(item)
            if segment:
                segments.append(segment)

            for contiguous in segments:
                for (left_frame, left_contact), (
                    right_frame,
                    right_contact,
                ) in zip(contiguous, contiguous[1:]):
                    if right_frame - left_frame > max_gap_frames:
                        continue
                    if left_contact and right_contact:
                        split_result[split]["hold"] += 1
                    elif left_contact and not right_contact:
                        split_result[split]["release"] += 1
                    elif not left_contact and right_contact:
                        split_result[split]["onset"] += 1
                    else:
                        split_result[split]["noncontact"] += 1

                index = 0
                while index < len(contiguous):
                    if not contiguous[index][1]:
                        index += 1
                        continue
                    end = index
                    while (
                        end + 1 < len(contiguous)
                        and contiguous[end + 1][1]
                    ):
                        end += 1
                    onset_observed = (
                        index > 0 and not contiguous[index - 1][1]
                    )
                    release_observed = (
                        end + 1 < len(contiguous)
                        and not contiguous[end + 1][1]
                    )
                    if onset_observed and release_observed:
                        key_name = "complete_episode"
                    elif onset_observed:
                        key_name = "onset_only_episode"
                    elif release_observed:
                        key_name = "release_only_episode"
                    else:
                        key_name = "truncated_episode"
                    split_result[split][key_name] += 1
                    index = end + 1
        result[f"{threshold:.6f}"] = split_result
    return result


def contiguous_runs(frames, max_gap_frames):
    ordered = sorted(frames)
    runs = []
    current = []
    for frame in ordered:
        if (
            current
            and frame - current[-1] > max_gap_frames
        ):
            runs.append(current)
            current = []
        current.append(frame)
    if current:
        runs.append(current)
    return runs


def bimanual_summary(frames, max_gap_frames):
    groups = defaultdict(list)
    for row in frames:
        if not row["same_object_bimanual"]:
            continue
        object_name = (
            row["left"]["object_name"]
            if row["left"]["valid"]
            else row["right"]["object_name"]
        )
        groups[(row["split"], row["video_id"], object_name)].append(
            row["frame"]
        )
    result = {}
    for split in ("train", "test"):
        selected = {
            key: value
            for key, value in groups.items()
            if key[0] == split
        }
        run_lengths = []
        both_contact_frames = 0
        for key, values in selected.items():
            runs = contiguous_runs(values, max_gap_frames)
            run_lengths.extend(len(run) for run in runs)
        for row in frames:
            if row["split"] == split and row["same_object_both_contact"]:
                both_contact_frames += 1
        result[split] = {
            "group_count": len(selected),
            "frame_count": int(sum(len(value) for value in selected.values())),
            "both_contact_frame_count": both_contact_frames,
            "run_count": len(run_lengths),
            "run_length": quantiles(run_lengths),
        }
    return result


def video_split_summary(frames):
    by_split = {
        split: {
            row["video_id"] for row in frames
            if row["split"] == split
        }
        for split in ("train", "test")
    }
    overlap = sorted(by_split["train"] & by_split["test"])
    return {
        "train_video_count": len(by_split["train"]),
        "test_video_count": len(by_split["test"]),
        "overlap_video_count": len(overlap),
        "overlap_videos": overlap,
        "video_disjoint": not overlap,
    }


def build_decision(
    episodes,
    transitions,
    bimanual,
    split,
    sensitivity,
):
    complete = episodes["all"]["complete_count"]
    train = transitions["train"]
    result = {
        "hold_model_labels_ready": (
            train["contact_to_contact"] >= 1000
        ),
        "onset_model_labels_ready": (
            train["noncontact_to_contact"] >= 100
        ),
        "release_model_labels_ready": (
            train["contact_to_noncontact"] >= 100
            and complete >= 50
        ),
        "official_split_video_disjoint": split["video_disjoint"],
        "bimanual_hold_analysis_ready": (
            bimanual["train"]["group_count"] >= 20
            and bimanual["train"]["frame_count"] >= 1000
        ),
        "handover_analysis_ready": False,
    }
    result["B_full_model_ready_at_official_3mm"] = all((
        result["hold_model_labels_ready"],
        result["onset_model_labels_ready"],
        result["release_model_labels_ready"],
        result["official_split_video_disjoint"],
    ))
    strict = sensitivity.get("0.001000", {})
    strict_train = strict.get("train", {})
    strict_test = strict.get("test", {})
    result["strict_contact_1mm_candidate"] = {
        "train_hold": strict_train.get("hold", 0),
        "train_onset": strict_train.get("onset", 0),
        "train_release": strict_train.get("release", 0),
        "train_complete_episode": strict_train.get(
            "complete_episode",
            0,
        ),
        "test_complete_episode": strict_test.get(
            "complete_episode",
            0,
        ),
        "event_labels_ready": (
            strict_train.get("onset", 0) >= 100
            and strict_train.get("release", 0) >= 100
            and strict_train.get("complete_episode", 0) >= 50
        ),
        "requires_new_protocol_and_video_split": True,
    }
    result["recommended_action"] = (
        "proceed_with_full_B_model"
        if result["B_full_model_ready_at_official_3mm"]
        else (
            "build_video_disjoint_strict_contact_protocol"
            if result["strict_contact_1mm_candidate"][
                "event_labels_ready"
            ]
            else "limit_to_hold_or_bimanual_analysis"
        )
    )
    return result


def main():
    args = parse_args()
    frames = load_jsonl_gzip(args.frames)
    episodes = load_jsonl_gzip(args.episodes)
    episode_stats = episode_summary(episodes)
    transitions = transition_summary(frames, args.max_gap_frames)
    bimanual = bimanual_summary(frames, args.max_gap_frames)
    split = video_split_summary(frames)
    sensitivity = threshold_sensitivity(
        frames,
        (0.0005, 0.001, 0.0015, 0.002, 0.003),
        args.max_gap_frames,
    )
    decision = build_decision(
        episode_stats,
        transitions,
        bimanual,
        split,
        sensitivity,
    )
    result = {
        "frame_count": len(frames),
        "episode_count": len(episodes),
        "max_gap_frames": args.max_gap_frames,
        "episodes": episode_stats,
        "transitions": transitions,
        "bimanual": bimanual,
        "split": split,
        "threshold_sensitivity": sensitivity,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
