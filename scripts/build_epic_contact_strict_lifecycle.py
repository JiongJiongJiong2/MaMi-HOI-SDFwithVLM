#!/usr/bin/env python3
"""Build strict 1 mm lifecycle data with participant-disjoint splits."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-m", type=float, default=0.001)
    parser.add_argument("--max-gap-frames", type=int, default=1)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl_gzip(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def write_jsonl_gzip(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def participant_id(video_id):
    return str(video_id).split("_", 1)[0]


def build_observations(frames):
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
                row["video_id"],
                hand_row["clip_id"],
                hand,
                hand_row["object_name"],
            )
            groups[key].append((
                int(row["frame"]),
                float(hand_row["min_distance_m"]),
            ))
    return groups


def split_contiguous(values, max_gap_frames):
    segments = []
    segment = []
    for item in sorted(values):
        if (
            segment
            and item[0] - segment[-1][0] > max_gap_frames
        ):
            segments.append(segment)
            segment = []
        segment.append(item)
    if segment:
        segments.append(segment)
    return segments


def build_episodes(groups, threshold_m, max_gap_frames):
    episodes = []
    transition_stats = defaultdict(Counter)
    for key, raw_values in groups.items():
        video_id, clip_id, hand, object_name = key
        participant = participant_id(video_id)
        values = [
            (frame, distance <= threshold_m)
            for frame, distance in sorted(raw_values)
        ]
        for contiguous in split_contiguous(values, max_gap_frames):
            for (left_frame, left_contact), (
                right_frame,
                right_contact,
            ) in zip(contiguous, contiguous[1:]):
                if right_frame - left_frame > max_gap_frames:
                    continue
                if left_contact and right_contact:
                    transition_stats[participant]["hold"] += 1
                elif left_contact and not right_contact:
                    transition_stats[participant]["release"] += 1
                elif not left_contact and right_contact:
                    transition_stats[participant]["onset"] += 1
                else:
                    transition_stats[participant]["noncontact"] += 1

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
                episode_rows = contiguous[index : end + 1]
                distances = [
                    distance
                    for frame, _ in episode_rows
                    for distance in [
                        next(
                            value
                            for value in raw_values
                            if value[0] == frame
                        )
                    ]
                ]
                episodes.append({
                    "video_id": video_id,
                    "participant_id": participant,
                    "clip_id": clip_id,
                    "hand": hand,
                    "object_name": object_name,
                    "onset_frame": episode_rows[0][0],
                    "last_contact_frame": episode_rows[-1][0],
                    "contact_frame_count": len(episode_rows),
                    "duration_frames": (
                        episode_rows[-1][0]
                        - episode_rows[0][0]
                        + 1
                    ),
                    "max_gap": max(
                        [
                            right[0] - left[0]
                            for left, right in zip(
                                episode_rows,
                                episode_rows[1:],
                            )
                        ]
                        or [0]
                    ),
                    "onset_observed": bool(onset_observed),
                    "release_observed": bool(release_observed),
                    "mean_distance_m": float(np.mean(distances)),
                })
                index = end + 1
    return episodes, transition_stats


def greedy_participant_split(
    episodes,
    transition_stats,
    fractions,
):
    participants = sorted({
        row["participant_id"] for row in episodes
    } | set(transition_stats))
    episode_counts = Counter(
        row["participant_id"] for row in episodes
    )
    complete_counts = Counter(
        row["participant_id"]
        for row in episodes
        if row["onset_observed"] and row["release_observed"]
    )
    weights = {
        participant: (
            transition_stats[participant]["hold"] * 0.1
            + transition_stats[participant]["onset"] * 2.0
            + transition_stats[participant]["release"] * 2.0
            + complete_counts[participant] * 5.0
        )
        for participant in participants
    }
    split_names = ("train", "dev", "test")
    target_weight = {
        split: sum(weights.values()) * fractions[split]
        for split in split_names
    }
    current_weight = {split: 0.0 for split in split_names}
    assignment = {}
    for participant in sorted(
        participants,
        key=lambda value: (-weights[value], value),
    ):
        candidates = []
        for split in split_names:
            projected = (
                current_weight[split] + weights[participant]
            )
            error = abs(
                projected / max(target_weight[split], 1.0) - 1.0
            )
            candidates.append((error, split))
        _, selected = min(candidates)
        assignment[participant] = selected
        current_weight[selected] += weights[participant]
    return assignment


def assign_frame_splits(frames, participant_to_split, threshold_m):
    output = []
    for row in frames:
        new_row = {
            "split": participant_to_split[
                participant_id(row["video_id"])
            ],
            "official_split": row["split"],
            "video_id": row["video_id"],
            "frame": int(row["frame"]),
            "left": dict(row["left"]),
            "right": dict(row["right"]),
        }
        for hand in ("left", "right"):
            hand_row = new_row[hand]
            hand_row["contact"] = bool(
                hand_row["valid"]
                and hand_row["min_distance_m"] is not None
                and float(hand_row["min_distance_m"]) <= threshold_m
            )
        new_row["same_object_bimanual"] = bool(
            new_row["left"]["valid"]
            and new_row["right"]["valid"]
            and new_row["left"]["object_name"]
            == new_row["right"]["object_name"]
        )
        new_row["same_object_both_contact"] = bool(
            new_row["same_object_bimanual"]
            and new_row["left"]["contact"]
            and new_row["right"]["contact"]
        )
        output.append(new_row)
    return output


def summarize(
    episodes,
    frames,
    participant_to_split,
    transition_stats,
):
    result = {}
    for split in ("train", "dev", "test"):
        split_episodes = [
            row for row in episodes
            if participant_to_split[row["participant_id"]] == split
        ]
        split_frames = [
            row for row in frames if row["split"] == split
        ]
        participants = {
            participant
            for participant, assigned in participant_to_split.items()
            if assigned == split
        }
        transition_totals = Counter()
        for participant in participants:
            transition_totals.update(transition_stats[participant])
        result[split] = {
            "participant_count": len(participants),
            "video_count": len({row["video_id"] for row in split_frames}),
            "frame_count": len(split_frames),
            "episode_count": len(split_episodes),
            "complete_episode_count": int(sum(
                row["onset_observed"] and row["release_observed"]
                for row in split_episodes
            )),
            "onset_episode_count": int(sum(
                row["onset_observed"] for row in split_episodes
            )),
            "release_episode_count": int(sum(
                row["release_observed"] for row in split_episodes
            )),
            "transition_counts": dict(transition_totals),
            "same_object_bimanual_frame_count": int(sum(
                row["same_object_bimanual"] for row in split_frames
            )),
            "same_object_both_contact_frame_count": int(sum(
                row["same_object_both_contact"] for row in split_frames
            )),
        }
    return result


def build_gate(splits, strict_frames):
    by_split = {
        split: {
            row["video_id"]
            for row in strict_frames
            if row["split"] == split
        }
        for split in ("train", "dev", "test")
    }
    video_disjoint = (
        not (by_split["train"] & by_split["dev"])
        and not (by_split["train"] & by_split["test"])
        and not (by_split["dev"] & by_split["test"])
    )
    every_split_has_events = all(
        splits[split]["transition_counts"].get("hold", 0) > 0
        and splits[split]["transition_counts"].get("onset", 0) > 0
        and splits[split]["transition_counts"].get("release", 0) > 0
        for split in ("train", "dev", "test")
    )
    gate = {
        "participant_disjoint": True,
        "video_disjoint": video_disjoint,
        "every_split_has_hold_onset_release": every_split_has_events,
        "train_has_100_onset_and_release": (
            splits["train"]["transition_counts"].get("onset", 0) >= 100
            and splits["train"]["transition_counts"].get(
                "release",
                0,
            ) >= 100
        ),
        "dev_test_have_20_complete_episodes": (
            splits["dev"]["complete_episode_count"] >= 20
            and splits["test"]["complete_episode_count"] >= 20
        ),
    }
    gate["pass"] = all(gate.values())
    return gate


def main():
    args = parse_args()
    fractions = {
        "train": args.train_fraction,
        "dev": args.dev_fraction,
        "test": args.test_fraction,
    }
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to one")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_jsonl_gzip(args.frames)
    observations = build_observations(frames)
    episodes, transition_stats = build_episodes(
        observations,
        args.threshold_m,
        args.max_gap_frames,
    )
    participant_to_split = greedy_participant_split(
        episodes,
        transition_stats,
        fractions,
    )
    for row in episodes:
        row["split"] = participant_to_split[row["participant_id"]]
    strict_frames = assign_frame_splits(
        frames,
        participant_to_split,
        args.threshold_m,
    )
    summary = {
        "threshold_m": args.threshold_m,
        "max_gap_frames": args.max_gap_frames,
        "split_fractions": fractions,
        "participant_to_split": participant_to_split,
        "splits": summarize(
            episodes,
            strict_frames,
            participant_to_split,
            transition_stats,
        ),
    }
    summary["gate"] = build_gate(summary["splits"], strict_frames)
    frame_path = args.output_dir / "epic_contact_strict_frames_v1.jsonl.gz"
    episode_path = (
        args.output_dir / "epic_contact_strict_episodes_v1.jsonl.gz"
    )
    split_path = args.output_dir / "epic_contact_strict_split_v1.json"
    summary_path = args.output_dir / "summary.json"
    write_jsonl_gzip(frame_path, strict_frames)
    write_jsonl_gzip(episode_path, episodes)
    split_path.write_text(
        json.dumps(participant_to_split, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["outputs"] = {
        "frames": {
            "path": str(frame_path),
            "rows": len(strict_frames),
            "sha256": sha256_file(frame_path),
        },
        "episodes": {
            "path": str(episode_path),
            "rows": len(episodes),
            "sha256": sha256_file(episode_path),
        },
        "split": {
            "path": str(split_path),
            "sha256": sha256_file(split_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
