#!/usr/bin/env python3
"""Build compact OakInk handover event geometry for C0."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from scripts.build_oakink_handover_manifest import (
        load_downsampled_object,
        load_object_id_mapping,
        load_pickle,
        parse_sample_name,
        parse_sequence,
        transform_vertices,
    )
    from scripts.build_oakink_handover_windows import (
        read_candidates,
        select_primary_events,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_oakink_handover_manifest import (  # noqa: E402
        load_downsampled_object,
        load_object_id_mapping,
        load_pickle,
        parse_sample_name,
        parse_sequence,
        transform_vertices,
    )
    from build_oakink_handover_windows import (  # noqa: E402
        read_candidates,
        select_primary_events,
    )


HISTORY = 30
FUTURE = 30
PRIMARY_THRESHOLD_M = 0.005
CONTACT_THRESHOLD_M = 0.005
MINIMUM_WINDOW_FRAMES = 45


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--objects-zip", type=Path, required=True)
    parser.add_argument("--meta-zip", type=Path, required=True)
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-id", default="0")
    parser.add_argument("--history", type=int, default=HISTORY)
    parser.add_argument("--future", type=int, default=FUTURE)
    parser.add_argument("--threshold-m", type=float, default=PRIMARY_THRESHOLD_M)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def event_window_indices(release, length, history, future):
    release = int(release)
    length = int(length)
    indices = np.arange(release - history, release + future + 1)
    valid = (indices >= 0) & (indices < length)
    return np.clip(indices, 0, max(length - 1, 0)), valid


def contact_region(object_vertices, hand_vertices, threshold_m):
    vertices = np.asarray(object_vertices, dtype=np.float64)
    hand = np.asarray(hand_vertices, dtype=np.float64)
    distances = cKDTree(hand).query(vertices, k=1, workers=-1)[0]
    indices = np.flatnonzero(distances <= threshold_m).astype(np.int32)
    return indices, distances[indices].astype(np.float32)


def object_vertex_distances(object_vertices, hand_vertices):
    return cKDTree(hand_vertices).query(
        np.asarray(object_vertices, dtype=np.float64),
        k=1,
        workers=-1,
    )[0].astype(np.float64)


def object_diameter(vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    lower = vertices.min(axis=0)
    upper = vertices.max(axis=0)
    return float(np.linalg.norm(upper - lower))


def min_hand_object_distance(object_vertices, hand_vertices):
    return float(
        cKDTree(object_vertices).query(
            np.asarray(hand_vertices),
            k=1,
            workers=-1,
        )[0].min()
    )


def build_sequence_frames(annotation_names, selected_sequences, camera_id):
    frames = defaultdict(set)
    for name in annotation_names:
        if (
            not name.startswith("anno/hand_v/")
            or not name.endswith(".pkl")
        ):
            continue
        sample = parse_sample_name(name)
        if (
            sample is None
            or sample["camera"] != camera_id
            or sample["subject_flag"] != "0"
            or sample["sequence"] not in selected_sequences
        ):
            continue
        frames[sample["sequence"]].add((
            sample["timestamp"],
            sample["frame"],
        ))
    return {
        sequence: sorted(values, key=lambda row: (row[0], row[1]))
        for sequence, values in frames.items()
    }


def event_stem(sequence, timestamp, frame, flag, camera_id):
    return (
        f"{sequence}__{timestamp}__{flag}__"
        f"{frame}__{camera_id}"
    )


def load_hand(archive, kind, stem, flag):
    return np.asarray(
        load_pickle(archive, f"anno/{kind}/{stem.format(flag=flag)}.pkl"),
        dtype=np.float32,
    )


def load_object_transform(archive, stem):
    return np.asarray(
        load_pickle(
            archive,
            f"anno/obj_transf/{stem.format(flag='0')}.pkl",
        ),
        dtype=np.float32,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import zipfile

    trajectories = np.load(args.trajectories, allow_pickle=False)
    trajectory_index = {
        str(sequence): index
        for index, sequence in enumerate(trajectories["sequences"])
    }
    primary_events = select_primary_events(
        read_candidates(args.candidates, args.threshold_m)
    )
    if args.limit:
        selected = sorted(primary_events)[:args.limit]
        primary_events = {
            sequence: primary_events[sequence] for sequence in selected
        }

    missing = sorted(set(primary_events) - set(trajectory_index))
    if missing:
        raise RuntimeError(f"events missing trajectories: {missing}")

    with zipfile.ZipFile(args.meta_zip) as meta_zip:
        object_mapping = load_object_id_mapping(meta_zip)
    with zipfile.ZipFile(args.annotations_zip) as annotations_zip:
        annotation_names = annotations_zip.namelist()
        sequence_frames = build_sequence_frames(
            annotation_names,
            set(primary_events),
            args.camera_id,
        )
        with zipfile.ZipFile(args.objects_zip) as objects_zip:
            object_cache = {}
            for sequence in sorted(primary_events):
                parsed = parse_sequence(sequence)
                object_id = parsed["object_id"]
                object_name = object_mapping[object_id]["name"]
                object_cache[object_id] = {
                    "name": object_name,
                    "vertices": load_downsampled_object(
                        objects_zip,
                        object_name,
                    ),
                }

            sequences = sorted(primary_events)
            count = len(sequences)
            width = args.history + args.future + 1
            giver_vertices = np.full(
                (count, width, 778, 3),
                np.nan,
                dtype=np.float32,
            )
            receiver_vertices = np.full_like(giver_vertices, np.nan)
            giver_joints = np.full(
                (count, width, 21, 3),
                np.nan,
                dtype=np.float32,
            )
            receiver_joints = np.full_like(giver_joints, np.nan)
            object_transforms = np.full(
                (count, width, 4, 4),
                np.nan,
                dtype=np.float32,
            )
            valid = np.zeros((count, width), dtype=bool)
            release_indices = np.zeros(count, dtype=np.int32)
            receiver_starts = np.zeros(count, dtype=np.int32)
            receiver_ends = np.zeros(count, dtype=np.int32)
            giver_distance = np.full((count, width), np.nan, dtype=np.float32)
            receiver_distance = np.full(
                (count, width),
                np.nan,
                dtype=np.float32,
            )
            object_indices = np.zeros(count, dtype=np.int32)
            current_offsets = [0]
            current_indices = []
            current_distances = []
            target_offsets = [0]
            target_indices = []
            target_distances = []
            manifest_rows = []

            object_ids = sorted(object_cache)
            object_id_to_index = {
                object_id: index
                for index, object_id in enumerate(object_ids)
            }

            for event_index, sequence in enumerate(sequences):
                event = primary_events[sequence]
                trajectory_idx = trajectory_index[sequence]
                length = int(trajectories["lengths"][trajectory_idx])
                release = int(event["giver_release"])
                safe_indices, frame_valid = event_window_indices(
                    release,
                    length,
                    args.history,
                    args.future,
                )
                release_indices[event_index] = release
                receiver_starts[event_index] = int(event["receiver_start"])
                receiver_ends[event_index] = int(event["receiver_end"])
                valid[event_index] = frame_valid
                giver_distance[event_index] = (
                    trajectories["giver_distance_m"][trajectory_idx, :length][
                        safe_indices
                    ]
                )
                receiver_distance[event_index] = (
                    trajectories["receiver_distance_m"][
                        trajectory_idx, :length
                    ][safe_indices]
                )

                parsed = parse_sequence(sequence)
                object_id = parsed["object_id"]
                object_indices[event_index] = object_id_to_index[object_id]
                object_vertices = object_cache[object_id]["vertices"]

                valid_sequence_positions = []
                for position, (source_frame, is_valid) in enumerate(
                    zip(safe_indices, frame_valid)
                ):
                    if not is_valid:
                        continue
                    timestamp, frame = sequence_frames[sequence][source_frame]
                    stem = event_stem(
                        sequence,
                        timestamp,
                        frame,
                        "{flag}",
                        args.camera_id,
                    )
                    giver_vertices[event_index, position] = load_hand(
                        annotations_zip,
                        "hand_v",
                        stem,
                        "0",
                    )
                    receiver_vertices[event_index, position] = load_hand(
                        annotations_zip,
                        "hand_v",
                        stem,
                        "1",
                    )
                    giver_joints[event_index, position] = load_hand(
                        annotations_zip,
                        "hand_j",
                        stem,
                        "0",
                    )
                    receiver_joints[event_index, position] = load_hand(
                        annotations_zip,
                        "hand_j",
                        stem,
                        "1",
                    )
                    object_transforms[event_index, position] = (
                        load_object_transform(annotations_zip, stem)
                    )
                    valid_sequence_positions.append(
                        (source_frame, position)
                    )

                current_source = max(0, release - 15)
                current_position = int(
                    np.flatnonzero(safe_indices == current_source)[0]
                )
                current_object_vertices = transform_vertices(
                    object_vertices,
                    object_transforms[event_index, current_position],
                )
                current_region_indices, current_region_distances = contact_region(
                    current_object_vertices,
                    giver_vertices[event_index, current_position],
                    args.threshold_m,
                )
                current_offsets.append(
                    current_offsets[-1] + len(current_region_indices)
                )
                current_indices.extend(
                    current_region_indices.tolist()
                )
                current_distances.extend(
                    current_region_distances.tolist()
                )

                target_mask = np.zeros(len(object_vertices), dtype=bool)
                target_minimum_distance = np.full(
                    len(object_vertices),
                    np.inf,
                    dtype=np.float64,
                )
                for source_frame, position in valid_sequence_positions:
                    if not (
                        int(event["receiver_start"])
                        <= source_frame
                        <= int(event["receiver_end"])
                    ):
                        continue
                    transformed_object = transform_vertices(
                        object_vertices,
                        object_transforms[event_index, position],
                    )
                    distances = object_vertex_distances(
                        transformed_object,
                        receiver_vertices[event_index, position],
                    )
                    target_mask |= distances <= args.threshold_m
                    target_minimum_distance = np.minimum(
                        target_minimum_distance,
                        distances,
                    )
                target_vertices = np.flatnonzero(target_mask).astype(np.int32)
                target_offsets.append(
                    target_offsets[-1] + len(target_vertices)
                )
                target_indices.extend(target_vertices.tolist())
                target_distances.extend(
                    target_minimum_distance[target_vertices].tolist()
                )

                manifest_rows.append({
                    "event_index": event_index,
                    "sequence": sequence,
                    "split": str(trajectories["splits"][trajectory_idx]),
                    "participant_pair": str(
                        trajectories["participant_pairs"][trajectory_idx]
                    ),
                    "object_id": object_id,
                    "object_name": object_cache[object_id]["name"],
                    "object_index": object_id_to_index[object_id],
                    "release": release,
                    "receiver_start": int(event["receiver_start"]),
                    "receiver_end": int(event["receiver_end"]),
                    "onset_minus_release": int(
                        event["onset_minus_release"]
                    ),
                    "valid_frames": int(frame_valid.sum()),
                })
                print(
                    f"[{event_index + 1}/{count}] {sequence} "
                    f"release={release} valid={int(frame_valid.sum())}/{width}",
                    flush=True,
                )

    objects_path = args.output_dir / "objects.npz"
    object_payload = {
        "object_ids": np.asarray(object_ids),
        "object_names": np.asarray([
            object_cache[object_id]["name"] for object_id in object_ids
        ]),
        "diameters": np.asarray([
            object_diameter(object_cache[object_id]["vertices"])
            for object_id in object_ids
        ], dtype=np.float32),
    }
    for index, object_id in enumerate(object_ids):
        object_payload[f"vertices_{index}"] = np.asarray(
            object_cache[object_id]["vertices"],
            dtype=np.float32,
        )
    np.savez_compressed(objects_path, **object_payload)
    np.savez_compressed(
        args.output_dir / "events.npz",
        sequences=np.asarray(sequences),
        splits=np.asarray([
            row["split"] for row in manifest_rows
        ]),
        participant_pairs=np.asarray([
            row["participant_pair"] for row in manifest_rows
        ]),
        object_ids=np.asarray([
            row["object_id"] for row in manifest_rows
        ]),
        object_names=np.asarray([
            row["object_name"] for row in manifest_rows
        ]),
        object_indices=object_indices,
        release_indices=release_indices,
        receiver_starts=receiver_starts,
        receiver_ends=receiver_ends,
        valid=valid,
        giver_distance_m=giver_distance,
        receiver_distance_m=receiver_distance,
        giver_hand_vertices=giver_vertices,
        receiver_hand_vertices=receiver_vertices,
        giver_hand_joints=giver_joints,
        receiver_hand_joints=receiver_joints,
        object_transforms=object_transforms,
    )
    np.savez_compressed(
        args.output_dir / "targets.npz",
        current_offsets=np.asarray(current_offsets, dtype=np.int64),
        current_indices=np.asarray(current_indices, dtype=np.int32),
        current_distances=np.asarray(current_distances, dtype=np.float32),
        target_offsets=np.asarray(target_offsets, dtype=np.int64),
        target_indices=np.asarray(target_indices, dtype=np.int32),
        target_distances=np.asarray(target_distances, dtype=np.float32),
    )
    with (args.output_dir / "manifest.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    split_counts = {
        split: int(sum(row["split"] == split for row in manifest_rows))
        for split in ("train", "val", "test")
    }
    summary = {
        "event_count": count,
        "split_counts": split_counts,
        "history": args.history,
        "future": args.future,
        "width": width,
        "minimum_window_frames": MINIMUM_WINDOW_FRAMES,
        "threshold_m": args.threshold_m,
        "camera_id": args.camera_id,
        "objects": len(object_ids),
        "giver_vertices_shape": list(giver_vertices.shape),
        "receiver_vertices_shape": list(receiver_vertices.shape),
        "manifest_sha256": __import__("hashlib").sha256(
            (args.output_dir / "manifest.jsonl").read_bytes()
        ).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
