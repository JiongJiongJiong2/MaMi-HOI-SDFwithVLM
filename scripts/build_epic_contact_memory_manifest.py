#!/usr/bin/env python3
"""Build compact EPIC-Contact correspondence shards for lifecycle memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from build_epic_contact_b_manifest import (
    StreamingTopLevelUnpickler,
    file_identity,
    parse_key,
    scalar,
)


CONTACT_THRESHOLD_M = 0.001
ACTION_NAMES = ("hold", "update", "close", "unknown")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pickle", type=Path)
    parser.add_argument("--test-pickle", type=Path)
    parser.add_argument("--strict-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--max-gap-frames", type=int, default=1)
    parser.add_argument("--contact-threshold-m", type=float, default=0.001)
    parser.add_argument("--max-samples-per-source", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.train_pickle and not args.test_pickle:
        raise ValueError("at least one pickle source is required")
    if args.top_k <= 0:
        raise ValueError("top-k must be positive")
    if args.max_gap_frames < 0:
        raise ValueError("max-gap-frames must be non-negative")
    return args


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def participant_id(video_id):
    return str(video_id).split("_", 1)[0]


def as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def as_vector(value):
    return as_numpy(value).reshape(-1)


def normalize_object_vertices(object_vertices, rotation, translation):
    vertices = np.asarray(object_vertices, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("object vertices must have shape [N, 3]")
    if rotation.shape != (3, 3):
        raise ValueError("object rotation must have shape [3, 3]")
    if translation.shape != (3,):
        raise ValueError("object translation must have shape [3]")
    return (vertices - translation[None]) @ rotation


def split_contiguous(frames, max_gap_frames):
    ordered = sorted(int(frame) for frame in frames)
    segments = []
    current = []
    for frame in ordered:
        if (
            current
            and frame - current[-1] > max_gap_frames
        ):
            segments.append(current)
            current = []
        current.append(frame)
    if current:
        segments.append(current)
    return segments


def top_contact_pairs(
    distances,
    object_indices,
    hand_vertices,
    object_vertices,
    top_k,
):
    distances = np.asarray(distances, dtype=np.float64).reshape(-1)
    object_indices = np.asarray(
        object_indices,
        dtype=np.int64,
    ).reshape(-1)
    hand_vertices = np.asarray(hand_vertices, dtype=np.float64)
    object_vertices = np.asarray(object_vertices, dtype=np.float64)
    if (
        distances.shape != object_indices.shape
        or distances.shape[0] != hand_vertices.shape[0]
    ):
        raise ValueError("distance, index, and hand vertex counts differ")
    finite = np.isfinite(distances) & (distances >= 0)
    finite_indices = np.flatnonzero(finite)
    if not len(finite_indices):
        return {
            "hand_indices": np.empty(0, dtype=np.int64),
            "object_indices": np.empty(0, dtype=np.int64),
            "distances_m": np.empty(0, dtype=np.float32),
            "hand_xyz": np.empty((0, 3), dtype=np.float32),
            "object_xyz": np.empty((0, 3), dtype=np.float32),
        }
    count = min(top_k, len(finite_indices))
    order = finite_indices[
        np.argsort(distances[finite_indices], kind="stable")[:count]
    ]
    object_order = object_indices[order]
    if np.any(
        (object_order < 0)
        | (object_order >= len(object_vertices))
    ):
        raise ValueError("object nearest-neighbor index out of range")
    return {
        "hand_indices": order.astype(np.int64),
        "object_indices": object_order.astype(np.int64),
        "distances_m": distances[order].astype(np.float32),
        "hand_xyz": hand_vertices[order].astype(np.float32),
        "object_xyz": object_vertices[object_order].astype(np.float32),
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class ClipAccumulator:
    def __init__(
        self,
        split,
        source_split,
        participant,
        video_id,
        clip_id,
        hand,
        object_name,
        top_k,
    ):
        self.split = split
        self.source_split = source_split
        self.participant = participant
        self.video_id = video_id
        self.clip_id = clip_id
        self.hand = hand
        self.object_name = object_name
        self.top_k = top_k
        self.frames = {}
        self.object_template = None
        self.object_faces = None
        self.object_diameter = None
        self.object_vertex_count = None

    def add(self, sample, contact_threshold_m):
        hand = self.hand
        valid_name = "left_valid" if hand == "left" else "right_valid"
        valid = bool(scalar(sample[valid_name]))
        if not valid:
            return
        frame = int(scalar(sample["frame_num"]))
        distance_name = "dist.lo" if hand == "left" else "dist.ro"
        index_name = "idx.lo" if hand == "left" else "idx.ro"
        vertex_name = (
            "mano.v3d.cam.l" if hand == "left"
            else "mano.v3d.cam.r"
        )
        pose_name = "mano.pose.l" if hand == "left" else "mano.pose.r"
        beta_name = "mano.beta.l" if hand == "left" else "mano.beta.r"
        translation_name = (
            "mano.cam_t.l" if hand == "left" else "mano.cam_t.r"
        )

        object_vertices = as_numpy(
            sample["object.v.cam"]
        ).reshape(-1, 3).astype(np.float64)
        rotation = as_numpy(
            sample["object.rot"]
        ).reshape(3, 3).astype(np.float64)
        translation = as_vector(
            sample["object.cam_t"]
        ).astype(np.float64)
        canonical = normalize_object_vertices(
            object_vertices,
            rotation,
            translation,
        ).astype(np.float32)
        if self.object_template is None:
            self.object_template = canonical
            self.object_faces = as_numpy(
                sample["object.f"]
            ).reshape(-1, 3).astype(np.int32)
            self.object_diameter = float(
                scalar(sample["object.diameter"])
            )
            self.object_vertex_count = len(canonical)
        else:
            if canonical.shape != self.object_template.shape:
                raise ValueError(
                    f"{self.clip_id} object vertex count changed"
                )
            error = float(
                np.max(np.abs(canonical - self.object_template))
            )
            if error > 1e-4:
                raise ValueError(
                    f"{self.clip_id} object topology is unstable: "
                    f"{error}"
                )

        hand_vertices = as_numpy(
            sample[vertex_name]
        ).reshape(-1, 3).astype(np.float64)
        pairs = top_contact_pairs(
            as_vector(sample[distance_name]),
            as_vector(sample[index_name]),
            hand_vertices,
            object_vertices,
            self.top_k,
        )
        minimum = (
            float(pairs["distances_m"][0])
            if len(pairs["distances_m"])
            else None
        )
        row = {
            "frame": frame,
            "valid": True,
            "contact": bool(
                minimum is not None
                and minimum <= contact_threshold_m
            ),
            "min_distance_m": minimum,
            "mano_pose": as_vector(sample[pose_name]).astype(np.float32),
            "mano_beta": as_vector(sample[beta_name]).astype(np.float32),
            "mano_translation": as_vector(
                sample[translation_name]
            ).astype(np.float32),
            "hand_vertices": hand_vertices.astype(np.float32),
            "object_rotation": rotation.astype(np.float32),
            "object_translation": translation.astype(np.float32),
            "top_hand_indices": pairs["hand_indices"],
            "top_object_indices": pairs["object_indices"],
            "top_distances_m": pairs["distances_m"],
            "top_hand_xyz": pairs["hand_xyz"],
            "top_object_xyz": pairs["object_xyz"],
        }
        if frame in self.frames:
            previous = self.frames[frame]
            if previous["contact"] != row["contact"]:
                raise ValueError(
                    f"{self.clip_id} duplicate frame contact mismatch"
                )
            return
        self.frames[frame] = row

    def segments(self, max_gap_frames):
        segments = split_contiguous(
            self.frames,
            max_gap_frames,
        )
        result = []
        for segment_index, frames in enumerate(segments):
            rows = [self.frames[frame] for frame in frames]
            result.append({
                "segment_index": segment_index,
                "frames": frames,
                "rows": rows,
            })
        return result


def stack_rows(rows, key, dtype):
    return np.asarray([row[key] for row in rows], dtype=dtype)


def save_segment(
    output_dir,
    accumulator,
    segment,
    resume=False,
):
    frames = segment["frames"]
    rows = segment["rows"]
    segment_id = (
        f"{accumulator.clip_id}_{accumulator.hand}_"
        f"{accumulator.object_name}_s{segment['segment_index']:03d}"
    )
    stored_relative = (
        Path("clips")
        / accumulator.split
        / accumulator.participant
        / f"{segment_id}.npz"
    )
    relative = (
        Path(accumulator.split)
        / accumulator.participant
        / f"{segment_id}.npz"
    )
    output_path = output_dir / stored_relative
    if not (resume and output_path.is_file()):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                frames=np.asarray(frames, dtype=np.int64),
                valid=stack_rows(rows, "valid", np.bool_),
                contact=stack_rows(rows, "contact", np.bool_),
                min_distance_m=stack_rows(
                    rows,
                    "min_distance_m",
                    np.float32,
                ),
                mano_pose=stack_rows(rows, "mano_pose", np.float32),
                mano_beta=stack_rows(rows, "mano_beta", np.float32),
                mano_translation=stack_rows(
                    rows,
                    "mano_translation",
                    np.float32,
                ),
                hand_vertices=stack_rows(
                    rows,
                    "hand_vertices",
                    np.float32,
                ),
                object_rotation=stack_rows(
                    rows,
                    "object_rotation",
                    np.float32,
                ),
                object_translation=stack_rows(
                    rows,
                    "object_translation",
                    np.float32,
                ),
                top_hand_indices=stack_rows(
                    rows,
                    "top_hand_indices",
                    np.int64,
                ),
                top_object_indices=stack_rows(
                    rows,
                    "top_object_indices",
                    np.int64,
                ),
                top_distances_m=stack_rows(
                    rows,
                    "top_distances_m",
                    np.float32,
                ),
                top_hand_xyz=stack_rows(
                    rows,
                    "top_hand_xyz",
                    np.float32,
                ),
                top_object_xyz=stack_rows(
                    rows,
                    "top_object_xyz",
                    np.float32,
                ),
                object_vertices=accumulator.object_template,
                object_faces=accumulator.object_faces,
                object_diameter_m=np.asarray(
                    accumulator.object_diameter,
                    dtype=np.float32,
                ),
            )
        temporary.replace(output_path)
    return {
        "split": accumulator.split,
        "source_split": accumulator.source_split,
        "participant_id": accumulator.participant,
        "video_id": accumulator.video_id,
        "clip_id": accumulator.clip_id,
        "hand": accumulator.hand,
        "object_name": accumulator.object_name,
        "segment_index": segment["segment_index"],
        "segment_id": segment_id,
        "frame_count": len(frames),
        "frame_start": int(min(frames)),
        "frame_end": int(max(frames)),
        "path": str(stored_relative),
        "local_path": str(output_path),
        "relative_path": str(stored_relative),
        "sha256": sha256_file(output_path),
    }


def process_source(
    source_split,
    path,
    participant_to_split,
    accumulators,
    args,
):
    processed = 0

    def on_sample(key, sample):
        nonlocal processed
        if (
            args.max_samples_per_source
            and processed >= args.max_samples_per_source
        ):
            raise StopIteration
        metadata = parse_key(key)
        participant = participant_id(metadata["video_id"])
        split = participant_to_split.get(participant)
        if split is None:
            raise ValueError(
                f"participant {participant} missing from strict split"
            )
        key_tuple = (
            split,
            source_split,
            participant,
            metadata["video_id"],
            metadata["clip_id"],
            metadata["hand"],
            metadata["object_name"],
        )
        accumulator = accumulators.get(key_tuple)
        if accumulator is None:
            accumulator = ClipAccumulator(
                split=split,
                source_split=source_split,
                participant=participant,
                video_id=metadata["video_id"],
                clip_id=metadata["clip_id"],
                hand=metadata["hand"],
                object_name=metadata["object_name"],
                top_k=args.top_k,
            )
            accumulators[key_tuple] = accumulator
        accumulator.add(
            sample,
            args.contact_threshold_m,
        )
        processed += 1
        if (
            args.progress_every
            and processed % args.progress_every == 0
        ):
            print(
                f"{source_split}: processed {processed}",
                flush=True,
            )

    with Path(path).open("rb") as handle:
        unpickler = StreamingTopLevelUnpickler(
            handle,
            on_sample,
        )
        try:
            unpickler.load()
        except StopIteration:
            pass
    return processed


def main():
    args = parse_args()
    participant_to_split = json.loads(
        args.strict_split.read_text(encoding="utf-8")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accumulators = {}
    sources = {}
    for source_split, path in (
        ("train", args.train_pickle),
        ("test", args.test_pickle),
    ):
        if path is None:
            continue
        processed = process_source(
            source_split,
            path,
            participant_to_split,
            accumulators,
            args,
        )
        sources[source_split] = {
            **file_identity(path),
            "records_processed": processed,
        }

    index_rows = []
    for key in sorted(accumulators):
        accumulator = accumulators[key]
        for segment in accumulator.segments(args.max_gap_frames):
            if not segment["rows"]:
                continue
            index_rows.append(save_segment(
                args.output_dir,
                accumulator,
                segment,
                resume=args.resume,
            ))

    split_counts = Counter(row["split"] for row in index_rows)
    participant_counts = {
        split: len({
            row["participant_id"]
            for row in index_rows
            if row["split"] == split
        })
        for split in ("train", "dev", "test")
    }
    frame_counts = Counter()
    for row in index_rows:
        frame_counts[row["split"]] += row["frame_count"]
    write_json(
        args.output_dir / "manifest.json",
        {
            "method": (
                "stream EPIC-Contact records into compact Top-K "
                "material-point correspondence shards"
            ),
            "strict_split": str(args.strict_split),
            "strict_split_sha256": sha256_file(args.strict_split),
            "contact_threshold_m": args.contact_threshold_m,
            "top_k": args.top_k,
            "max_gap_frames": args.max_gap_frames,
            "sources": sources,
            "sequence_count": len(index_rows),
            "split_sequence_counts": dict(split_counts),
            "split_frame_counts": dict(frame_counts),
            "split_participant_counts": participant_counts,
            "index_path": str(args.output_dir / "index.jsonl"),
            "test_note": (
                "strict test split is reused from the EPIC event baseline "
                "and is a locked confirmation, not a pristine test"
            ),
        },
    )
    index_path = args.output_dir / "index.jsonl"
    temporary = index_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(index_path)
    print(json.dumps({
        "manifest": str(args.output_dir / "manifest.json"),
        "sequences": len(index_rows),
        "split_sequence_counts": dict(split_counts),
        "split_frame_counts": dict(frame_counts),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
