#!/usr/bin/env python3
"""Compute camera-0 contact trajectories for non-handover OakInk intents."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.build_oakink_handover_manifest import (
        load_downsampled_object,
        load_object_id_mapping,
        min_vertex_distance,
        parse_sample_name,
        transform_vertices,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_oakink_handover_manifest import (  # noqa: E402
        load_downsampled_object,
        load_object_id_mapping,
        min_vertex_distance,
        parse_sample_name,
        transform_vertices,
    )


INTENTS = {"0001": "use", "0002": "hold", "0003": "liftup"}
SUBJECT_SPLITS = {
    "train": {"0000", "0005", "0006", "0009", "0010"},
    "val": {"0002", "0004", "0007", "0008"},
    "test": {"0001", "0003"},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--objects-zip", type=Path, required=True)
    parser.add_argument("--meta-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def split_for_subject(subject):
    for split, subjects in SUBJECT_SPLITS.items():
        if subject in subjects:
            return split
    return None


def parse_nonhandover_sequence(sequence):
    parts = sequence.split("_")
    if len(parts) != 3 or parts[1] not in INTENTS:
        return None
    return {
        "object_id": parts[0],
        "intent_id": parts[1],
        "subject": parts[2],
    }


def main():
    args = parse_args()
    import zipfile

    with zipfile.ZipFile(args.meta_zip) as meta_zip:
        object_mapping = load_object_id_mapping(meta_zip)
    with zipfile.ZipFile(args.annotations_zip) as annotations_zip:
        sequence_frames = defaultdict(set)
        for name in annotations_zip.namelist():
            if not name.startswith("anno/hand_v/") or not name.endswith(".pkl"):
                continue
            sample = parse_sample_name(name)
            if sample is None:
                continue
            parsed = parse_nonhandover_sequence(sample["sequence"])
            if parsed is None:
                continue
            if sample["camera"] != "0" or sample["subject_flag"] != "0":
                continue
            sequence_frames[sample["sequence"]].add((
                sample["timestamp"],
                sample["frame"],
            ))

        sequence_meta = {}
        for sequence, frames in sequence_frames.items():
            parsed = parse_nonhandover_sequence(sequence)
            split = split_for_subject(parsed["subject"])
            if split is None:
                continue
            sequence_meta[sequence] = {
                **parsed,
                "split": split,
                "frames": sorted(frames, key=lambda item: (item[0], item[1])),
            }
        object_names = {
            object_id: object_mapping[object_id]["name"]
            for object_id in {
                row["object_id"] for row in sequence_meta.values()
            }
        }
        with zipfile.ZipFile(args.objects_zip) as objects_zip:
            object_vertices = {
                object_id: load_downsampled_object(
                    objects_zip,
                    object_name,
                )
                for object_id, object_name in object_names.items()
            }
        records = []
        for index, (sequence, metadata) in enumerate(
            sorted(sequence_meta.items()),
            start=1,
        ):
            distances = []
            for timestamp, frame in metadata["frames"]:
                stem = (
                    f"{sequence}__{timestamp}__0__{frame}__0"
                )
                transform = pickle.loads(annotations_zip.read(
                    f"anno/obj_transf/{stem}.pkl"
                ))
                transformed = transform_vertices(
                    object_vertices[metadata["object_id"]],
                    transform,
                )
                hand = pickle.loads(
                    annotations_zip.read(f"anno/hand_v/{stem}.pkl")
                )
                distances.append(min_vertex_distance(hand, transformed))
            records.append({
                "sequence": sequence,
                "split": metadata["split"],
                "subject": metadata["subject"],
                "intent_id": metadata["intent_id"],
                "intent_name": INTENTS[metadata["intent_id"]],
                "object_id": metadata["object_id"],
                "object_name": object_names[metadata["object_id"]],
                "distance_m": np.asarray(distances, dtype=np.float32),
            })
            if index % 20 == 0 or index == len(sequence_meta):
                print(
                    f"[{index}/{len(sequence_meta)}] {sequence}",
                    flush=True,
                )

    maximum_length = max(len(row["distance_m"]) for row in records)
    distances = np.full(
        (len(records), maximum_length),
        np.nan,
        dtype=np.float32,
    )
    lengths = np.zeros(len(records), dtype=np.int32)
    for index, row in enumerate(records):
        length = len(row["distance_m"])
        lengths[index] = length
        distances[index, :length] = row["distance_m"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sequences=np.asarray([row["sequence"] for row in records]),
        splits=np.asarray([row["split"] for row in records]),
        subjects=np.asarray([row["subject"] for row in records]),
        intent_ids=np.asarray([row["intent_id"] for row in records]),
        intent_names=np.asarray([row["intent_name"] for row in records]),
        object_ids=np.asarray([row["object_id"] for row in records]),
        object_names=np.asarray([row["object_name"] for row in records]),
        lengths=lengths,
        distance_m=distances,
    )
    summary = {
        "sequences": len(records),
        "frames": int(lengths.sum()),
        "by_split": {
            split: {
                "sequences": sum(row["split"] == split for row in records),
                "frames": int(sum(
                    len(row["distance_m"])
                    for row in records
                    if row["split"] == split
                )),
                "intents": sorted({
                    row["intent_name"]
                    for row in records
                    if row["split"] == split
                }),
            }
            for split in ("train", "val", "test")
        },
    }
    args.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
