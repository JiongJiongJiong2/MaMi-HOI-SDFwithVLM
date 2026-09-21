#!/usr/bin/env python3
"""Build a participant-disjoint OakInk handover geometry manifest."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

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


SUBJECT_SPLITS = {
    "train": {"0000", "0005", "0006", "0009", "0010"},
    "val": {"0002", "0004", "0007", "0008"},
    "test": {"0001", "0003"},
}
PRIMARY_THRESHOLD_M = 0.005
THRESHOLDS_M = (0.001, 0.003, 0.005, 0.010)
MIN_CONTACT_FRAMES = 15
ROLE_SWITCH_WINDOW_FRAMES = 30


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations-zip", type=Path, required=True)
    parser.add_argument("--objects-zip", type=Path, required=True)
    parser.add_argument("--meta-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-id", default="0")
    return parser.parse_args()


def parse_sequence(sequence):
    parts = sequence.split("_")
    if len(parts) < 4 or parts[1] != "0004":
        return None
    return {
        "object_id": parts[0],
        "intent_id": parts[1],
        "giver": parts[2],
        "receiver": parts[3],
    }


def split_for_subjects(giver, receiver):
    for split, subjects in SUBJECT_SPLITS.items():
        if giver in subjects and receiver in subjects:
            return split
    return None


def parse_sample_name(path):
    base = path.rsplit("/", 1)[-1]
    fields = base[:-4].split("__")
    if len(fields) != 5:
        return None
    sequence, timestamp, subject_flag, frame, camera = fields
    return {
        "sequence": sequence,
        "timestamp": timestamp,
        "subject_flag": subject_flag,
        "frame": int(frame),
        "camera": camera,
    }


def load_pickle(archive, path):
    return pickle.loads(archive.read(path))


def load_object_id_mapping(meta_zip):
    with meta_zip.open("metaV2/object_id.json") as handle:
        return json.load(handle)


def load_downsampled_object(objects_zip, object_name):
    prefix = f"OakInkObjectsV2/{object_name}/align_ds/"
    candidates = [
        name for name in objects_zip.namelist()
        if name.startswith(prefix) and name.lower().endswith(".obj")
    ]
    if not candidates:
        raise FileNotFoundError(object_name)
    lines = objects_zip.read(sorted(candidates)[0]).decode(
        "latin1"
    ).splitlines()
    vertices = np.asarray([
        [float(value) for value in line.split()[1:4]]
        for line in lines
        if line.startswith("v ")
    ], dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(object_name)
    return vertices


def transform_vertices(vertices, transform):
    homogenous = np.concatenate(
        (vertices, np.ones((len(vertices), 1))),
        axis=1,
    )
    return (transform @ homogenous.T).T[:, :3]


def min_vertex_distance(hand_vertices, object_vertices):
    tree = cKDTree(object_vertices)
    distances = tree.query(hand_vertices, k=1, workers=-1)[0]
    return float(distances.min())


def role_switch_event(
    giver_contact,
    receiver_contact,
    minimum_frames,
    window_frames,
):
    giver_stable = stable_contact(
        giver_contact,
        minimum_frames,
        maximum_gap=0,
    )
    receiver_stable = stable_contact(
        receiver_contact,
        minimum_frames,
        maximum_gap=0,
    )
    giver_segments = binary_segments(giver_stable)
    receiver_segments = binary_segments(receiver_stable)
    candidates = []
    for giver_start, giver_end in giver_segments:
        for receiver_start, receiver_end in receiver_segments:
            gap = receiver_start - giver_end
            if -window_frames <= gap <= window_frames:
                candidates.append({
                    "giver_start": int(giver_start),
                    "giver_release": int(giver_end),
                    "receiver_start": int(receiver_start),
                    "receiver_end": int(receiver_end),
                    "onset_minus_release": int(gap),
                })
    return giver_stable, receiver_stable, candidates


def threshold_summary(records, threshold, minimum_frames, window_frames):
    summary = {
        "sequences": len(records),
        "frames": int(sum(len(row["giver_distance_m"]) for row in records)),
        "giver_contact_frames": 0,
        "receiver_contact_frames": 0,
        "both_contact_frames": 0,
        "both_contact_sequences": 0,
        "detected_role_switch_sequences": 0,
        "candidates": [],
    }
    for record in records:
        giver = (
            np.isfinite(record["giver_distance_m"])
            & (record["giver_distance_m"] <= threshold)
        )
        receiver = (
            np.isfinite(record["receiver_distance_m"])
            & (record["receiver_distance_m"] <= threshold)
        )
        giver_stable, receiver_stable, candidates = role_switch_event(
            giver,
            receiver,
            minimum_frames,
            window_frames,
        )
        summary["giver_contact_frames"] += int(giver.sum())
        summary["receiver_contact_frames"] += int(receiver.sum())
        summary["both_contact_frames"] += int((giver & receiver).sum())
        if giver.any() and receiver.any():
            summary["both_contact_sequences"] += 1
        if candidates:
            summary["detected_role_switch_sequences"] += 1
        summary["candidates"].extend({
            "sequence": record["sequence"],
            "split": record["split"],
            "participant_pair": record["participant_pair"],
            "object_id": record["object_id"],
            "object_name": record["object_name"],
            **candidate,
        } for candidate in candidates)
    return summary


def write_jsonl_gzip(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import zipfile

    with zipfile.ZipFile(args.meta_zip) as meta_zip:
        object_mapping = load_object_id_mapping(meta_zip)

    with zipfile.ZipFile(args.annotations_zip) as annotations_zip:
        names = annotations_zip.namelist()
        sequence_frames = defaultdict(set)
        for name in names:
            if (
                not name.startswith("anno/hand_v/")
                or not name.endswith(".pkl")
            ):
                continue
            sample = parse_sample_name(name)
            if sample is None:
                continue
            parsed = parse_sequence(sample["sequence"])
            if parsed is None:
                continue
            if sample["camera"] != args.camera_id:
                continue
            if sample["subject_flag"] == "0":
                sequence_frames[sample["sequence"]].add((
                    sample["timestamp"],
                    sample["frame"],
                ))

        sequence_meta = {}
        for sequence in sequence_frames:
            parsed = parse_sequence(sequence)
            split = split_for_subjects(
                parsed["giver"],
                parsed["receiver"],
            )
            if split is not None:
                sequence_meta[sequence] = {
                    **parsed,
                    "split": split,
                    "frames": sorted(
                        sequence_frames[sequence],
                        key=lambda item: (item[0], item[1]),
                    ),
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
        failures = []
        for index, (sequence, metadata) in enumerate(
            sorted(sequence_meta.items()),
            start=1,
        ):
            try:
                giver_distances = []
                receiver_distances = []
                for timestamp, frame in metadata["frames"]:
                    stem = (
                        f"{sequence}__{timestamp}__{{flag}}__"
                        f"{frame}__{args.camera_id}"
                    )
                    object_path = f"anno/obj_transf/{stem.format(flag='0')}.pkl"
                    object_transform = load_pickle(
                        annotations_zip,
                        object_path,
                    )
                    transformed_object = transform_vertices(
                        object_vertices[metadata["object_id"]],
                        object_transform,
                    )
                    for flag, output in (
                        ("0", giver_distances),
                        ("1", receiver_distances),
                    ):
                        hand_path = f"anno/hand_v/{stem.format(flag=flag)}.pkl"
                        hand_vertices = load_pickle(
                            annotations_zip,
                            hand_path,
                        )
                        output.append(min_vertex_distance(
                            hand_vertices,
                            transformed_object,
                        ))
                records.append({
                    "sequence": sequence,
                    "split": metadata["split"],
                    "participant_pair": (
                        f"{metadata['giver']}->{metadata['receiver']}"
                    ),
                    "object_id": metadata["object_id"],
                    "object_name": object_names[metadata["object_id"]],
                    "giver_distance_m": np.asarray(
                        giver_distances,
                        dtype=np.float32,
                    ),
                    "receiver_distance_m": np.asarray(
                        receiver_distances,
                        dtype=np.float32,
                    ),
                })
                print(
                    f"[{index}/{len(sequence_meta)}] {sequence} "
                    f"frames={len(giver_distances)}",
                    flush=True,
                )
            except Exception as error:
                failures.append({
                    "sequence": sequence,
                    "error": repr(error),
                })
                print(f"FAILED {sequence}: {error}", flush=True)

    if not records:
        raise RuntimeError("no OakInk handover records")
    split_records = {
        split: [row for row in records if row["split"] == split]
        for split in ("train", "val", "test")
    }
    threshold_results = {
        f"{threshold:.4f}": {
            split: threshold_summary(
                rows,
                threshold,
                MIN_CONTACT_FRAMES,
                ROLE_SWITCH_WINDOW_FRAMES,
            )
            for split, rows in split_records.items()
        }
        for threshold in THRESHOLDS_M
    }
    primary = threshold_results[f"{PRIMARY_THRESHOLD_M:.4f}"]
    gate = {
        split: {
            "sequences_ge_20": primary[split]["sequences"] >= 20,
            "participants_ge_2": len({
                row["participant_pair"].split("->")[0]
                for row in split_records[split]
            } | {
                row["participant_pair"].split("->")[1]
                for row in split_records[split]
            }) >= 2,
            "objects_ge_5": len({
                row["object_name"] for row in split_records[split]
            }) >= 5,
            "both_contact_rate_ge_0_9": (
                primary[split]["both_contact_sequences"]
                / max(primary[split]["sequences"], 1)
                >= 0.9
            ),
            "role_switch_rate_ge_0_5": (
                primary[split]["detected_role_switch_sequences"]
                / max(primary[split]["sequences"], 1)
                >= 0.5
            ),
        }
        for split in ("train", "val", "test")
    }
    for split, checks in gate.items():
        checks["overall"] = all(checks.values())
    result = {
        "subject_splits": {
            split: sorted(subjects)
            for split, subjects in SUBJECT_SPLITS.items()
        },
        "availability": {
            "retained_sequences": len(records),
            "failure_count": len(failures),
            "split_sequences": {
                split: len(rows)
                for split, rows in split_records.items()
            },
        },
        "primary_threshold_m": PRIMARY_THRESHOLD_M,
        "thresholds": threshold_results,
        "gate": gate,
        "failures": failures,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        args.output_dir / "handover_trajectories.npz",
        sequences=np.asarray([row["sequence"] for row in records]),
        splits=np.asarray([row["split"] for row in records]),
        participant_pairs=np.asarray([
            row["participant_pair"] for row in records
        ]),
        object_ids=np.asarray([row["object_id"] for row in records]),
        object_names=np.asarray([
            row["object_name"] for row in records
        ]),
        giver_distance_m=np.asarray([
            row["giver_distance_m"] for row in records
        ], dtype=object),
        receiver_distance_m=np.asarray([
            row["receiver_distance_m"] for row in records
        ], dtype=object),
    )
    write_jsonl_gzip(
        args.output_dir / "role_switch_candidates.jsonl.gz",
        primary["train"]["candidates"]
        + primary["val"]["candidates"]
        + primary["test"]["candidates"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
