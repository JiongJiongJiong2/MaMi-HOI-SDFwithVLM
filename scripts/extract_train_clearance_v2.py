"""Build a train-only, trainer-consumable unsigned clearance calibration.

The output target file is intentionally flat:

    {
      "version": 2,
      "units": "m",
      "fallback": {"left": ..., "right": ...},
      "<object>": {"left": ..., "right": ...},
      ...
    }

Per-query records and provenance are written separately so the target file
matches the existing trainer loader without any hidden fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manip.model.sdf_utils import (
    point_to_triangle_unsigned_distance_candidates,
)


EXPECTED_TRAIN_SEQUENCES = 4380
EXPECTED_OBJECTS = 13
DEFAULT_CAP_PER_SEQUENCE_HAND = 20


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-windows", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--max-contact-frames-per-sequence-hand",
        type=int,
        default=DEFAULT_CAP_PER_SEQUENCE_HAND,
    )
    parser.add_argument("--candidate-count", type=int, default=64)
    parser.add_argument("--point-chunk", type=int, default=4096)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evenly_spaced_indices(count: int, cap: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    if count <= cap:
        return np.arange(count, dtype=np.int64)
    return np.unique(
        np.rint(np.linspace(0, count - 1, num=cap)).astype(np.int64)
    )


def object_name_for_sequence(sequence: str, object_names: tuple[str, ...]) -> str:
    matches = [name for name in object_names if name in sequence]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one object name in {sequence!r}, got {matches}"
        )
    return matches[0]


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(value)


def candidate_distances(
    points: torch.Tensor,
    triangles: torch.Tensor,
    candidate_count: int,
    point_chunk: int,
) -> torch.Tensor:
    centroids = triangles.mean(dim=1)
    kept = min(candidate_count, len(triangles))
    distances = []
    for start in range(0, len(points), point_chunk):
        query = points[start : start + point_chunk]
        centroid_distances = torch.cdist(query, centroids)
        _, candidate_indices = torch.topk(
            centroid_distances,
            k=kept,
            dim=1,
            largest=False,
        )
        candidates = triangles[candidate_indices]
        distances.append(
            point_to_triangle_unsigned_distance_candidates(
                query,
                candidates,
                point_chunk=min(512, len(query)),
            )
        )
    return torch.cat(distances)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    split = json.loads(args.split.read_text(encoding="utf-8"))
    validation = set(str(value) for value in split["validation_sequences"])
    test = set(str(value) for value in split["test_sequences"])
    if validation.intersection(test):
        raise ValueError("Validation and test split are not disjoint")

    rest_mesh_dir = args.data_root / "rest_object_geo"
    object_names = tuple(sorted(path.stem for path in rest_mesh_dir.glob("*.ply")))
    if len(object_names) != EXPECTED_OBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_OBJECTS} object meshes, got {object_names}"
        )
    object_to_index = {name: index for index, name in enumerate(object_names)}

    print(
        json.dumps(
            {
                "stage": "loading_train_windows",
                "path": str(args.train_windows),
            }
        ),
        flush=True,
    )
    windows = joblib.load(args.train_windows)
    windows_by_sequence = defaultdict(list)
    frames_by_sequence = defaultdict(set)
    for key, window in windows.items():
        sequence = str(window["seq_name"])
        start = int(window["start_t_idx"])
        end = int(window["end_t_idx"])
        if end < start:
            raise ValueError(f"Invalid window bounds for {sequence}: {start}..{end}")
        windows_by_sequence[sequence].append((start, end, int(key)))
        frames_by_sequence[sequence].update(range(start, end + 1))

    train_sequences = sorted(windows_by_sequence)
    sequence_to_index = {
        sequence: index for index, sequence in enumerate(train_sequences)
    }
    overlap_validation = sorted(set(train_sequences).intersection(validation))
    overlap_test = sorted(set(train_sequences).intersection(test))
    if overlap_validation or overlap_test:
        raise ValueError(
            "Train windows overlap frozen validation/test sequences: "
            f"validation={overlap_validation[:5]}, test={overlap_test[:5]}"
        )
    if len(train_sequences) != EXPECTED_TRAIN_SEQUENCES:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_SEQUENCES} train sequences, "
            f"got {len(train_sequences)}"
        )

    contact_root = args.data_root / "contact_labels_w_semantics_npy_files"
    selected_frames = {}
    missing_contact_labels = []
    sequences_with_contact = 0
    available_contact_counts = defaultdict(int)
    selected_contact_counts = defaultdict(int)
    for sequence in train_sequences:
        contact_path = contact_root / f"{sequence}.npy"
        if not contact_path.exists():
            missing_contact_labels.append(sequence)
            continue
        contact = np.load(contact_path, allow_pickle=False)
        object_name = object_name_for_sequence(sequence, object_names)
        covered_frames = sorted(
            frame
            for frame in frames_by_sequence[sequence]
            if 0 <= frame < len(contact)
        )
        has_contact = False
        for hand_index in (0, 1):
            ids = np.asarray(
                [
                    frame
                    for frame in covered_frames
                    if float(contact[frame, hand_index]) > 0.5
                ],
                dtype=np.int64,
            )
            available_contact_counts[(object_name, hand_index)] += int(len(ids))
            selected = ids[
                evenly_spaced_indices(
                    len(ids),
                    args.max_contact_frames_per_sequence_hand,
                )
            ]
            selected_frames[(sequence, hand_index)] = set(
                int(value) for value in selected
            )
            selected_contact_counts[(object_name, hand_index)] += int(
                len(selected)
            )
            has_contact = has_contact or bool(len(selected))
        sequences_with_contact += int(has_contact)

    if missing_contact_labels:
        raise FileNotFoundError(
            f"Missing contact labels for {len(missing_contact_labels)} sequences: "
            f"{missing_contact_labels[:5]}"
        )

    values = {
        "sequence_index": [],
        "frame_index": [],
        "hand": [],
        "object_index": [],
        "world_point": [],
        "canonical_point": [],
        "object_rotation": [],
        "object_com": [],
        "source_window_key": [],
        "source_local_frame": [],
    }
    seen_selected = set()
    rotation_error_max = 0.0
    determinant_min = float("inf")
    determinant_max = float("-inf")

    for sequence in train_sequences:
        object_name = object_name_for_sequence(sequence, object_names)
        object_index = object_to_index[object_name]
        for start, end, key in sorted(windows_by_sequence[sequence]):
            window = windows[key]
            motion = np.asarray(window["motion"], dtype=np.float64)
            rotations = np.asarray(window["obj_rot_mat"], dtype=np.float64)
            com = np.asarray(window["window_obj_com_pos"], dtype=np.float64)
            available = min(
                len(motion),
                len(rotations),
                len(com),
                end - start + 1,
            )
            for local_frame in range(available):
                absolute_frame = start + local_frame
                pending_hands = [
                    hand_index
                    for hand_index in (0, 1)
                    if absolute_frame
                    in selected_frames[(sequence, hand_index)]
                    and (sequence, absolute_frame, hand_index)
                    not in seen_selected
                ]
                if not pending_hands:
                    continue
                rotation = rotations[local_frame]
                translation = com[local_frame]
                gram = rotation.T @ rotation
                orthogonality_error = float(
                    np.linalg.norm(gram - np.eye(3), ord="fro")
                )
                determinant = float(np.linalg.det(rotation))
                rotation_error_max = max(rotation_error_max, orthogonality_error)
                determinant_min = min(determinant_min, determinant)
                determinant_max = max(determinant_max, determinant)
                if orthogonality_error > 1e-5 or not (
                    0.99999 <= determinant <= 1.00001
                ):
                    raise ValueError(
                        f"Invalid GT rotation at {sequence}:{absolute_frame}: "
                        f"orthogonality={orthogonality_error}, det={determinant}"
                    )
                joints = motion[local_frame, :72].reshape(24, 3)
                for hand_index in pending_hands:
                    world_point = joints[[22, 23][hand_index]]
                    relative = world_point - translation
                    canonical_point = relative @ rotation
                    values["sequence_index"].append(sequence_to_index[sequence])
                    values["frame_index"].append(absolute_frame)
                    values["hand"].append(hand_index)
                    values["object_index"].append(object_index)
                    values["world_point"].append(world_point)
                    values["canonical_point"].append(canonical_point)
                    values["object_rotation"].append(rotation)
                    values["object_com"].append(translation)
                    values["source_window_key"].append(key)
                    values["source_local_frame"].append(local_frame)
                    seen_selected.add((sequence, absolute_frame, hand_index))

    canonical_points = torch.tensor(
        np.asarray(values["canonical_point"], dtype=np.float32),
        device=device,
    )
    object_indices = np.asarray(values["object_index"], dtype=np.int64)
    hands = np.asarray(values["hand"], dtype=np.int64)
    distances = np.full(len(canonical_points), np.nan, dtype=np.float64)
    for object_index, object_name in enumerate(object_names):
        mesh = trimesh.load_mesh(rest_mesh_dir / f"{object_name}.ply")
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        triangles = torch.tensor(
            vertices[faces],
            dtype=torch.float32,
            device=device,
        )
        for hand_index in (0, 1):
            indices = np.flatnonzero(
                (object_indices == object_index) & (hands == hand_index)
            )
            if not len(indices):
                raise AssertionError(
                    f"No sampled records for {object_name} hand {hand_index}"
                )
            with torch.no_grad():
                result = candidate_distances(
                    canonical_points[indices],
                    triangles,
                    candidate_count=args.candidate_count,
                    point_chunk=args.point_chunk,
                )
            distances[indices] = result.detach().cpu().numpy().astype(
                np.float64
            )

    if not np.isfinite(distances).all() or (distances < 0.0).any():
        raise AssertionError("Invalid calibration distances")

    targets = {"version": 2, "units": "m"}
    fallback_values = {0: [], 1: []}
    per_object = {}
    for object_index, object_name in enumerate(object_names):
        per_object[object_name] = {}
        targets[object_name] = {}
        for hand_index, hand_name in enumerate(("left", "right")):
            selected = distances[
                (object_indices == object_index) & (hands == hand_index)
            ]
            if not len(selected):
                raise AssertionError(f"Missing selected values for {object_name}")
            median = float(np.median(selected))
            if not np.isfinite(median) or median <= 0.0:
                raise AssertionError(
                    f"Invalid median for {object_name}.{hand_name}: {median}"
                )
            targets[object_name][hand_name] = median
            per_object[object_name][hand_name] = {
                "median_m": median,
                "query_count": int(len(selected)),
            }
            fallback_values[hand_index].extend(selected.tolist())

    targets["fallback"] = {
        hand_name: float(np.median(fallback_values[hand_index]))
        for hand_index, hand_name in enumerate(("left", "right"))
    }

    sequence_names = np.asarray(train_sequences, dtype="U")
    sequence_index = np.asarray(values["sequence_index"], dtype=np.uint16)
    fallback_fraction = 0.0
    for sequence_idx, hand_index in zip(sequence_index, hands):
        sequence = train_sequences[int(sequence_idx)]
        object_name = object_name_for_sequence(sequence, object_names)
        hand_name = "left" if hand_index == 0 else "right"
        if (
            object_name not in targets
            or hand_name not in targets[object_name]
        ):
            fallback_fraction += 1.0
    fallback_fraction /= max(len(sequence_index), 1)

    targets_path = args.output_root / "unsigned_contact_clearance_train_v2.json"
    records_path = args.output_root / "unsigned_contact_clearance_train_v2_records.npz"
    provenance_path = (
        args.output_root / "unsigned_contact_clearance_train_v2_provenance.json"
    )
    targets_path.write_text(
        json.dumps(targets, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    np.savez_compressed(
        records_path,
        sequence_names=sequence_names,
        sequence_index=sequence_index,
        frame_index=np.asarray(values["frame_index"], dtype=np.uint32),
        hand=np.asarray(values["hand"], dtype=np.uint8),
        object_index=np.asarray(values["object_index"], dtype=np.uint8),
        distance_m=distances.astype(np.float64),
        world_point=np.asarray(values["world_point"], dtype=np.float64),
        canonical_point=np.asarray(
            values["canonical_point"], dtype=np.float64
        ),
        object_rotation=np.asarray(
            values["object_rotation"], dtype=np.float64
        ),
        object_com=np.asarray(values["object_com"], dtype=np.float64),
        source_window_key=np.asarray(
            values["source_window_key"], dtype=np.int64
        ),
        source_local_frame=np.asarray(
            values["source_local_frame"], dtype=np.uint16
        ),
    )

    hash_paths = {
        "train_windows": args.train_windows,
        "split_manifest": args.split,
        "extractor_script": Path(__file__).resolve(),
    }
    for object_name in object_names:
        hash_paths[f"mesh:{object_name}"] = (
            rest_mesh_dir / f"{object_name}.ply"
        )
    provenance = {
        "version": 2,
        "status": "complete",
        "experiment": "DSR-01R",
        "object_names": list(object_names),
        "paths": {
            **{key: str(path) for key, path in hash_paths.items()},
            "targets": str(targets_path),
            "records": str(records_path),
        },
        "hashes": {key: sha256(path) for key, path in hash_paths.items()},
        "sampling": {
            "policy": (
                "All training sequences; per sequence and hand, up to "
                f"{args.max_contact_frames_per_sequence_hand} evenly spaced "
                "GT-contact absolute frames from the union of processed windows."
            ),
            "cap_per_sequence_hand": (
                args.max_contact_frames_per_sequence_hand
            ),
            "scope": (
                "sequence-balanced temporal subset; not exhaustive over every "
                "window and frame"
            ),
        },
        "coordinate_contract": {
            "world_to_object": "(world_point - object_com) @ object_rotation",
            "object_to_world": (
                "object_point @ object_rotation.T + object_com"
            ),
        },
        "query_contract": {
            "mesh_loader": (
                "trimesh.load_mesh, matching "
                "CanoObjectTrajDataset.load_rest_pose_object_geometry"
            ),
            "candidate_count": args.candidate_count,
            "candidate_selection": (
                "top-k triangle centroids by Euclidean distance"
            ),
            "distance": (
                "exact unsigned point-to-triangle distance within candidates"
            ),
            "device": str(device),
        },
        "coverage": {
            "train_windows": len(windows),
            "train_sequences": len(train_sequences),
            "sequences_with_contact": sequences_with_contact,
            "validation_overlap": len(overlap_validation),
            "test_overlap": len(overlap_test),
            "missing_contact_label_sequences": len(missing_contact_labels),
            "records": int(len(distances)),
            "objects": len(object_names),
            "object_hands": len(object_names) * 2,
            "fallback_fraction": fallback_fraction,
            "per_object": per_object,
            "available_contact_frames": {
                object_name: {
                    "left": available_contact_counts[(object_name, 0)],
                    "right": available_contact_counts[(object_name, 1)],
                }
                for object_name in object_names
            },
            "selected_contact_frames": {
                object_name: {
                    "left": selected_contact_counts[(object_name, 0)],
                    "right": selected_contact_counts[(object_name, 1)],
                }
                for object_name in object_names
            },
        },
        "rotation_validity": {
            "max_frobenius_orthogonality_error": rotation_error_max,
            "determinant_min": determinant_min,
            "determinant_max": determinant_max,
        },
        "completeness": {
            "status": "complete",
            "expected_train_sequences": EXPECTED_TRAIN_SEQUENCES,
            "expected_objects": EXPECTED_OBJECTS,
            "missing_records": False,
            "fallback_fraction": fallback_fraction,
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "targets": str(targets_path),
                "records": str(records_path),
                "provenance": str(provenance_path),
                "records_count": int(len(distances)),
                "fallback": targets["fallback"],
                "fallback_fraction": fallback_fraction,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
