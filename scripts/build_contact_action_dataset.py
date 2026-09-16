#!/usr/bin/env python3
"""Build action-conditioned contact/response windows from MaMi processed data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F

try:
    import trimesh
except ImportError:
    trimesh = None

from manip.world_model.contact_action.features import (
    ACTION_DIM,
    STATE_DIM,
    build_window_features,
    make_training_samples,
    select_contact_event_anchors,
)


TRAIN_OBJECTS = (
    "largetable",
    "woodchair",
    "plasticbox",
    "largebox",
    "smallbox",
    "trashcan",
    "monitor",
    "floorlamp",
    "clothesstand",
)
VALIDATION_OBJECTS = (
    "smalltable",
    "whitechair",
    "suitcase",
    "tripod",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--anchor_stride", type=int, default=8)
    parser.add_argument("--max_samples_per_window", type=int, default=8)
    parser.add_argument("--max_train_windows", type=int, default=2000)
    parser.add_argument("--max_val_windows", type=int, default=400)
    parser.add_argument("--event_window_fraction", type=float, default=0.8)
    parser.add_argument("--max_event_samples_per_window", type=int, default=16)
    parser.add_argument(
        "--max_background_samples_per_window",
        type=int,
        default=4,
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path):
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def object_from_sequence(sequence_name):
    parts = sequence_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected sequence name: {sequence_name}")
    return parts[1]


def load_sequence_transitions(sequence_name, contact_root, cache):
    if sequence_name in cache:
        return cache[sequence_name]
    contact_path = contact_root / f"{sequence_name}.npy"
    if not contact_path.is_file():
        raise FileNotFoundError(contact_path)
    contact = np.load(contact_path)
    if contact.ndim != 2 or contact.shape[1] < 2:
        raise ValueError(f"Unexpected contact shape for {sequence_name}")
    binary = contact[:, :2] >= 0.5
    changed = np.any(binary[1:] != binary[:-1], axis=1)
    transitions = np.flatnonzero(changed) + 1
    cache[sequence_name] = transitions
    return transitions


def find_object_mesh(data_root: Path, object_name):
    folder = data_root / "captured_objects"
    candidates = sorted(folder.glob(f"{object_name}*"))
    if not candidates:
        raise FileNotFoundError(f"No PLY mesh under {folder}")
    return candidates[0]


def bounds_from_obj(path):
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            values = line.split()
            if len(values) < 4:
                continue
            vertices.append([float(value) for value in values[1:4]])
    if not vertices:
        raise ValueError(f"No vertices found in {path}")
    vertices = np.asarray(vertices, dtype=np.float64)
    return np.stack([vertices.min(axis=0), vertices.max(axis=0)])


def load_object_scale(data_root: Path, object_name, cache):
    if object_name in cache:
        return cache[object_name]
    mesh_path = find_object_mesh(data_root, object_name)
    if mesh_path.suffix.lower() == ".obj" or trimesh is None:
        bounds = bounds_from_obj(mesh_path)
    else:
        mesh = trimesh.load(mesh_path, force="mesh")
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
    scale = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid extent for {object_name}: {scale}")
    cache[object_name] = scale
    return scale


def load_object_sdf(data_root: Path, object_name, cache):
    if object_name in cache:
        return cache[object_name]
    sdf_dir = data_root / "rest_object_sdf_256_npy_files"
    sdf_path = sdf_dir / f"{object_name}.ply.npy"
    metadata_path = sdf_dir / f"{object_name}.ply.json"
    if not sdf_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing SDF pair: {sdf_path}, {metadata_path}"
        )
    sdf = np.load(sdf_path, allow_pickle=True)
    if sdf.shape != (256, 256, 256):
        raise ValueError(f"SDF must be 256^3, got {sdf.shape}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    grid = torch.from_numpy(np.ascontiguousarray(sdf)).float()[None, None]
    grid = F.interpolate(
        grid,
        size=(64, 64, 64),
        mode="trilinear",
        align_corners=True,
    )
    centroid = torch.tensor(
        metadata["centroid"],
        dtype=torch.float32,
    )[None]
    extents = torch.tensor(
        metadata["extents"],
        dtype=torch.float32,
    )[None]
    cache[object_name] = (grid, centroid, extents)
    return cache[object_name]


def select_window_indices(
    window_data,
    allowed_objects,
    maximum,
    seed,
    contact_root,
    event_window_fraction,
):
    items = list(window_data.values())
    candidates = [
        index
        for index, item in enumerate(items)
        if object_from_sequence(item["seq_name"]) in allowed_objects
    ]
    random.Random(seed).shuffle(candidates)
    if not candidates:
        raise ValueError("No windows matched the requested object split")
    if maximum is None:
        return candidates
    if not 0.0 <= event_window_fraction <= 1.0:
        raise ValueError("event_window_fraction must lie in [0, 1]")

    event_target = int(round(maximum * event_window_fraction))
    background_target = maximum - event_target
    event_indices = []
    background_indices = []
    transition_cache = {}
    for index in candidates:
        item = items[index]
        sequence_name = item["seq_name"]
        transitions = load_sequence_transitions(
            sequence_name,
            contact_root,
            transition_cache,
        )
        start = int(item["start_t_idx"])
        end = int(item["end_t_idx"])
        contains_event = bool(
            np.any((transitions >= start) & (transitions <= end))
        )
        if contains_event and len(event_indices) < event_target:
            event_indices.append(index)
        elif not contains_event and len(background_indices) < background_target:
            background_indices.append(index)
        if (
            len(event_indices) >= event_target
            and len(background_indices) >= background_target
        ):
            break

    selected = event_indices + background_indices
    if len(selected) < maximum:
        selected_set = set(selected)
        for index in candidates:
            if index in selected_set:
                continue
            selected.append(index)
            selected_set.add(index)
            if len(selected) >= maximum:
                break
    random.Random(seed + 1).shuffle(selected)
    return selected


def build_split(
    window_data,
    data_root,
    contact_root,
    jpos_minimum,
    jpos_maximum,
    allowed_objects,
    maximum_windows,
    history,
    horizon,
    anchor_stride,
    max_samples_per_window,
    event_window_fraction,
    max_event_samples_per_window,
    max_background_samples_per_window,
    seed,
):
    scale_cache = {}
    sdf_cache = {}
    items = list(window_data.values())
    indices = select_window_indices(
        window_data,
        allowed_objects,
        maximum_windows,
        seed,
        contact_root,
        event_window_fraction,
    )
    outputs = []
    for count, index in enumerate(indices, start=1):
        item = items[index]
        sequence_name = item["seq_name"]
        object_name = object_from_sequence(sequence_name)
        contact_path = contact_root / f"{sequence_name}.npy"
        if not contact_path.is_file():
            raise FileNotFoundError(contact_path)

        contact = np.load(contact_path)
        start = int(item["start_t_idx"])
        end = int(item["end_t_idx"])
        window_contact = contact[start : end + 1]
        expected_frames = item["motion"].shape[0]
        if len(window_contact) != expected_frames:
            window_contact = contact[start : start + expected_frames]
        if len(window_contact) != expected_frames:
            raise ValueError(
                f"contact/motion length mismatch for {sequence_name}: "
                f"{len(window_contact)} vs {expected_frames}"
            )

        features = build_window_features(
            motion=item["motion"],
            contact_labels=window_contact,
            object_pos=item["window_obj_com_pos"],
            object_rot=item["obj_rot_mat"],
            jpos_minimum=jpos_minimum,
            jpos_maximum=jpos_maximum,
            object_scale=load_object_scale(data_root, object_name, scale_cache),
            object_sdf=load_object_sdf(data_root, object_name, sdf_cache),
        )
        event_anchors = select_contact_event_anchors(
            features.states,
            history=history,
            horizon=horizon,
            event_samples=max_event_samples_per_window,
            background_samples=max_background_samples_per_window,
        )
        samples = make_training_samples(
            features.states,
            features.actions,
            history=history,
            horizon=horizon,
            stride=anchor_stride,
            max_samples=max_samples_per_window,
            anchor_indices=event_anchors,
        )
        if samples is None:
            continue
        samples["sequence_index"] = np.full(
            samples["state_history"].shape[0],
            index,
            dtype=np.int64,
        )
        samples["anchor_index"] = samples.pop("anchor_indices")
        samples["object_index"] = np.full(
            samples["state_history"].shape[0],
            VALIDATION_OBJECTS.index(object_name)
            if object_name in VALIDATION_OBJECTS
            else TRAIN_OBJECTS.index(object_name),
            dtype=np.int64,
        )
        outputs.append(samples)

        if count % 50 == 0:
            print(f"processed {count}/{len(indices)} windows")

    if not outputs:
        raise ValueError("No training samples were produced")

    keys = (
        "state_history",
        "action_history",
        "future_actions",
        "future_states",
        "sequence_index",
        "anchor_index",
        "object_index",
    )
    return {
        key: np.concatenate([output[key] for output in outputs], axis=0)
        for key in keys
    }


def validate_shapes(split, name):
    count = split["state_history"].shape[0]
    expected = {
        "state_history": (count,),
        "action_history": (count,),
        "future_actions": (count,),
        "future_states": (count,),
        "sequence_index": (count,),
        "anchor_index": (count,),
        "object_index": (count,),
    }
    for key, prefix in expected.items():
        if split[key].shape[: len(prefix)] != prefix:
            raise ValueError(f"{name}.{key} has invalid shape {split[key].shape}")
    if split["state_history"].shape[-1] != STATE_DIM:
        raise ValueError(f"{name}.state_history must end in {STATE_DIM}")
    for key in ("action_history", "future_actions"):
        if split[key].shape[-1] != ACTION_DIM:
            raise ValueError(f"{name}.{key} must end in {ACTION_DIM}")
    if split["future_states"].shape[-1] != STATE_DIM:
        raise ValueError(f"{name}.future_states must end in {STATE_DIM}")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_root / (
        f"cano_train_diffusion_manip_window_{args.window}_joints24.p"
    )
    val_path = data_root / (
        f"cano_test_diffusion_manip_window_{args.window}_joints24.p"
    )
    stats_path = data_root / (
        f"cano_min_max_mean_std_data_window_{args.window}_joints24.p"
    )
    for path in (train_path, val_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_data = joblib.load(train_path)
    val_data = joblib.load(val_path)
    stats = joblib.load(stats_path)
    jpos_minimum = np.asarray(stats["global_jpos_min"], dtype=np.float32)
    jpos_maximum = np.asarray(stats["global_jpos_max"], dtype=np.float32)
    contact_root = data_root / "contact_labels_w_semantics_npy_files"

    train_split = build_split(
        train_data,
        data_root,
        contact_root,
        jpos_minimum,
        jpos_maximum,
        set(TRAIN_OBJECTS),
        args.max_train_windows,
        args.history,
        args.horizon,
        args.anchor_stride,
        args.max_samples_per_window,
        args.event_window_fraction,
        args.max_event_samples_per_window,
        args.max_background_samples_per_window,
        args.seed,
    )
    val_split = build_split(
        val_data,
        data_root,
        contact_root,
        jpos_minimum,
        jpos_maximum,
        set(VALIDATION_OBJECTS),
        args.max_val_windows,
        args.history,
        args.horizon,
        args.anchor_stride,
        args.max_samples_per_window,
        args.event_window_fraction,
        args.max_event_samples_per_window,
        args.max_background_samples_per_window,
        args.seed + 1,
    )
    validate_shapes(train_split, "train")
    validate_shapes(val_split, "val")

    train_output = output_dir / "train.npz"
    val_output = output_dir / "val.npz"
    np.savez_compressed(train_output, **train_split)
    np.savez_compressed(val_output, **val_split)

    metadata = {
        "format_version": 2,
        "data_root": str(data_root),
        "window": int(args.window),
        "history": int(args.history),
        "horizon": int(args.horizon),
        "anchor_stride": int(args.anchor_stride),
        "event_window_fraction": float(args.event_window_fraction),
        "max_event_samples_per_window": int(
            args.max_event_samples_per_window
        ),
        "max_background_samples_per_window": int(
            args.max_background_samples_per_window
        ),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "train_objects": list(TRAIN_OBJECTS),
        "validation_objects": list(VALIDATION_OBJECTS),
        "train_samples": int(train_split["state_history"].shape[0]),
        "validation_samples": int(val_split["state_history"].shape[0]),
        "state_layout": {
            "0:3": "object translation relative anchor in object-local frame / scale",
            "3:9": "object rotation relative anchor, 6D",
            "9:15": "left/right palm in object frame / scale",
            "15:21": "left/right palm local velocity / scale",
            "21:24": "object local translation velocity / scale",
            "24:26": "left/right palm proxy signed clearance / scale",
            "26:32": "left/right palm proxy surface normal",
            "32:34": "left/right contact probability",
        },
        "geometry_query": {
            "source": "rest_object_sdf_256_npy_files",
            "grid_resolution_used": 64,
            "query_point": "SMPL-X 24-joint hand joints 22/23",
            "clearance_units": "normalized by object max extent",
            "normal": "normalized finite-difference SDF gradient",
        },
        "action_layout": {
            "0:6": "left/right palm world delta transformed to current object frame / scale",
            "6:9": "object translation delta / scale",
            "9:15": "object rotation delta, 6D",
        },
        "sources": {
            "train": file_identity(train_path),
            "validation": file_identity(val_path),
            "statistics": file_identity(stats_path),
        },
        "outputs": {
            "train": {
                "path": str(train_output),
                "sha256": sha256_file(train_output),
            },
            "validation": {
                "path": str(val_output),
                "sha256": sha256_file(val_output),
            },
        },
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {metadata_path}; train={metadata['train_samples']}, "
        f"val={metadata['validation_samples']}"
    )


if __name__ == "__main__":
    main()
