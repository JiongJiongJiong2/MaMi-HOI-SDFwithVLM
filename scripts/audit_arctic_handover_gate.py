#!/usr/bin/env python3
"""Audit public ARCTIC sequences for contact and role-switch candidates."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np


PRIMARY_THRESHOLD_M = 0.003
THRESHOLDS_M = (0.001, 0.003, 0.005, 0.010)
MIN_CONTACT_FRAMES = 15
BRIDGE_GAP_FRAMES = 0
HANDOVER_WINDOW_FRAMES = 15


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic-root", type=Path, required=True)
    parser.add_argument("--body-models", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-chunk", type=int, default=16)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--min-contact-frames", type=int, default=MIN_CONTACT_FRAMES)
    parser.add_argument("--bridge-gap-frames", type=int, default=BRIDGE_GAP_FRAMES)
    parser.add_argument("--handover-window-frames", type=int, default=HANDOVER_WINDOW_FRAMES)
    return parser.parse_args()


def close_short_false_gaps(mask, maximum_gap):
    mask = np.asarray(mask, dtype=bool)
    result = mask.copy()
    if maximum_gap <= 0:
        return result
    index = 0
    while index < len(mask):
        if mask[index]:
            index += 1
            continue
        end = index
        while end < len(mask) and not mask[end]:
            end += 1
        if (
            index > 0
            and end < len(mask)
            and mask[index - 1]
            and mask[end]
            and end - index <= maximum_gap
        ):
            result[index:end] = True
        index = end
    return result


def remove_short_true_runs(mask, minimum_length):
    mask = np.asarray(mask, dtype=bool)
    result = mask.copy()
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        end = index
        while end < len(mask) and mask[end]:
            end += 1
        if end - index < minimum_length:
            result[index:end] = False
        index = end
    return result


def stable_contact(
    raw_contact,
    minimum_length,
    maximum_gap,
):
    bridged = close_short_false_gaps(raw_contact, maximum_gap)
    return remove_short_true_runs(bridged, minimum_length)


def binary_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    segments = []
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        end = index
        while end < len(mask) and mask[end]:
            end += 1
        segments.append((index, end))
        index = end
    return segments


def role_switch_candidates(
    outgoing_segments,
    receiving_segments,
    minimum_frames,
    window_frames,
    outgoing_hand,
    receiving_hand,
):
    candidates = []
    for outgoing_start, outgoing_end in outgoing_segments:
        if outgoing_end - outgoing_start < minimum_frames:
            continue
        release_frame = outgoing_end
        for receiving_start, receiving_end in receiving_segments:
            if receiving_end - receiving_start < minimum_frames:
                continue
            if not (
                release_frame - window_frames
                <= receiving_start
                <= release_frame + window_frames
            ):
                continue
            candidates.append({
                "outgoing_hand": outgoing_hand,
                "receiving_hand": receiving_hand,
                "direction": f"{outgoing_hand}_to_{receiving_hand}",
                "outgoing_start": int(outgoing_start),
                "outgoing_release": int(release_frame),
                "receiving_start": int(receiving_start),
                "receiving_end": int(receiving_end),
                "onset_minus_release": int(receiving_start - release_frame),
            })
    return candidates


def contact_summary(
    right_distance_m,
    left_distance_m,
    threshold_m,
    minimum_frames,
    maximum_gap,
    window_frames,
):
    right_distance_m = np.asarray(right_distance_m, dtype=np.float64)
    left_distance_m = np.asarray(left_distance_m, dtype=np.float64)
    right_raw = np.isfinite(right_distance_m) & (
        right_distance_m <= threshold_m
    )
    left_raw = np.isfinite(left_distance_m) & (
        left_distance_m <= threshold_m
    )
    right_stable = stable_contact(
        right_raw,
        minimum_frames,
        maximum_gap,
    )
    left_stable = stable_contact(
        left_raw,
        minimum_frames,
        maximum_gap,
    )
    right_segments = binary_segments(right_stable)
    left_segments = binary_segments(left_stable)
    candidates = role_switch_candidates(
        right_segments,
        left_segments,
        minimum_frames,
        window_frames,
        "right",
        "left",
    )
    candidates.extend(
        role_switch_candidates(
            left_segments,
            right_segments,
            minimum_frames,
            window_frames,
            "left",
            "right",
        )
    )
    return {
        "frame_contact_right": int(right_raw.sum()),
        "frame_contact_left": int(left_raw.sum()),
        "frame_contact_both": int((right_raw & left_raw).sum()),
        "stable_segments_right": len(right_segments),
        "stable_segments_left": len(left_segments),
        "stable_bimanual_frames": int(
            (right_stable & left_stable).sum()
        ),
        "has_stable_bimanual_overlap": bool(
            (right_stable & left_stable).any()
        ),
        "segments": {
            "right": right_segments,
            "left": left_segments,
        },
        "candidates": candidates,
    }


def load_obj_vertices(path):
    vertices = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()
                vertices.append([float(value) for value in values[1:4]])
    vertices = np.asarray(vertices, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"invalid object mesh: {path}")
    return vertices / 1000.0


def axis_angle_to_matrix_torch(axis_angle):
    import torch

    axis_angle = torch.as_tensor(axis_angle, dtype=torch.float32)
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    safe_theta = theta.clamp_min(1e-8)
    axis = axis_angle / safe_theta
    x, y, z = axis.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((
        torch.stack((zero, -z, y), dim=-1),
        torch.stack((z, zero, -x), dim=-1),
        torch.stack((-y, x, zero), dim=-1),
    ), dim=-2)
    identity = torch.eye(3, device=axis.device).expand(
        axis.shape[0],
        3,
        3,
    )
    sin_theta = torch.sin(theta)[..., None]
    cos_theta = torch.cos(theta)[..., None]
    matrix = (
        identity
        + sin_theta * skew
        + (1.0 - cos_theta) * (skew @ skew)
    )
    return torch.where(
        (theta > 1e-8)[..., None],
        matrix,
        identity,
    )


def transform_object_vertices(
    template_vertices,
    top_mask,
    object_parameters,
    device,
):
    import torch

    template = torch.as_tensor(
        template_vertices,
        dtype=torch.float32,
        device=device,
    )
    mask = torch.as_tensor(top_mask, dtype=torch.bool, device=device)
    parameters = torch.as_tensor(
        object_parameters,
        dtype=torch.float32,
        device=device,
    )
    articulation = parameters[:, 0]
    global_orientation = parameters[:, 1:4]
    translation = parameters[:, 4:7] / 1000.0
    frame_count = parameters.shape[0]
    vertices = template[None].expand(frame_count, -1, -1).clone()

    articulation_axis = torch.zeros(
        (frame_count, 3),
        dtype=torch.float32,
        device=device,
    )
    articulation_axis[:, 2] = -articulation
    articulation_rotation = axis_angle_to_matrix_torch(
        articulation_axis
    )
    vertices[:, mask] = torch.einsum(
        "tij,tvj->tvi",
        articulation_rotation,
        vertices[:, mask],
    )

    global_rotation = axis_angle_to_matrix_torch(global_orientation)
    vertices = torch.einsum(
        "tij,tvj->tvi",
        global_rotation,
        vertices,
    )
    return vertices + translation[:, None, :]


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_official_splits(split_root):
    protocol = load_json(split_root / "protocol_p1.json")
    assignment = {}
    for split in ("train", "val", "test"):
        for sequence in protocol[split]:
            assignment[sequence] = split
    return protocol, assignment


def sequence_to_filename(key):
    return key.replace("/", "__") + ".npz"


class ArcticRunner:
    def __init__(self, args):
        import smplx
        import torch
        from pytorch3d.ops import knn_points

        self.smplx = smplx
        self.torch = torch
        self.knn_points = knn_points
        self.args = args
        self.device = torch.device(args.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.right_model = smplx.MANO(
            str(args.body_models / "mano"),
            create_transl=False,
            use_pca=False,
            flat_hand_mean=False,
            is_rhand=True,
        ).to(self.device)
        self.left_model = smplx.MANO(
            str(args.body_models / "mano"),
            create_transl=False,
            use_pca=False,
            flat_hand_mean=False,
            is_rhand=False,
        ).to(self.device)
        self.object_cache = {}

    def object_template(self, object_name):
        if object_name in self.object_cache:
            return self.object_cache[object_name]
        root = (
            self.args.arctic_root
            / "meta"
            / "object_vtemplates"
            / object_name
        )
        vertices = load_obj_vertices(root / "mesh.obj")
        parts = np.asarray(load_json(root / "parts.json"), dtype=np.int64)
        if len(parts) != len(vertices):
            raise ValueError(f"parts/vertex mismatch for {object_name}")
        self.object_cache[object_name] = (
            vertices,
            parts.astype(bool),
        )
        return self.object_cache[object_name]

    def process(self, sequence_path):
        torch = self.torch
        mano = np.load(
            sequence_path,
            allow_pickle=True,
        ).item()
        object_path = Path(
            str(sequence_path).replace(".mano.npy", ".object.npy")
        )
        object_parameters = np.load(object_path, allow_pickle=True)
        frame_count = len(object_parameters)

        right = mano["right"]
        left = mano["left"]
        for hand_name, hand in (("right", right), ("left", left)):
            if len(hand["rot"]) != frame_count:
                raise ValueError(
                    f"{hand_name} frame mismatch in {sequence_path}"
                )

        object_name = sequence_path.name.split("_", 1)[0]
        template_vertices, top_mask = self.object_template(object_name)
        object_vertices = transform_object_vertices(
            template_vertices,
            top_mask,
            object_parameters,
            self.device,
        )

        right_distances = np.empty(frame_count, dtype=np.float32)
        left_distances = np.empty(frame_count, dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, frame_count, self.args.frame_chunk):
                end = min(start + self.args.frame_chunk, frame_count)
                object_chunk = object_vertices[start:end]
                for hand_name, hand, model, output in (
                    ("right", right, self.right_model, right_distances),
                    ("left", left, self.left_model, left_distances),
                ):
                    global_orient = torch.as_tensor(
                        hand["rot"][start:end],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    hand_pose = torch.as_tensor(
                        hand["pose"][start:end],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    betas = torch.as_tensor(
                        np.repeat(
                            np.asarray(hand["shape"])[None],
                            end - start,
                            axis=0,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    transl = torch.as_tensor(
                        hand["trans"][start:end],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    hand_vertices = model(
                        global_orient=global_orient,
                        hand_pose=hand_pose,
                        betas=betas,
                        transl=transl,
                    ).vertices
                    squared, _, _ = self.knn_points(
                        hand_vertices.contiguous(),
                        object_chunk.contiguous(),
                        K=1,
                    )
                    vertex_distances = torch.sqrt(
                        torch.clamp(squared[..., 0], min=0.0)
                    )
                    output[start:end] = (
                        vertex_distances.min(dim=1).values
                        .detach()
                        .cpu()
                        .numpy()
                    )
        return {
            "right_distance_m": right_distances,
            "left_distance_m": left_distances,
        }


def aggregate_split(
    records,
    threshold_m,
    minimum_frames,
    maximum_gap,
    window_frames,
):
    result = {
        "frames": int(sum(len(row["right_distance_m"]) for row in records)),
        "sequences": len(records),
        "participants": len({row["participant_id"] for row in records}),
        "objects": sorted({row["object_name"] for row in records}),
        "frame_contact_right": 0,
        "frame_contact_left": 0,
        "frame_contact_both": 0,
        "stable_segments_right": 0,
        "stable_segments_left": 0,
        "stable_bimanual_frames": 0,
        "bimanual_sequences": 0,
        "candidates": [],
    }
    for record in records:
        summary = contact_summary(
            record["right_distance_m"],
            record["left_distance_m"],
            threshold_m,
            minimum_frames,
            maximum_gap,
            window_frames,
        )
        for key in (
            "frame_contact_right",
            "frame_contact_left",
            "frame_contact_both",
            "stable_segments_right",
            "stable_segments_left",
            "stable_bimanual_frames",
        ):
            result[key] += summary[key]
        if summary["has_stable_bimanual_overlap"]:
            result["bimanual_sequences"] += 1
        result["candidates"].extend({
            **candidate,
            "sequence": record["sequence"],
            "participant_id": record["participant_id"],
            "object_name": record["object_name"],
        } for candidate in summary["candidates"])
    result["candidate_count"] = len(result["candidates"])
    result["candidate_sequences"] = len({
        candidate["sequence"] for candidate in result["candidates"]
    })
    result["candidate_participants"] = len({
        candidate["participant_id"] for candidate in result["candidates"]
    })
    result["candidate_objects"] = sorted({
        candidate["object_name"] for candidate in result["candidates"]
    })
    result["candidate_directions"] = {
        direction: sum(
            candidate["direction"] == direction
            for candidate in result["candidates"]
        )
        for direction in ("left_to_right", "right_to_left")
    }
    return result


def pilot_gate(train, val):
    checks = {
        "train_candidates_ge_10": train["candidate_count"] >= 10,
        "train_candidate_participants_ge_3": (
            train["candidate_participants"] >= 3
        ),
        "train_both_directions_ge_2": all(
            count >= 2
            for count in train["candidate_directions"].values()
        ),
        "train_candidate_objects_ge_3": (
            len(train["candidate_objects"]) >= 3
        ),
        "train_bimanual_sequences_ge_1": (
            train["bimanual_sequences"] >= 1
        ),
        "val_bimanual_sequences_ge_1": val["bimanual_sequences"] >= 1,
        "val_candidates_ge_1": val["candidate_count"] >= 1,
    }
    checks["pilot_overall"] = all(checks.values())
    checks["official_test_available"] = False
    checks["final_benchmark_ready"] = False
    return checks


def write_candidates(path, by_threshold):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for threshold, split_rows in sorted(by_threshold.items()):
            for split, row in sorted(split_rows.items()):
                for candidate in row["candidates"]:
                    handle.write(json.dumps({
                        "threshold_m": threshold,
                        "split": split,
                        **candidate,
                    }, sort_keys=True) + "\n")


def write_trajectories(path, records):
    order = sorted(records, key=lambda row: row["sequence"])
    maximum_length = max(len(row["right_distance_m"]) for row in order)
    count = len(order)
    right = np.full((count, maximum_length), np.nan, dtype=np.float32)
    left = np.full((count, maximum_length), np.nan, dtype=np.float32)
    lengths = np.zeros(count, dtype=np.int32)
    keys = []
    splits = []
    participants = []
    objects = []
    for index, row in enumerate(order):
        length = len(row["right_distance_m"])
        lengths[index] = length
        right[index, :length] = row["right_distance_m"]
        left[index, :length] = row["left_distance_m"]
        keys.append(row["sequence"])
        splits.append(row["split"])
        participants.append(row["participant_id"])
        objects.append(row["object_name"])
    np.savez_compressed(
        path,
        keys=np.asarray(keys),
        splits=np.asarray(splits),
        participants=np.asarray(participants),
        objects=np.asarray(objects),
        lengths=lengths,
        right_distance_m=right,
        left_distance_m=left,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    split_root = args.arctic_root / "splits_json"
    protocol, assignment = load_official_splits(split_root)
    sequence_paths = sorted(
        (args.arctic_root / "raw_seqs").glob("*/*.mano.npy")
    )
    if args.max_sequences is not None:
        sequence_paths = sequence_paths[:args.max_sequences]

    available_keys = {
        f"{path.parent.name}/{path.name.replace('.mano.npy', '')}"
        for path in sequence_paths
    }
    missing_test = [
        key for key in protocol["test"] if key not in available_keys
    ]
    print(
        f"found {len(sequence_paths)} sequences; "
        f"official test missing {len(missing_test)}",
        flush=True,
    )

    runner = ArcticRunner(args)
    failures = []
    for index, sequence_path in enumerate(sequence_paths, start=1):
        key = (
            f"{sequence_path.parent.name}/"
            f"{sequence_path.name.replace('.mano.npy', '')}"
        )
        output_path = trajectory_dir / sequence_to_filename(key)
        if output_path.exists():
            print(f"[{index}/{len(sequence_paths)}] skip {key}", flush=True)
            continue
        try:
            trajectory = runner.process(sequence_path)
            np.savez_compressed(
                output_path,
                right_distance_m=trajectory["right_distance_m"],
                left_distance_m=trajectory["left_distance_m"],
            )
            print(
                f"[{index}/{len(sequence_paths)}] {key} "
                f"frames={len(trajectory['right_distance_m'])} "
                f"right_min={float(trajectory['right_distance_m'].min()):.6f} "
                f"left_min={float(trajectory['left_distance_m'].min()):.6f}",
                flush=True,
            )
        except Exception:
            failures.append({
                "sequence": key,
                "traceback": traceback.format_exc(),
            })
            print(f"[{index}/{len(sequence_paths)}] FAILED {key}", flush=True)

    records = []
    for sequence_path in sequence_paths:
        key = (
            f"{sequence_path.parent.name}/"
            f"{sequence_path.name.replace('.mano.npy', '')}"
        )
        output_path = trajectory_dir / sequence_to_filename(key)
        if not output_path.exists():
            continue
        arrays = np.load(output_path)
        records.append({
            "sequence": key,
            "split": assignment.get(key, "unassigned"),
            "participant_id": key.split("/", 1)[0],
            "object_name": sequence_path.name.split("_", 1)[0],
            "right_distance_m": arrays["right_distance_m"],
            "left_distance_m": arrays["left_distance_m"],
        })

    if not records:
        raise RuntimeError("no trajectories were produced")
    if len(failures) > max(1, math.ceil(0.02 * len(sequence_paths))):
        raise RuntimeError(f"too many failed sequences: {len(failures)}")

    split_records = {
        split: [row for row in records if row["split"] == split]
        for split in ("train", "val", "test", "unassigned")
    }
    by_threshold = {}
    for threshold in THRESHOLDS_M:
        by_threshold[threshold] = {
            split: aggregate_split(
                rows,
                threshold,
                args.min_contact_frames,
                args.bridge_gap_frames,
                args.handover_window_frames,
            )
            for split, rows in split_records.items()
        }
    gate = pilot_gate(
        by_threshold[PRIMARY_THRESHOLD_M]["train"],
        by_threshold[PRIMARY_THRESHOLD_M]["val"],
    )
    result = {
        "config": {
            "primary_threshold_m": PRIMARY_THRESHOLD_M,
            "thresholds_m": list(THRESHOLDS_M),
            "minimum_contact_frames": args.min_contact_frames,
            "bridge_gap_frames": args.bridge_gap_frames,
            "handover_window_frames": args.handover_window_frames,
        },
        "availability": {
            "sequences_processed": len(records),
            "official_split_counts": {
                split: len(protocol[split])
                for split in ("train", "val", "test")
            },
            "available_split_counts": {
                split: len(rows)
                for split, rows in split_records.items()
            },
            "missing_official_test_sequences": len(missing_test),
            "official_test_subjects": sorted({
                key.split("/", 1)[0] for key in protocol["test"]
            }),
        },
        "by_threshold": {
            f"{threshold:.4f}": split_rows
            for threshold, split_rows in by_threshold.items()
        },
        "gate": gate,
        "failures": failures,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_trajectories(
        args.output_dir / "distance_trajectories.npz",
        records,
    )
    write_candidates(
        args.output_dir / "role_switch_candidates.jsonl.gz",
        by_threshold,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
