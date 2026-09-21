#!/usr/bin/env python3
"""Audit whether an OakInk C0 event dataset supports future conditioning."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

try:
    from scripts.build_oakink_c0_event_dataset import (
        MINIMUM_WINDOW_FRAMES,
        object_diameter,
    )
except ModuleNotFoundError:
    from build_oakink_c0_event_dataset import (  # noqa: E402
        MINIMUM_WINDOW_FRAMES,
        object_diameter,
    )


CONTACT_THRESHOLD_M = 0.005
NEAR_THRESHOLD_M = 0.020
MINIMUM_COMPLETE_EVENTS = 60
MINIMUM_SPLIT_EVENTS = 12
MINIMUM_CONTACT_FRACTION = 0.8
MINIMUM_OBJECT_GROUPS = 20
MINIMUM_CENTROID_SEPARATION_FRACTION = 0.05
MINIMUM_HARD_CONTROL_EVENTS = 20
GEOMETRY_TOLERANCE_M = 1e-5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_npz(path):
    with np.load(path, allow_pickle=False) as arrays:
        return {
            name: np.asarray(arrays[name])
            for name in arrays.files
        }


def load_objects(path):
    arrays = load_npz(path)
    result = []
    for index, object_id in enumerate(arrays["object_ids"]):
        result.append({
            "object_id": str(object_id),
            "object_name": str(arrays["object_names"][index]),
            "diameter": float(arrays["diameters"][index]),
            "vertices": np.asarray(
                arrays[f"vertices_{index}"],
                dtype=np.float64,
            ),
        })
    return result


def reconstructed_distance(
    object_vertices,
    transform,
    hand_vertices,
):
    homogeneous = np.concatenate(
        (
            object_vertices,
            np.ones((len(object_vertices), 1), dtype=np.float64),
        ),
        axis=1,
    )
    transformed = (transform @ homogeneous.T).T[:, :3]
    return float(
        cKDTree(transformed).query(
            hand_vertices,
            k=1,
            workers=-1,
        )[0].min()
    )


def centroid(indices, object_vertices):
    if len(indices) == 0:
        return np.full(3, np.nan, dtype=np.float64)
    return object_vertices[indices].mean(axis=0)


def main():
    args = parse_args()
    events = load_npz(args.dataset_dir / "events.npz")
    targets = load_npz(args.dataset_dir / "targets.npz")
    objects = load_objects(args.dataset_dir / "objects.npz")

    event_count = len(events["sequences"])
    valid_counts = events["valid"].sum(axis=1)
    complete_mask = valid_counts >= MINIMUM_WINDOW_FRAMES
    split_complete = {
        split: int(
            np.sum(complete_mask & (events["splits"] == split))
        )
        for split in ("train", "val", "test")
    }

    current_offsets = targets["current_offsets"]
    target_offsets = targets["target_offsets"]
    current_sizes = np.diff(current_offsets)
    target_sizes = np.diff(target_offsets)
    nonempty_current = current_sizes > 0
    nonempty_target = target_sizes > 0

    geometry_errors = []
    reconstruction_checked = 0
    per_event = []
    object_events = defaultdict(list)
    for event_index in range(event_count):
        object_index = int(events["object_indices"][event_index])
        object_row = objects[object_index]
        object_vertices = object_row["vertices"]
        hand_indices = targets["target_indices"][
            target_offsets[event_index]:target_offsets[event_index + 1]
        ]
        target_centroid = centroid(hand_indices, object_vertices)
        object_events[object_index].append(event_index)

        hard_control = False
        valid_positions = np.flatnonzero(events["valid"][event_index])
        for position in valid_positions:
            if position >= 30:
                break
            if (
                events["giver_distance_m"][event_index, position]
                <= CONTACT_THRESHOLD_M
                and events["receiver_distance_m"][event_index, position]
                <= NEAR_THRESHOLD_M
            ):
                hard_control = True
                break

        if reconstruction_checked < 12 and complete_mask[event_index]:
            for position in valid_positions:
                giver_actual = float(
                    events["giver_distance_m"][event_index, position]
                )
                giver_reconstructed = reconstructed_distance(
                    object_vertices,
                    events["object_transforms"][event_index, position],
                    events["giver_hand_vertices"][event_index, position],
                )
                receiver_actual = float(
                    events["receiver_distance_m"][event_index, position]
                )
                receiver_reconstructed = reconstructed_distance(
                    object_vertices,
                    events["object_transforms"][event_index, position],
                    events["receiver_hand_vertices"][
                        event_index, position
                    ],
                )
                geometry_errors.extend([
                    abs(giver_actual - giver_reconstructed),
                    abs(receiver_actual - receiver_reconstructed),
                ])
            reconstruction_checked += 1

        per_event.append({
            "event_index": event_index,
            "sequence": str(events["sequences"][event_index]),
            "split": str(events["splits"][event_index]),
            "valid_frames": int(valid_counts[event_index]),
            "complete": bool(complete_mask[event_index]),
            "current_contact_vertices": int(current_sizes[event_index]),
            "receiver_target_vertices": int(target_sizes[event_index]),
            "hard_control": bool(hard_control),
            "receiver_target_centroid": (
                target_centroid.tolist()
                if np.isfinite(target_centroid).all()
                else None
            ),
        })

    contrast_groups = []
    for object_index, event_indices in sorted(object_events.items()):
        if len(event_indices) < 2:
            continue
        object_row = objects[object_index]
        diameter = object_row["diameter"]
        centroids = []
        for event_index in event_indices:
            indices = targets["target_indices"][
                target_offsets[event_index]:target_offsets[event_index + 1]
            ]
            centroids.append(
                centroid(indices, object_row["vertices"])
            )
        maximum = 0.0
        maximum_pair = None
        for left in range(len(centroids)):
            for right in range(left + 1, len(centroids)):
                if not (
                    np.isfinite(centroids[left]).all()
                    and np.isfinite(centroids[right]).all()
                ):
                    continue
                distance = float(
                    np.linalg.norm(centroids[left] - centroids[right])
                )
                if distance > maximum:
                    maximum = distance
                    maximum_pair = (
                        event_indices[left],
                        event_indices[right],
                    )
        fraction = maximum / max(diameter, 1e-12)
        contrast_groups.append({
            "object_index": object_index,
            "object_id": object_row["object_id"],
            "object_name": object_row["object_name"],
            "events": len(event_indices),
            "maximum_centroid_distance_m": maximum,
            "maximum_fraction_of_diameter": fraction,
            "pair": maximum_pair,
            "supports_contrast": (
                fraction >= MINIMUM_CENTROID_SEPARATION_FRACTION
            ),
        })

    contrast_count = int(sum(
        row["supports_contrast"] for row in contrast_groups
    ))
    hard_control_count = int(sum(
        row["hard_control"] for row in per_event if row["complete"]
    ))
    contact_fraction = float(
        np.mean(
            nonempty_current[complete_mask]
            & nonempty_target[complete_mask]
        )
    ) if complete_mask.any() else 0.0
    geometry_max_error = (
        float(max(geometry_errors)) if geometry_errors else float("inf")
    )

    checks = {
        "complete_events_ge_60": (
            int(complete_mask.sum()) >= MINIMUM_COMPLETE_EVENTS
        ),
        "each_split_ge_12": all(
            value >= MINIMUM_SPLIT_EVENTS
            for value in split_complete.values()
        ),
        "geometry_reconstruction": (
            reconstruction_checked > 0
            and geometry_max_error <= GEOMETRY_TOLERANCE_M
        ),
        "contact_regions_ge_0_8": (
            contact_fraction >= MINIMUM_CONTACT_FRACTION
        ),
        "same_object_groups_ge_20": (
            contrast_count >= MINIMUM_OBJECT_GROUPS
        ),
        "hard_control_events_ge_20": (
            hard_control_count >= MINIMUM_HARD_CONTROL_EVENTS
        ),
    }
    result = {
        "event_count": event_count,
        "complete_events": int(complete_mask.sum()),
        "split_complete_events": split_complete,
        "valid_frame_range": [
            int(valid_counts.min()),
            int(valid_counts.max()),
        ],
        "current_contact_events": int(nonempty_current.sum()),
        "receiver_target_events": int(nonempty_target.sum()),
        "contact_region_fraction": contact_fraction,
        "same_object_groups": len(contrast_groups),
        "same_object_contrast_groups": contrast_count,
        "hard_control_events": hard_control_count,
        "geometry_reconstruction_events": reconstruction_checked,
        "geometry_reconstruction_max_error_m": geometry_max_error,
        "checks": checks,
        "overall": all(checks.values()),
        "contrast_groups": contrast_groups,
        "events": per_event,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "event_count": event_count,
        "complete_events": int(complete_mask.sum()),
        "split_complete_events": split_complete,
        "contact_region_fraction": contact_fraction,
        "same_object_contrast_groups": contrast_count,
        "hard_control_events": hard_control_count,
        "geometry_reconstruction_max_error_m": geometry_max_error,
        "checks": checks,
        "overall": all(checks.values()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
