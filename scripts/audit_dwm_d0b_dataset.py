#!/usr/bin/env python3
"""Audit DWM D0-B counterfactual dataset gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manip.world_model.dwm.schema import CONTACT_MODE_LABELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--repeat-a", type=Path, default=None)
    parser.add_argument("--repeat-b", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_split(root, split):
    return {
        key: np.asarray(value)
        for key, value in np.load(
            root / f"{split}.npz",
            allow_pickle=False,
        ).items()
    }


def group_keys(data):
    return {
        (str(object_id), int(reset_id))
        for object_id, reset_id in zip(
            data["object_id"],
            data["reset_id"],
        )
    }


def excitation_rate(data):
    groups = {}
    for index, key in enumerate(zip(data["object_id"], data["reset_id"])):
        groups.setdefault((str(key[0]), int(key[1])), []).append(index)
    passing = 0
    pair_counts = []
    for indices in groups.values():
        poses = data["object_pose"][indices, -1]
        distinct_pairs = 0
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                delta = poses[left] - poses[right]
                if (
                    np.linalg.norm(delta[:3]) > 0.002
                    or np.linalg.norm(delta[3:]) > 0.02
                ):
                    distinct_pairs += 1
        pair_counts.append(distinct_pairs)
        passing += distinct_pairs >= 4
    return {
        "groups": len(groups),
        "passing_groups": passing,
        "rate": passing / len(groups) if groups else 0.0,
        "minimum_distinct_pairs": min(pair_counts) if pair_counts else 0,
        "median_distinct_pairs": (
            float(np.median(pair_counts)) if pair_counts else 0.0
        ),
        "maximum_distinct_pairs": max(pair_counts) if pair_counts else 0,
    }


def reproducibility(repeat_a, repeat_b):
    if repeat_a is None or repeat_b is None:
        return None
    result = {}
    maximum = 0.0
    for split in ("train", "val", "test"):
        left = load_split(repeat_a, split)
        right = load_split(repeat_b, split)
        if left.keys() != right.keys():
            raise ValueError(f"repeat schemas differ for {split}")
        split_max = 0.0
        for key in left:
            if left[key].dtype.kind in "fiu":
                difference = float(np.max(np.abs(
                    left[key].astype(np.float64)
                    - right[key].astype(np.float64)
                )))
            else:
                difference = 0.0 if np.array_equal(
                    left[key],
                    right[key],
                ) else float("inf")
            split_max = max(split_max, difference)
        result[split] = split_max
        maximum = max(maximum, split_max)
    result["maximum"] = maximum
    return result


def main():
    args = parse_args()
    data = {
        split: load_split(args.dataset_dir, split)
        for split in ("train", "val", "test")
    }
    group_keys_by_split = {
        split: group_keys(split_data)
        for split, split_data in data.items()
    }
    overlaps = {
        "train_val": len(
            group_keys_by_split["train"] & group_keys_by_split["val"]
        ),
        "train_test": len(
            group_keys_by_split["train"] & group_keys_by_split["test"]
        ),
        "val_test": len(
            group_keys_by_split["val"] & group_keys_by_split["test"]
        ),
    }
    mode_counts = {
        label: int(np.sum(
            np.concatenate([
                data[split]["contact_mode"].reshape(-1)
                for split in ("train", "val", "test")
            ]) == index
        ))
        for index, label in enumerate(CONTACT_MODE_LABELS)
    }
    excitation = excitation_rate(data["train"])
    repeat = reproducibility(args.repeat_a, args.repeat_b)
    shapes_valid = all(
        split_data["states"].shape[1:] == (9, 168)
        and split_data["actions"].shape[1:] == (8, 51)
        for split_data in data.values()
    )
    checks = {
        "shapes_valid": shapes_valid,
        "split_groups_disjoint": all(
            value == 0 for value in overlaps.values()
        ),
        "excitation_rate_ge_0_8": excitation["rate"] >= 0.8,
        "mode_coverage_ge_100": all(
            count >= 100 for count in mode_counts.values()
        ),
        "reproducibility_available": repeat is not None,
        "reproducibility_exact": (
            repeat is not None and repeat["maximum"] == 0.0
        ),
    }
    result = {
        "trajectories": {
            split: int(split_data["states"].shape[0])
            for split, split_data in data.items()
        },
        "groups": {
            split: len(keys)
            for split, keys in group_keys_by_split.items()
        },
        "overlaps": overlaps,
        "excitation": excitation,
        "mode_counts": mode_counts,
        "reproducibility": repeat,
        "checks": checks,
        "overall": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
