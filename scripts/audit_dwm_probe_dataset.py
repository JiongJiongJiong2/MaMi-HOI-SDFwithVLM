#!/usr/bin/env python3
"""Audit the DWM D0-C probe dataset gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from manip.world_model.dwm.branches import BRANCH_NAMES
from manip.world_model.dwm.schema import CONTACT_MODE_LABELS


SPLITS = ("train", "val", "test")
PROBE_MODES = ("none", "fixed", "random")


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


def reproducibility(repeat_a, repeat_b):
    if repeat_a is None or repeat_b is None:
        return None
    result = {}
    maximum = 0.0
    for split in SPLITS:
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


def utility_range(data, mask=None):
    utility = data["utility"].astype(np.float64)
    if mask is not None:
        utility = utility[mask]
    ranges = np.max(utility, axis=1) - np.min(utility, axis=1)
    return {
        "groups": int(ranges.shape[0]),
        "range_gt_2mm": int(np.sum(ranges > 0.002)),
        "rate": float(np.mean(ranges > 0.002)),
        "minimum_range": float(np.min(ranges)),
        "median_range": float(np.median(ranges)),
    }


def mode_counts(data):
    values = np.concatenate([
        data["candidate_contact_mode"].reshape(-1),
    ])
    return {
        label: int(np.sum(values == index))
        for index, label in enumerate(CONTACT_MODE_LABELS)
    }


def rank_consistency(data):
    utility = data["utility"].astype(np.float64)
    rank = data["rank"]
    order = np.argsort(-utility, axis=1, kind="stable")
    expected = np.empty_like(order)
    expected[
        np.arange(order.shape[0])[:, None],
        order,
    ] = np.arange(order.shape[1])[None, :]
    return bool(np.array_equal(expected, rank))


def target_not_candidate(data, tolerance=1e-7):
    target = data["target_action"].astype(np.float64)
    candidate = data["candidate_action"].astype(np.float64)
    minimum = float("inf")
    for index in range(candidate.shape[1]):
        difference = np.max(
            np.abs(target - candidate[:, index]),
            axis=(1, 2),
        )
        minimum = min(minimum, float(np.min(difference)))
    return minimum > tolerance, minimum


def probe_nesting(data):
    groups = {}
    for index, key in enumerate(zip(
        data["object_id"],
        data["reset_id"],
        data["probe_mode"],
    )):
        groups.setdefault(
            (str(key[0]), int(key[1]), str(key[2])),
            [],
        ).append(index)
    valid = True
    checked = 0
    for rows in groups.values():
        rows = sorted(rows, key=lambda index: int(data["probe_length"][index]))
        for left, right in zip(rows, rows[1:]):
            left_length = int(data["probe_length"][left])
            right_length = int(data["probe_length"][right])
            if right_length <= left_length:
                valid = False
                break
            valid = valid and bool(np.all(
                data["probe_mask"][right, :right_length]
            ))
            valid = valid and bool(np.all(
                ~data["probe_mask"][right, right_length:]
            ))
            valid = valid and bool(np.allclose(
                data["probe_action"][left, :left_length],
                data["probe_action"][right, :left_length],
                atol=0.0,
                rtol=0.0,
            ))
            valid = valid and bool(np.allclose(
                data["probe_state"][left, :left_length + 1],
                data["probe_state"][right, :left_length + 1],
                atol=0.0,
                rtol=0.0,
            ))
            checked += 1
    return valid, checked


def schema_checks(data):
    expected_shapes = {
        "initial_state": (None, 168),
        "probe_action": (None, 4, 51),
        "probe_state": (None, 5, 168),
        "probe_mask": (None, 4),
        "post_probe_state": (None, 168),
        "candidate_action": (None, 13, 8, 51),
        "candidate_final_object_pose": (None, 13, 9),
        "candidate_contact_mode": (None, 13, 9),
        "candidate_contact_impulse": (None, 13, 8, 3),
        "target_action": (None, 8, 51),
        "target_translation": (None, 3),
        "utility": (None, 13),
        "rank": (None, 13),
    }
    checks = {}
    groups = data["utility"].shape[0]
    for key, shape in expected_shapes.items():
        checks[f"shape_{key}"] = data[key].shape == (groups, *shape[1:])
    checks["branch_names"] = bool(np.array_equal(
        data["branch_names"],
        np.asarray(BRANCH_NAMES),
    ))
    checks["finite_inputs"] = bool(np.isfinite(
        data["candidate_action"]
    ).all())
    return checks


def main():
    args = parse_args()
    data = {
        split: load_split(args.dataset_dir, split)
        for split in SPLITS
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
    primary_mask = data["train"]["probe_length"] <= 2
    train_range = utility_range(data["train"], primary_mask)
    budget4_range = utility_range(
        data["train"],
        data["train"]["probe_length"] == 4,
    )
    modes = {
        label: sum(
            mode_counts(data[split])[label]
            for split in SPLITS
        )
        for label in CONTACT_MODE_LABELS
    }
    rank_ok = all(rank_consistency(data[split]) for split in SPLITS)
    target_checks = {
        split: target_not_candidate(data[split])
        for split in SPLITS
    }
    nesting_checks = {
        split: probe_nesting(data[split])
        for split in SPLITS
    }
    schema = {
        split: schema_checks(data[split])
        for split in SPLITS
    }
    repeat = reproducibility(args.repeat_a, args.repeat_b)
    checks = {
        "split_groups_disjoint": all(
            value == 0 for value in overlaps.values()
        ),
        "primary_utility_range_ge_0_8": train_range["rate"] >= 0.8,
        "mode_coverage_ge_100": all(
            count >= 100 for count in modes.values()
        ),
        "rank_consistent": rank_ok,
        "target_distinct": all(
            value[0] for value in target_checks.values()
        ),
        "probe_nested": all(
            value[0] for value in nesting_checks.values()
        ),
        "schema_valid": all(
            all(value.values()) for value in schema.values()
        ),
        "reproducibility_available": repeat is not None,
        "reproducibility_exact": (
            repeat is not None and repeat["maximum"] == 0.0
        ),
    }
    result = {
        "counts": {
            split: int(data[split]["utility"].shape[0])
            for split in SPLITS
        },
        "overlaps": overlaps,
        "utility_range": train_range,
        "budget4_utility_range": budget4_range,
        "mode_counts": modes,
        "rank_consistent": rank_ok,
        "target_checks": {
            split: {
                "valid": value[0],
                "minimum_gap": value[1],
            }
            for split, value in target_checks.items()
        },
        "probe_nesting": {
            split: {
                "valid": value[0],
                "comparisons": value[1],
            }
            for split, value in nesting_checks.items()
        },
        "schema": schema,
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
