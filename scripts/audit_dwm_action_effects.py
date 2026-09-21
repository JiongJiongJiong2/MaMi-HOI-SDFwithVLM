#!/usr/bin/env python3
"""Audit action-dependent object response in the DWM D0-B dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--val-npz", type=Path, required=True)
    parser.add_argument("--test-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1e-2)
    return parser.parse_args()


def load_split(path, horizon):
    with np.load(path, allow_pickle=False) as data:
        return {
            "state": np.asarray(data["states"][:, 0], dtype=np.float64),
            "action": np.asarray(
                data["actions"][:, :horizon],
                dtype=np.float64,
            ),
            "pose": np.asarray(data["object_pose"], dtype=np.float64),
            "mode": np.asarray(data["contact_mode"], dtype=np.int64),
            "object_id": np.asarray(data["object_id"]),
            "reset_id": np.asarray(data["reset_id"], dtype=np.int64),
            "branch_name": np.asarray(data["branch_name"]),
        }


def object_delta(split, horizon):
    return (
        split["pose"][:, horizon, :]
        - split["pose"][:, 0, :]
    )


def center_by_reset(split, target):
    centered = target.copy()
    groups = {}
    for row, (object_id, reset_id) in enumerate(zip(
        split["object_id"],
        split["reset_id"],
    )):
        groups.setdefault((str(object_id), int(reset_id)), []).append(row)
    for rows in groups.values():
        centered[rows] -= centered[rows].mean(axis=0, keepdims=True)
    return centered, groups


def macro_f1(prediction, target, class_count):
    values = []
    for class_index in range(class_count):
        truth = target == class_index
        predicted = prediction == class_index
        tp = int(np.sum(truth & predicted))
        fp = int(np.sum(~truth & predicted))
        fn = int(np.sum(truth & ~predicted))
        denominator = 2 * tp + fp + fn
        values.append(2.0 * tp / denominator if denominator else 0.0)
    return float(np.mean(values))


def ridge_fit_predict(
    train_x,
    train_y,
    test_x,
    ridge,
):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_x = (train_x - mean) / scale
    test_x = (test_x - mean) / scale
    train_x = np.concatenate([
        np.ones((train_x.shape[0], 1)),
        train_x,
    ], axis=1)
    test_x = np.concatenate([
        np.ones((test_x.shape[0], 1)),
        test_x,
    ], axis=1)
    gram = train_x.T @ train_x
    penalty = np.eye(gram.shape[0]) * ridge
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + penalty,
        train_x.T @ train_y,
    )
    return test_x @ coefficients


def state_features(split):
    state = split["state"]
    return np.concatenate([
        state[:, :15],
        state[:, 30:75],
        state[:, 120:168],
    ], axis=1)


def action_features(split, mode, seed):
    action = split["action"]
    if mode == "true":
        selected = action
    elif mode == "zero":
        selected = np.zeros_like(action)
    elif mode == "shuffled":
        generator = np.random.default_rng(seed)
        selected = action[generator.permutation(action.shape[0])]
    else:
        raise ValueError(f"unsupported action mode: {mode}")
    return np.concatenate([
        selected.mean(axis=1),
        selected[:, 0],
        selected[:, -1],
    ], axis=1)


def branch_features(split, branches, mode, seed):
    index = {branch: column for column, branch in enumerate(branches)}
    rows = np.asarray([
        index[str(branch)] for branch in split["branch_name"]
    ])
    features = np.zeros((rows.shape[0], len(branches)))
    if mode == "true":
        features[np.arange(rows.shape[0]), rows] = 1.0
    elif mode == "shuffled":
        generator = np.random.default_rng(seed)
        shuffled = generator.permutation(rows)
        features[np.arange(rows.shape[0]), shuffled] = 1.0
    elif mode != "zero":
        raise ValueError(f"unsupported branch mode: {mode}")
    return features


def score_arm(
    train_split,
    test_split,
    target_mode,
    action_mode,
    seed,
    horizon,
    ridge,
):
    train_target = object_delta(train_split, horizon)
    test_target = object_delta(test_split, horizon)
    if target_mode == "centered":
        train_target, _ = center_by_reset(train_split, train_target)
        test_target, _ = center_by_reset(test_split, test_target)
    train_x = np.concatenate([
        state_features(train_split),
        action_features(train_split, action_mode, seed),
    ], axis=1)
    test_x = np.concatenate([
        state_features(test_split),
        action_features(test_split, action_mode, seed + 1),
    ], axis=1)
    prediction = ridge_fit_predict(
        train_x,
        train_target,
        test_x,
        ridge,
    )
    return {
        "translation_l1": float(
            np.mean(np.abs(prediction[:, :3] - test_target[:, :3]))
        ),
        "rotation_l1": float(
            np.mean(np.abs(prediction[:, 3:9] - test_target[:, 3:9]))
        ),
        "all9_l1": float(np.mean(np.abs(prediction - test_target))),
    }


def score_branch_arm(
    train_split,
    test_split,
    branches,
    target_mode,
    branch_mode,
    seed,
    horizon,
    ridge,
    include_state,
):
    train_target = object_delta(train_split, horizon)
    test_target = object_delta(test_split, horizon)
    if target_mode == "centered":
        train_target, _ = center_by_reset(train_split, train_target)
        test_target, _ = center_by_reset(test_split, test_target)
    train_parts = [
        branch_features(
            train_split,
            branches,
            branch_mode,
            seed,
        ),
    ]
    test_parts = [
        branch_features(
            test_split,
            branches,
            branch_mode,
            seed + 1,
        ),
    ]
    if include_state:
        train_parts.insert(0, state_features(train_split))
        test_parts.insert(0, state_features(test_split))
    prediction = ridge_fit_predict(
        np.concatenate(train_parts, axis=1),
        train_target,
        np.concatenate(test_parts, axis=1),
        ridge,
    )
    return {
        "translation_l1": float(
            np.mean(np.abs(prediction[:, :3] - test_target[:, :3]))
        ),
        "rotation_l1": float(
            np.mean(np.abs(prediction[:, 3:9] - test_target[:, 3:9]))
        ),
        "all9_l1": float(np.mean(np.abs(prediction - test_target))),
    }


def score_tree_branch_arm(
    train_split,
    test_split,
    branches,
    branch_mode,
    seed,
    horizon,
):
    from sklearn.ensemble import ExtraTreesRegressor

    train_target = object_delta(train_split, horizon)
    test_target = object_delta(test_split, horizon)
    train_target, _ = center_by_reset(train_split, train_target)
    test_target, _ = center_by_reset(test_split, test_target)
    train_x = np.concatenate([
        state_features(train_split),
        branch_features(
            train_split,
            branches,
            branch_mode,
            seed,
        ),
    ], axis=1)
    test_x = np.concatenate([
        state_features(test_split),
        branch_features(
            test_split,
            branches,
            branch_mode,
            seed + 1,
        ),
    ], axis=1)
    model = ExtraTreesRegressor(
        n_estimators=240,
        min_samples_leaf=4,
        max_features=0.7,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(train_x, train_target)
    prediction = model.predict(test_x)
    return {
        "translation_l1": float(
            np.mean(np.abs(prediction[:, :3] - test_target[:, :3]))
        ),
        "rotation_l1": float(
            np.mean(np.abs(prediction[:, 3:9] - test_target[:, 3:9]))
        ),
        "all9_l1": float(np.mean(np.abs(prediction - test_target))),
    }


def split_reset_holdout(split):
    mask = split["reset_id"] % 5 != 0
    inverse = ~mask
    return {
        key: value[mask] if isinstance(value, np.ndarray) else value
        for key, value in split.items()
    }, {
        key: value[inverse] if isinstance(value, np.ndarray) else value
        for key, value in split.items()
    }


def excitation_summary(split, horizon):
    target = object_delta(split, horizon)
    centered, groups = center_by_reset(split, target)
    index_by_group = groups
    branch_order = sorted(set(str(value) for value in split["branch_name"]))
    branch_to_index = {
        branch: index for index, branch in enumerate(branch_order)
    }
    range_translation = []
    range_rotation = []
    centered_translation = []
    centered_rotation = []
    singular_ratios = []
    for rows in index_by_group.values():
        rows = sorted(
            rows,
            key=lambda row: branch_to_index[str(split["branch_name"][row])],
        )
        response = target[rows]
        response_centered = centered[rows]
        translation_axes = response[:, :3]
        rotation_axes = response[:, 3:9]
        range_translation.append(float(np.mean(np.ptp(
            translation_axes,
            axis=0,
        ))))
        range_rotation.append(float(np.mean(np.ptp(
            rotation_axes,
            axis=0,
        ))))
        centered_translation.append(float(np.mean(
            np.linalg.norm(response_centered[:, :3], axis=1)
        )))
        centered_rotation.append(float(np.mean(
            np.linalg.norm(response_centered[:, 3:9], axis=1)
        )))
        singular_values = np.linalg.svd(
            response_centered,
            compute_uv=False,
        )
        if singular_values[0] > 1e-12:
            singular_ratios.append(float(
                singular_values[1] / singular_values[0]
            ))
    return {
        "resets": len(index_by_group),
        "branches": branch_order,
        "mean_per_axis_translation_range_m": float(np.mean(
            range_translation
        )),
        "mean_per_axis_rotation_range_6d": float(np.mean(
            range_rotation
        )),
        "mean_centered_translation_norm_m": float(np.mean(
            centered_translation
        )),
        "mean_centered_rotation_norm_6d": float(np.mean(
            centered_rotation
        )),
        "median_second_to_first_singular_ratio": float(np.median(
            singular_ratios
        )),
    }


def branch_mode_summary(split, horizon):
    result = {}
    for branch in sorted(set(str(value) for value in split["branch_name"])):
        mask = split["branch_name"] == branch
        branch_split = subset(split, mask)
        result[branch] = {
            "h8_translation_mean_abs": np.mean(
                np.abs(object_delta(branch_split, horizon)[:, :3])
            ).tolist(),
            "h8_rotation_mean_abs": np.mean(
                np.abs(object_delta(branch_split, horizon)[:, 3:9])
            ).tolist(),
            "mode_counts": {
                str(mode): int(np.sum(split["mode"][mask, horizon] == mode))
                for mode in sorted(set(split["mode"][:, horizon].tolist()))
            },
        }
    return result


def subset(split, mask):
    return {
        key: value[mask] if isinstance(value, np.ndarray) else value
        for key, value in split.items()
    }


def main():
    args = parse_args()
    train = load_split(args.train_npz, args.horizon)
    val = load_split(args.val_npz, args.horizon)
    test = load_split(args.test_npz, args.horizon)
    train_fit, train_holdout = split_reset_holdout(train)
    report = {
        "dataset": {
            "train_rows": int(train["state"].shape[0]),
            "val_rows": int(val["state"].shape[0]),
            "test_rows": int(test["state"].shape[0]),
            "horizon": args.horizon,
        },
        "excitation": {
            "train": excitation_summary(train, args.horizon),
            "val": excitation_summary(val, args.horizon),
            "test": excitation_summary(test, args.horizon),
        },
        "branch_modes": {
            "train": branch_mode_summary(train, args.horizon),
            "test": branch_mode_summary(test, args.horizon),
        },
        "ridge_response_prediction": {},
        "ridge_branch_prediction": {},
        "tree_branch_prediction": {},
    }
    for target_mode in ("absolute", "centered"):
        for split_name, split in (
            ("reset_holdout", train_holdout),
            ("object_held_out_val", val),
            ("object_held_out_test", test),
        ):
            for action_mode in ("true", "shuffled", "zero"):
                key = f"{target_mode}/{split_name}/{action_mode}"
                report["ridge_response_prediction"][key] = score_arm(
                    train_fit,
                    split,
                    target_mode,
                    action_mode,
                    seed=1100,
                    horizon=args.horizon,
                    ridge=args.ridge,
                )
    branches = sorted(set(str(value) for value in train["branch_name"]))
    for target_mode in ("absolute", "centered"):
        for split_name, split in (
            ("reset_holdout", train_holdout),
            ("object_held_out_val", val),
            ("object_held_out_test", test),
        ):
            for include_state in (False, True):
                state_label = "state_plus_branch" if include_state else "branch_only"
                for branch_mode in ("true", "shuffled", "zero"):
                    key = (
                        f"{target_mode}/{split_name}/{state_label}/"
                        f"{branch_mode}"
                    )
                    report["ridge_branch_prediction"][key] = (
                        score_branch_arm(
                            train_fit,
                            split,
                            branches,
                            target_mode,
                            branch_mode,
                            seed=3300,
                            horizon=args.horizon,
                            ridge=args.ridge,
                            include_state=include_state,
                        )
                    )
    for split_name, split in (
        ("reset_holdout", train_holdout),
        ("object_held_out_test", test),
    ):
        for branch_mode in ("true", "shuffled", "zero"):
            key = f"{split_name}/{branch_mode}"
            report["tree_branch_prediction"][key] = (
                score_tree_branch_arm(
                    train_fit,
                    split,
                    branches,
                    branch_mode,
                    seed=4400,
                    horizon=args.horizon,
                )
            )

    train_target = object_delta(train_fit, args.horizon)
    train_mode = train_fit["mode"][:, args.horizon]
    feature = np.concatenate([
        state_features(train_fit),
        action_features(train_fit, "true", 2200),
    ], axis=1)
    mean = feature.mean(axis=0)
    scale = feature.std(axis=0)
    scale[scale < 1e-8] = 1.0
    feature = (feature - mean) / scale
    feature = np.concatenate([
        np.ones((feature.shape[0], 1)),
        feature,
    ], axis=1)
    gram = feature.T @ feature + args.ridge * np.eye(feature.shape[1])
    gram[0, 0] -= args.ridge
    coefficients = np.linalg.solve(
        gram,
        feature.T @ train_target,
    )
    mode_report = {}
    for name, eval_split in (
        ("reset_holdout", train_holdout),
        ("val", val),
        ("test", test),
    ):
        eval_x = np.concatenate([
            state_features(eval_split),
            action_features(eval_split, "true", 2201),
        ], axis=1)
        eval_x = (eval_x - mean) / scale
        eval_x = np.concatenate([
            np.ones((eval_x.shape[0], 1)),
            eval_x,
        ], axis=1)
        predicted_response = eval_x @ coefficients
        predicted_mode = np.argmax(
            predicted_response[:, :4],
            axis=1,
        )
        truth_mode = eval_split["mode"][:, args.horizon]
        mode_report[name] = {
            "accuracy": float(np.mean(predicted_mode == truth_mode)),
            "macro_f1": macro_f1(predicted_mode, truth_mode, 4),
        }
    report["ridge_mode_from_object_response"] = mode_report

    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "excitation": report["excitation"],
        "ridge": report["ridge_response_prediction"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
