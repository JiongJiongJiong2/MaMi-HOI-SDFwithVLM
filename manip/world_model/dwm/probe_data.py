"""Data helpers for the DWM P0 probe-ranking pilot."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import OBJECT_CONFIG_BY_ID


SPLITS = ("train", "val", "test")


def load_split(path):
    return {
        key: np.asarray(value)
        for key, value in np.load(
            Path(path),
            allow_pickle=False,
        ).items()
    }


def candidate_delta(split):
    initial = split["post_probe_state"][:, :9][:, None]
    final_pose = split["candidate_final_object_pose"]
    return (final_pose - initial).astype(np.float32)


def centered_candidate_delta(split):
    delta = candidate_delta(split)
    return (delta - delta.mean(axis=1, keepdims=True)).astype(np.float32)


def oracle_response_context(split):
    rows = []
    shape_names = ("sphere", "box", "cylinder")
    for object_id in split["object_id"]:
        config = OBJECT_CONFIG_BY_ID[str(object_id)]
        shape = np.asarray([
            1.0 if config.shape == name else 0.0
            for name in shape_names
        ], dtype=np.float32)
        rows.append(np.concatenate([
            np.asarray([
                (config.mass - 0.10) / 0.45,
                (config.friction - 0.45) / 0.55,
            ], dtype=np.float32),
            shape,
        ]))
    return np.stack(rows)


def condition_mask(split, mode=None, length=None):
    mask = np.ones(split["utility"].shape[0], dtype=np.bool_)
    if mode is not None:
        mask &= split["probe_mode"] == mode
    if length is not None:
        mask &= split["probe_length"] == int(length)
    return mask


def group_key(split, row):
    return (
        str(split["object_id"][row]),
        int(split["reset_id"][row]),
        str(split["probe_mode"][row]),
        int(split["probe_length"][row]),
    )


def split_indices(split, indices):
    return {
        key: value[indices] if isinstance(value, np.ndarray) else value
        for key, value in split.items()
    }


def make_paired_conditions(split):
    """Build the fixed/random budget matrix used by the evaluator."""

    result = {}
    for mode in ("none", "fixed", "random"):
        for length in ((0,) if mode == "none" else (1, 2, 4)):
            key = f"{mode}:{length}"
            result[key] = condition_mask(split, mode, length)
    return result
