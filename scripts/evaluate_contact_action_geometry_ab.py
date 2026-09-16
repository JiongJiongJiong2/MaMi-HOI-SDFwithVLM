#!/usr/bin/env python3
"""Evaluate Stage 2 checkpoints with sequence-grouped paired bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from manip.world_model.contact_action.geometry import build_geometry_bank
from manip.world_model.contact_action.model import ContactActionTransition
from manip.world_model.contact_action.model_geometry import (
    ContactActionGeometryTransition,
)
from scripts.build_contact_action_dataset import (
    TRAIN_OBJECTS,
    VALIDATION_OBJECTS,
)
from scripts.train_contact_action_world_model import (
    binary_auc,
    collect_rollout,
    load_split,
)


ARMS = ("A", "A_plus", "B_shuffle", "B")
HAND_NAMES = ("left", "right")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(11, 12, 13))
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--bootstrap_samples", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_arm_rows(root, seed, arm, args, val_split):
    checkpoint_path = Path(root) / f"seed_{seed}" / arm / "best.pt"
    if not checkpoint_path.is_file():
        return None
    checkpoint = torch.load(
        checkpoint_path,
        map_location=args.device,
    )
    saved = SimpleNamespace(**checkpoint["args"])
    geometry_mode = getattr(saved, "geometry_mode", "none")
    geometry_bank = None
    split = val_split
    if geometry_mode != "none":
        geometry_bank = build_geometry_bank(
            args.data_root,
            TRAIN_OBJECTS + VALIDATION_OBJECTS,
            args.device,
        )
        split = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in val_split.items()
        }
        split["object_index"] = split["object_index"] + len(TRAIN_OBJECTS)
        model = ContactActionGeometryTransition(
            hidden_size=saved.hidden_size,
            residual_scale=saved.residual_scale,
            geometry_output_size=saved.geometry_output_size,
            geometry_patch_grid=saved.geometry_patch_grid,
            geometry_radius_normalized=saved.geometry_radius_normalized,
            geometry_mode=geometry_mode,
            residual_mask=getattr(saved, "residual_mask", "palm"),
        ).to(args.device)
    else:
        model = ContactActionTransition(
            hidden_size=saved.hidden_size,
            residual_scale=saved.residual_scale,
            residual_mask=getattr(saved, "residual_mask", "palm"),
        ).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    indices = torch.arange(split["state_history"].shape[0])
    with torch.no_grad():
        output, targets = collect_rollout(
            model,
            split,
            indices,
            args.horizon,
            teacher_forcing_ratio=0.0,
            device=args.device,
            geometry_bank=geometry_bank,
            geometry_mode=geometry_mode,
            batch_size=args.eval_batch_size,
        )
    current = split["state_history"][:, -1, 32:34].to(args.device)
    target = targets[:, :, 32:34]
    onset_probability = torch.sigmoid(output["onset_logits"])
    release_probability = torch.sigmoid(output["release_logits"])
    sequence_index = split["sequence_index"].numpy()

    rows = {}
    for horizon in range(args.horizon):
        for hand_index, hand_name in enumerate(HAND_NAMES):
            current_contact = current[:, hand_index] >= 0.5
            target_contact = target[:, horizon, hand_index] >= 0.5
            onset_mask = (~current_contact) & target_contact
            stable_off = (~current_contact) & (~target_contact)
            release_mask = current_contact & (~target_contact)
            stable_on = current_contact & target_contact
            onset_union = onset_mask | stable_off
            release_union = release_mask | stable_on
            for event_name, mask, labels, scores in (
                (
                    "onset",
                    onset_union,
                    onset_mask,
                    onset_probability[:, horizon, hand_index],
                ),
                (
                    "release",
                    release_union,
                    release_mask,
                    release_probability[:, horizon, hand_index],
                ),
            ):
                selected = mask.detach().cpu().numpy()
                if not selected.any():
                    continue
                key = f"h{horizon + 1}_{hand_name}_{event_name}"
                rows[key] = {
                    "labels": labels[mask].detach().cpu().numpy(),
                    "scores": scores[mask].detach().cpu().numpy(),
                    "groups": sequence_index[selected],
                }
    return rows


def grouped_bootstrap_auc_delta(
    left_rows,
    right_rows,
    samples,
    seed,
):
    keys = sorted(set(left_rows) & set(right_rows))
    if not keys:
        return {"mean": None, "ci95": None, "event_groups": 0}
    unique_groups = np.unique(left_rows[keys[0]]["groups"])
    group_lookup = {
        key: {
            group: np.flatnonzero(
                left_rows[key]["groups"] == group
            )
            for group in unique_groups
        }
        for key in keys
    }
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        deltas = []
        for key in keys:
            index_parts = [
                group_lookup[key][group]
                for group in sampled_groups
                if len(group_lookup[key][group])
            ]
            if not index_parts:
                continue
            indices = np.concatenate(index_parts)
            labels = left_rows[key]["labels"][indices]
            if labels.min() == labels.max():
                continue
            left_auc = binary_auc(
                left_rows[key]["scores"][indices],
                labels,
            )
            right_auc = binary_auc(
                right_rows[key]["scores"][indices],
                labels,
            )
            if left_auc is None or right_auc is None:
                continue
            deltas.append(left_auc - right_auc)
        draws[draw] = np.mean(deltas) if deltas else np.nan
    draws = draws[np.isfinite(draws)]
    return {
        "mean": float(np.mean(draws)) if len(draws) else None,
        "ci95": (
            [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ]
            if len(draws)
            else None
        ),
        "event_groups": len(keys),
    }


def main():
    args = parse_args()
    val_split = load_split(args.val_npz)
    result = {
        "root": args.root,
        "seeds": list(args.seeds),
        "bootstrap_samples": args.bootstrap_samples,
        "comparisons": {},
        "complete_seeds": [],
    }
    for seed in args.seeds:
        arm_rows = {}
        for arm in ARMS:
            rows = load_arm_rows(
                args.root,
                seed,
                arm,
                args,
                val_split,
            )
            if rows is not None:
                arm_rows[arm] = rows
        if not all(arm in arm_rows for arm in ARMS):
            continue
        result["complete_seeds"].append(seed)
        for left, right in (
            ("B", "A"),
            ("B", "A_plus"),
            ("B", "B_shuffle"),
        ):
            key = f"{left}-{right}"
            result["comparisons"].setdefault(key, {})
            result["comparisons"][key][str(seed)] = (
                grouped_bootstrap_auc_delta(
                    arm_rows[left],
                    arm_rows[right],
                    args.bootstrap_samples,
                    seed=1000 + seed,
                )
            )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
