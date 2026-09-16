#!/usr/bin/env python3
"""Calibrate onset/release thresholds on train and evaluate on validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from manip.world_model.contact_action.features import CONTACT_SLICE
from manip.world_model.contact_action.model import ContactActionTransition
from scripts.train_contact_action_world_model import load_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def rollout_probabilities(model, split, device):
    model.eval()
    with torch.no_grad():
        output = model.rollout(
            split["state_history"].to(device),
            split["action_history"].to(device),
            split["future_actions"].to(device),
            teacher_states=None,
            teacher_forcing_ratio=0.0,
        )
    return {
        "contact": torch.sigmoid(output["contact_logits"]).cpu().numpy(),
        "onset": torch.sigmoid(output["onset_logits"]).cpu().numpy(),
        "release": torch.sigmoid(output["release_logits"]).cpu().numpy(),
    }


def event_masks(split):
    current = split["state_history"][:, -1, CONTACT_SLICE].numpy() >= 0.5
    target = split["future_states"][:, :, CONTACT_SLICE].numpy() >= 0.5
    return {
        "current": current,
        "target": target,
        "onset": (~current[:, None]) & target,
        "stable_off": (~current[:, None]) & (~target),
        "release": current[:, None] & (~target),
        "stable_on": current[:, None] & target,
    }


def best_balanced_threshold(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    if not labels.any() or labels.all():
        return None
    best = None
    for threshold in np.linspace(0.01, 0.99, 99):
        prediction = scores >= threshold
        true_positive = float((prediction & labels).sum())
        true_negative = float((~prediction & ~labels).sum())
        recall = true_positive / max(float(labels.sum()), 1.0)
        specificity = true_negative / max(float((~labels).sum()), 1.0)
        balanced_accuracy = 0.5 * (recall + specificity)
        candidate = (
            balanced_accuracy,
            float(threshold),
            recall,
            specificity,
        )
        if best is None or candidate[0] > best[0]:
            best = candidate
    return {
        "threshold": best[1],
        "train_balanced_accuracy": best[0],
        "train_recall": best[2],
        "train_specificity": best[3],
    }


def metrics_at_threshold(scores, labels, threshold):
    labels = np.asarray(labels, dtype=bool)
    prediction = scores >= threshold
    true_positive = float((prediction & labels).sum())
    true_negative = float((~prediction & ~labels).sum())
    recall = true_positive / max(float(labels.sum()), 1.0)
    specificity = true_negative / max(float((~labels).sum()), 1.0)
    return {
        "threshold": float(threshold),
        "recall": recall,
        "specificity": specificity,
        "false_positive": 1.0 - specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
    }


def main():
    args = parse_args()
    train_split = load_split(args.train_npz)
    val_split = load_split(args.val_npz)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = ContactActionTransition(
        hidden_size=checkpoint["args"]["hidden_size"],
        residual_scale=checkpoint["args"].get("residual_scale", 0.0),
        residual_mask=checkpoint["args"].get("residual_mask", "palm"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    train_probabilities = rollout_probabilities(model, train_split, args.device)
    val_probabilities = rollout_probabilities(model, val_split, args.device)
    train_masks = event_masks(train_split)
    val_masks = event_masks(val_split)

    thresholds = {}
    validation = {}
    for horizon in range(train_masks["target"].shape[1]):
        for hand_index, hand_name in enumerate(("left", "right")):
            for event_name, probability_key, mask_key in (
                ("onset", "onset", "onset"),
                ("release", "release", "release"),
            ):
                if event_name == "onset":
                    union_key = "stable_off"
                else:
                    union_key = "stable_on"
                train_union = (
                    train_masks[mask_key][:, horizon, hand_index]
                    | train_masks[union_key][:, horizon, hand_index]
                )
                val_union = (
                    val_masks[mask_key][:, horizon, hand_index]
                    | val_masks[union_key][:, horizon, hand_index]
                )
                train_labels = train_masks[mask_key][
                    train_union, horizon, hand_index
                ]
                val_labels = val_masks[mask_key][
                    val_union, horizon, hand_index
                ]
                train_scores = train_probabilities[probability_key][
                    train_union, horizon, hand_index
                ]
                val_scores = val_probabilities[probability_key][
                    val_union, horizon, hand_index
                ]
                calibration = best_balanced_threshold(
                    train_scores,
                    train_labels,
                )
                key = f"h{horizon + 1}_{hand_name}_{event_name}"
                if calibration is None:
                    thresholds[key] = None
                    validation[key] = None
                    continue
                thresholds[key] = calibration
                validation[key] = metrics_at_threshold(
                    val_scores,
                    val_labels,
                    calibration["threshold"],
                )

    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "thresholds": thresholds,
        "validation": validation,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
