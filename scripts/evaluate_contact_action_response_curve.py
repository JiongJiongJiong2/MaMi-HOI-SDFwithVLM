#!/usr/bin/env python3
"""Check monotonic contact response as the palm action moves along normal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from manip.world_model.contact_action.features import NORMAL_SLICE
from manip.world_model.contact_action.model import ContactActionTransition
from scripts.evaluate_contact_action_candidate_ranking import (
    cluster_bootstrap_interval,
    first_event_horizons,
)
from scripts.train_contact_action_world_model import load_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(-0.01, -0.005, 0.0, 0.005, 0.01),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def score_at_horizon(probability, horizon, hand_index):
    if horizon <= 0:
        return np.nan
    return float(probability[horizon - 1, hand_index])


def make_curve_candidates(split, alphas, device):
    actions = split["future_actions"].to(device).clone()
    state = split["state_history"][:, -1].to(device)
    normal = state[:, NORMAL_SLICE].reshape(-1, 2, 3)
    candidates = {}
    for hand_index in range(2):
        channel = slice(hand_index * 3, hand_index * 3 + 3)
        direction = normal[:, hand_index, None, :]
        for alpha in alphas:
            candidate = actions.clone()
            candidate[:, :, channel] = (
                candidate[:, :, channel] + float(alpha) * direction
            )
            candidates[(hand_index, float(alpha))] = candidate
    return candidates


def summarize_curve(
    indicators,
    groups,
    bootstrap_samples,
    seed,
):
    indicators = np.asarray(indicators, dtype=np.float64)
    return {
        "count": int(len(indicators)),
        "accuracy": float(indicators.mean()) if len(indicators) else None,
        "ci95": cluster_bootstrap_interval(
            indicators,
            groups,
            bootstrap_samples,
            seed,
        ),
    }


def main():
    args = parse_args()
    alphas = np.asarray(args.alphas, dtype=np.float64)
    if len(alphas) < 3 or not np.any(alphas < 0) or not np.any(alphas > 0):
        raise ValueError("alphas must include negative and positive values")

    split = load_split(args.val_npz)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = ContactActionTransition(
        hidden_size=checkpoint["args"]["hidden_size"],
        residual_scale=checkpoint["args"].get("residual_scale", 0.0),
        residual_mask=checkpoint["args"].get("residual_mask", "palm"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    current_contact = (
        split["state_history"][:, -1, 32:34].numpy() >= 0.5
    )
    target_contact = split["future_states"][:, :, 32:34].numpy() >= 0.5
    onset_horizons, release_horizons = first_event_horizons(
        current_contact,
        target_contact,
    )
    candidates = make_curve_candidates(split, alphas, args.device)
    sequence_index = split["sequence_index"].numpy()

    onset_probability = {}
    release_probability = {}
    with torch.no_grad():
        for key, actions in candidates.items():
            hand_index, alpha = key
            output = model.rollout(
                split["state_history"].to(args.device),
                split["action_history"].to(args.device),
                actions,
                teacher_states=None,
                teacher_forcing_ratio=0.0,
            )
            onset_probability[key] = (
                torch.sigmoid(output["onset_logits"]).cpu().numpy()
            )
            release_probability[key] = (
                torch.sigmoid(output["release_logits"]).cpu().numpy()
            )

    negative_alpha = float(alphas.min())
    positive_alpha = float(alphas.max())
    zero_alpha = float(alphas[np.argmin(np.abs(alphas))])

    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "alphas": alphas.tolist(),
        "bootstrap_samples": int(args.bootstrap_samples),
        "hands": {},
    }
    for hand_index, hand_name in enumerate(("left", "right")):
        result["hands"][hand_name] = {}
        for horizon_name, horizon_value in (
            ("all", None),
            ("h1", 1),
            ("h2", 2),
            ("h4", 4),
            ("h8", 8),
        ):
            onset_mask = onset_horizons[:, hand_index] > 0
            release_mask = release_horizons[:, hand_index] > 0
            if horizon_value is not None:
                onset_mask &= onset_horizons[:, hand_index] == horizon_value
                release_mask &= (
                    release_horizons[:, hand_index] == horizon_value
                )

            onset_values = np.asarray([
                score_at_horizon(
                    onset_probability[(hand_index, alpha)][sample_index],
                    int(onset_horizons[sample_index, hand_index]),
                    hand_index,
                )
                for sample_index in np.flatnonzero(onset_mask)
                for alpha in (negative_alpha, zero_alpha, positive_alpha)
            ]).reshape(-1, 3)
            release_values = np.asarray([
                score_at_horizon(
                    release_probability[(hand_index, alpha)][sample_index],
                    int(release_horizons[sample_index, hand_index]),
                    hand_index,
                )
                for sample_index in np.flatnonzero(release_mask)
                for alpha in (negative_alpha, zero_alpha, positive_alpha)
            ]).reshape(-1, 3)

            onset_monotonic = (
                (onset_values[:, 0] > onset_values[:, 1])
                & (onset_values[:, 1] > onset_values[:, 2])
            )
            release_monotonic = (
                (release_values[:, 2] > release_values[:, 1])
                & (release_values[:, 1] > release_values[:, 0])
            )
            result["hands"][hand_name][horizon_name] = {
                "onset": {
                    "count": int(onset_mask.sum()),
                    "monotonic_decrease": summarize_curve(
                        onset_monotonic,
                        sequence_index[onset_mask],
                        args.bootstrap_samples,
                        args.seed,
                    ),
                    "mean_scores": onset_values.mean(axis=0).tolist()
                    if len(onset_values)
                    else None,
                },
                "release": {
                    "count": int(release_mask.sum()),
                    "monotonic_increase": summarize_curve(
                        release_monotonic,
                        sequence_index[release_mask],
                        args.bootstrap_samples,
                        args.seed + 1,
                    ),
                    "mean_scores": release_values.mean(axis=0).tolist()
                    if len(release_values)
                    else None,
                },
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
