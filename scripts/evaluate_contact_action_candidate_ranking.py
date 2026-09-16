#!/usr/bin/env python3
"""Evaluate counterfactual candidate-action ranking for contact transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from manip.world_model.contact_action.features import CONTACT_SLICE, NORMAL_SLICE
from manip.world_model.contact_action.model import ContactActionTransition
from scripts.train_contact_action_world_model import load_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def make_candidate_actions(split, alpha, device):
    """Create inward, outward and zero future-action candidates."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    actions = split["future_actions"].to(device).clone()
    state = split["state_history"][:, -1].to(device)
    normal = state[:, NORMAL_SLICE].reshape(-1, 2, 3)
    candidates = {"hold": actions.clone()}
    for hand_index, hand_name in enumerate(("left", "right")):
        channel = slice(hand_index * 3, hand_index * 3 + 3)
        direction = normal[:, hand_index, None, :]
        inward = actions.clone()
        outward = actions.clone()
        inward[:, :, channel] = inward[:, :, channel] - alpha * direction
        outward[:, :, channel] = outward[:, :, channel] + alpha * direction
        candidates[f"{hand_name}_inward"] = inward
        candidates[f"{hand_name}_outward"] = outward
    return candidates


def rollout_candidate_scores(
    model,
    split,
    candidate_actions,
    device,
):
    """Return per-step contact transition probabilities for each candidate."""
    state_history = split["state_history"].to(device)
    action_history = split["action_history"].to(device)
    scores = {}
    with torch.no_grad():
        for name, actions in candidate_actions.items():
            output = model.rollout(
                state_history,
                action_history,
                actions,
                teacher_states=None,
                teacher_forcing_ratio=0.0,
            )
            scores[name] = {
                "onset": torch.sigmoid(output["onset_logits"]).cpu().numpy(),
                "release": torch.sigmoid(
                    output["release_logits"]
                ).cpu().numpy(),
            }
    return scores


def first_event_horizons(current_contact, target_contact):
    current = current_contact >= 0.5
    target = target_contact >= 0.5
    onset = np.full(current.shape, -1, dtype=np.int64)
    release = np.full(current.shape, -1, dtype=np.int64)
    for hand_index in range(current.shape[1]):
        for sample_index in range(current.shape[0]):
            if not current[sample_index, hand_index]:
                hits = np.flatnonzero(target[sample_index, :, hand_index])
            else:
                hits = np.flatnonzero(~target[sample_index, :, hand_index])
            if len(hits):
                if current[sample_index, hand_index]:
                    release[sample_index, hand_index] = int(hits[0]) + 1
                else:
                    onset[sample_index, hand_index] = int(hits[0]) + 1
    return onset, release


def event_score(event_probability, horizons, hand_index, mode):
    sample_indices = np.flatnonzero(horizons[:, hand_index] > 0)
    values = np.full(len(horizons), np.nan, dtype=np.float64)
    if mode == "step":
        for sample_index in sample_indices:
            horizon = int(horizons[sample_index, hand_index])
            values[sample_index] = event_probability[
                sample_index, horizon - 1, hand_index
            ]
    elif mode == "max":
        for sample_index in sample_indices:
            horizon = int(horizons[sample_index, hand_index])
            values[sample_index] = event_probability[
                sample_index, :horizon, hand_index
            ].max()
    else:
        raise ValueError(f"Unknown score mode: {mode}")
    return values


def cluster_bootstrap_interval(
    indicators,
    groups,
    samples,
    seed,
):
    indicators = np.asarray(indicators, dtype=np.float64)
    groups = np.asarray(groups)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    if len(unique_groups) < 2 or not len(indicators):
        return None
    group_indices = [
        np.flatnonzero(inverse == group_index)
        for group_index in range(len(unique_groups))
    ]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        sampled_groups = rng.integers(
            0,
            len(unique_groups),
            size=len(unique_groups),
        )
        selected = np.concatenate([
            group_indices[group_index]
            for group_index in sampled_groups
        ])
        values.append(float(np.mean(indicators[selected])))
    lower, upper = np.quantile(values, [0.025, 0.975])
    return [float(lower), float(upper)]


def summarize_pair(
    positive,
    negative,
    groups,
    bootstrap_samples,
    seed,
):
    indicator = positive > negative
    return {
        "count": int(len(indicator)),
        "accuracy": float(indicator.mean()) if len(indicator) else None,
        "mean_margin": (
            float((positive - negative).mean())
            if len(indicator)
            else None
        ),
        "ci95": cluster_bootstrap_interval(
            indicator,
            groups,
            bootstrap_samples,
            seed,
        ),
    }


def summarize_top1(
    scores,
    expected_name,
    groups,
    bootstrap_samples,
    seed,
):
    names = list(scores)
    stacked = np.stack([scores[name] for name in names], axis=0)
    expected = names.index(expected_name)
    winner = np.argmax(stacked, axis=0)
    indicator = winner == expected
    return {
        "count": int(len(indicator)),
        "accuracy": float(indicator.mean()) if len(indicator) else None,
        "ci95": cluster_bootstrap_interval(
            indicator,
            groups,
            bootstrap_samples,
            seed,
        ),
    }


def evaluate_mode(
    score_mode,
    scores,
    onset_horizons,
    release_horizons,
    sequence_index,
    bootstrap_samples,
    seed,
):
    metrics = {}
    for hand_index, hand_name in enumerate(("left", "right")):
        onset = event_score(
            scores[f"{hand_name}_inward"]["onset"],
            onset_horizons,
            hand_index,
            score_mode,
        )
        onset_outward = event_score(
            scores[f"{hand_name}_outward"]["onset"],
            onset_horizons,
            hand_index,
            score_mode,
        )
        onset_hold = event_score(
            scores["hold"]["onset"],
            onset_horizons,
            hand_index,
            score_mode,
        )
        release = event_score(
            scores[f"{hand_name}_outward"]["release"],
            release_horizons,
            hand_index,
            score_mode,
        )
        release_inward = event_score(
            scores[f"{hand_name}_inward"]["release"],
            release_horizons,
            hand_index,
            score_mode,
        )
        release_hold = event_score(
            scores["hold"]["release"],
            release_horizons,
            hand_index,
            score_mode,
        )

        metrics[hand_name] = {}
        for horizon_name, horizon_value in (
            ("all", None),
            ("h1", 1),
            ("h2", 2),
            ("h4", 4),
            ("h8", 8),
        ):
            onset_mask = ~np.isnan(onset)
            release_mask = ~np.isnan(release)
            if horizon_value is not None:
                onset_mask &= onset_horizons[:, hand_index] == horizon_value
                release_mask &= (
                    release_horizons[:, hand_index] == horizon_value
                )
            onset_groups = sequence_index[onset_mask]
            release_groups = sequence_index[release_mask]
            metrics[hand_name][horizon_name] = {
                "onset": {
                    "count": int(onset_mask.sum()),
                    "inward_gt_outward": summarize_pair(
                        onset[onset_mask],
                        onset_outward[onset_mask],
                        onset_groups,
                        bootstrap_samples,
                        seed,
                    ),
                    "inward_gt_hold": summarize_pair(
                        onset[onset_mask],
                        onset_hold[onset_mask],
                        onset_groups,
                        bootstrap_samples,
                        seed + 1,
                    ),
                    "top1_inward": summarize_top1(
                        {
                            "inward": onset[onset_mask],
                            "outward": onset_outward[onset_mask],
                            "hold": onset_hold[onset_mask],
                        },
                        "inward",
                        onset_groups,
                        bootstrap_samples,
                        seed + 2,
                    ),
                },
                "release": {
                    "count": int(release_mask.sum()),
                    "outward_gt_inward": summarize_pair(
                        release[release_mask],
                        release_inward[release_mask],
                        release_groups,
                        bootstrap_samples,
                        seed + 3,
                    ),
                    "outward_gt_hold": summarize_pair(
                        release[release_mask],
                        release_hold[release_mask],
                        release_groups,
                        bootstrap_samples,
                        seed + 4,
                    ),
                    "top1_outward": summarize_top1(
                        {
                            "inward": release_inward[release_mask],
                            "outward": release[release_mask],
                            "hold": release_hold[release_mask],
                        },
                        "outward",
                        release_groups,
                        bootstrap_samples,
                        seed + 5,
                    ),
                },
            }
    return metrics


def main():
    args = parse_args()
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

    current_contact = split["state_history"][:, -1, CONTACT_SLICE].numpy()
    target_contact = split["future_states"][:, :, CONTACT_SLICE].numpy()
    onset_horizons, release_horizons = first_event_horizons(
        current_contact,
        target_contact,
    )
    candidates = make_candidate_actions(split, args.alpha, args.device)
    scores = rollout_candidate_scores(model, split, candidates, args.device)
    groups = split["sequence_index"].numpy()

    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "alpha": float(args.alpha),
        "bootstrap_samples": int(args.bootstrap_samples),
        "score_modes": {},
    }
    for score_mode in ("step", "max"):
        result["score_modes"][score_mode] = evaluate_mode(
            score_mode,
            scores,
            onset_horizons,
            release_horizons,
            groups,
            args.bootstrap_samples,
            args.seed + (0 if score_mode == "step" else 10),
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
