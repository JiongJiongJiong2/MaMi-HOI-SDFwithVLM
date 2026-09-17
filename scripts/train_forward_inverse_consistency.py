#!/usr/bin/env python3
"""Train and diagnose forward-inverse consistency on E2 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from manip.world_model.consistency import (
    CONTACT_DIM,
    CONSEQUENCE_DIM,
    HAND_ACTION_DIM,
    OBJECT_CONSEQUENCE_DIM,
    ForwardConsequenceModel,
    InverseActionModel,
    build_consequences,
    build_hand_actions,
    decode_forward_consequences,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--max_horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path, max_samples=0):
    with np.load(path) as data:
        split = {
            key: torch.from_numpy(data[key])
            for key in (
                "state_history",
                "action_history",
                "future_actions",
                "future_states",
                "sequence_index",
            )
        }
    if max_samples > 0:
        split = {
            key: value[:max_samples]
            for key, value in split.items()
        }
    return split


def batch_roll(value, shift=1):
    return torch.roll(value, shifts=shift, dims=0)


def forward_metrics(prediction, target):
    object_delta = prediction[..., :OBJECT_CONSEQUENCE_DIM]
    target_object_delta = target[..., :OBJECT_CONSEQUENCE_DIM]
    contact_logits = prediction[..., OBJECT_CONSEQUENCE_DIM:]
    target_contact = target[..., OBJECT_CONSEQUENCE_DIM:]
    contact_prediction = (torch.sigmoid(contact_logits) >= 0.5)
    contact_truth = target_contact >= 0.5
    tp = int((contact_prediction & contact_truth).sum().item())
    fp = int((contact_prediction & ~contact_truth).sum().item())
    fn = int((~contact_prediction & contact_truth).sum().item())
    return {
        "object_translation_l1": float(
            F.l1_loss(
                object_delta[..., :3],
                target_object_delta[..., :3],
            ).item()
        ),
        "object_rotation_l1": float(
            F.l1_loss(
                object_delta[..., 3:9],
                target_object_delta[..., 3:9],
            ).item()
        ),
        "contact_bce": float(
            F.binary_cross_entropy_with_logits(
                contact_logits,
                target_contact,
            ).item()
        ),
        "contact_f1": (
            float(2 * tp / (2 * tp + fp + fn))
            if 2 * tp + fp + fn
            else None
        ),
    }


def per_sample_action_error(prediction, target):
    return (prediction - target).abs().mean(
        dim=tuple(range(1, prediction.ndim))
    )


def cluster_bootstrap_interval(values, groups, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_indices = [
        np.flatnonzero(inverse == index)
        for index in range(len(unique_groups))
    ]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        selected_groups = rng.integers(
            0,
            len(unique_groups),
            size=len(unique_groups),
        )
        selected = np.concatenate([
            group_indices[index] for index in selected_groups
        ])
        draws.append(float(values[selected].mean()))
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def evaluate(
    forward_model,
    inverse_model,
    split,
    args,
    constant_action,
    bootstrap_samples=0,
):
    forward_model.eval()
    inverse_model.eval()
    state_history = split["state_history"].to(args.device)
    action_history = split["action_history"].to(args.device)
    future_actions = split["future_actions"].to(args.device)
    future_states = split["future_states"].to(args.device)
    sequence_index = split["sequence_index"].numpy()
    hand_actions = build_hand_actions(future_actions)
    consequences = build_consequences(future_actions, future_states)
    with torch.no_grad():
        forward_prediction = forward_model(
            state_history,
            action_history,
            hand_actions,
        )
        inverse_true = inverse_model(
            state_history,
            action_history,
            consequences,
        )
        inverse_shuffled = inverse_model(
            state_history,
            action_history,
            batch_roll(consequences),
        )
        cycle_prediction = inverse_model(
            state_history,
            action_history,
            decode_forward_consequences(forward_prediction),
        )
        forward_history_shuffled = forward_model(
            batch_roll(state_history),
            batch_roll(action_history),
            hand_actions,
        )
        forward_action_shuffled = forward_model(
            state_history,
            action_history,
            batch_roll(hand_actions),
        )

    result = {
        "horizons": {},
        "sequence_count": int(len(np.unique(sequence_index))),
        "sample_count": int(len(sequence_index)),
    }
    for horizon_index in range(args.max_horizon):
        horizon = horizon_index + 1
        action_target = hand_actions[:, horizon_index]
        constant_target = constant_action[horizon_index].to(
            args.device
        )
        inverse_error = per_sample_action_error(
            inverse_true[:, horizon_index],
            action_target,
        )
        shuffled_error = per_sample_action_error(
            inverse_shuffled[:, horizon_index],
            action_target,
        )
        cycle_error = per_sample_action_error(
            cycle_prediction[:, horizon_index],
            action_target,
        )
        constant_error = per_sample_action_error(
            constant_target.expand_as(action_target),
            action_target,
        )
        stride = (
            hand_actions[:, horizon_index]
            - hand_actions[:, horizon_index - 1]
            if horizon_index > 0
            else hand_actions[:, horizon_index]
        )
        action_energy = stride.abs().mean(dim=-1)
        forward = forward_metrics(
            forward_prediction[:, horizon_index],
            consequences[:, horizon_index],
        )
        history_shuffled = forward_metrics(
            forward_history_shuffled[:, horizon_index],
            consequences[:, horizon_index],
        )
        action_shuffled = forward_metrics(
            forward_action_shuffled[:, horizon_index],
            consequences[:, horizon_index],
        )
        result["horizons"][f"h{horizon}"] = {
            "forward": forward,
            "forward_history_shuffled": history_shuffled,
            "forward_action_shuffled": action_shuffled,
            "inverse_action_mae": float(inverse_error.mean().item()),
            "inverse_shuffled_action_mae": float(
                shuffled_error.mean().item()
            ),
            "inverse_constant_action_mae": float(
                constant_error.mean().item()
            ),
            "cycle_action_mae": float(cycle_error.mean().item()),
            "inverse_minus_shuffled": {
                "mean": float(
                    (inverse_error - shuffled_error).mean().item()
                ),
                "ci95": (
                    cluster_bootstrap_interval(
                        (inverse_error - shuffled_error).cpu().numpy(),
                        sequence_index,
                        bootstrap_samples,
                        args.seed + horizon,
                    )
                    if bootstrap_samples > 0
                    else None
                ),
            },
            "inverse_minus_constant": {
                "mean": float(
                    (inverse_error - constant_error).mean().item()
                ),
                "ci95": (
                    cluster_bootstrap_interval(
                        (inverse_error - constant_error).cpu().numpy(),
                        sequence_index,
                        bootstrap_samples,
                        args.seed + 100 + horizon,
                    )
                    if bootstrap_samples > 0
                    else None
                ),
            },
            "cycle_minus_inverse_true": {
                "mean": float(
                    (cycle_error - inverse_error).mean().item()
                ),
                "ci95": (
                    cluster_bootstrap_interval(
                        (cycle_error - inverse_error).cpu().numpy(),
                        sequence_index,
                        bootstrap_samples,
                        args.seed + 200 + horizon,
                    )
                    if bootstrap_samples > 0
                    else None
                ),
            },
            "action_stride_energy": float(action_energy.mean().item()),
        }
    return result


def train(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_split = load_split(args.train_npz, args.max_train_samples)
    val_split = load_split(args.val_npz, args.max_val_samples)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    forward_model = ForwardConsequenceModel(
        history=args.history,
        hidden_size=args.hidden_size,
        max_horizon=args.max_horizon,
    ).to(args.device)
    inverse_model = InverseActionModel(
        history=args.history,
        hidden_size=args.hidden_size,
        max_horizon=args.max_horizon,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        list(forward_model.parameters())
        + list(inverse_model.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    train_state = train_split["state_history"].to(args.device)
    train_action_history = train_split["action_history"].to(args.device)
    train_future_actions = train_split["future_actions"].to(args.device)
    train_future_states = train_split["future_states"].to(args.device)
    train_hand_actions = build_hand_actions(train_future_actions)
    train_consequences = build_consequences(
        train_future_actions,
        train_future_states,
    )
    train_size = train_state.shape[0]
    best_val = float("inf")
    history = []

    for epoch in range(args.epochs):
        forward_model.train()
        inverse_model.train()
        permutation = torch.randperm(train_size, device=args.device)
        losses = []
        for start in range(0, train_size, args.batch_size):
            indices = permutation[start : start + args.batch_size]
            states = train_state[indices]
            action_history = train_action_history[indices]
            hand_actions = train_hand_actions[indices]
            consequences = train_consequences[indices]
            forward_prediction = forward_model(
                states,
                action_history,
                hand_actions,
            )
            inverse_prediction = inverse_model(
                states,
                action_history,
                consequences,
            )
            forward_loss = (
                F.smooth_l1_loss(
                    forward_prediction[..., :OBJECT_CONSEQUENCE_DIM],
                    consequences[..., :OBJECT_CONSEQUENCE_DIM],
                )
                + F.binary_cross_entropy_with_logits(
                    forward_prediction[
                        ..., OBJECT_CONSEQUENCE_DIM:
                    ],
                    consequences[..., OBJECT_CONSEQUENCE_DIM:],
                )
            )
            inverse_loss = F.smooth_l1_loss(
                inverse_prediction,
                hand_actions,
            )
            loss = forward_loss + inverse_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(forward_model.parameters())
                + list(inverse_model.parameters()),
                5.0,
            )
            optimizer.step()
            losses.append(float(loss.detach().item()))

        val_metrics = evaluate(
            forward_model,
            inverse_model,
            val_split,
            args,
            train_hand_actions.mean(dim=0),
            bootstrap_samples=0,
        )
        val_loss = float(np.mean([
            metrics["forward"]["object_translation_l1"]
            + metrics["forward"]["contact_bce"]
            + metrics["inverse_action_mae"]
            for metrics in val_metrics["horizons"].values()
        ]))
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": val_loss,
        })
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "forward_state_dict": forward_model.state_dict(),
                    "inverse_state_dict": inverse_model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_val": best_val,
                },
                output_dir / "best.pt",
            )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f}"
        )

    checkpoint = torch.load(output_dir / "best.pt", map_location=args.device)
    forward_model.load_state_dict(checkpoint["forward_state_dict"])
    inverse_model.load_state_dict(checkpoint["inverse_state_dict"])
    metrics = evaluate(
        forward_model,
        inverse_model,
        val_split,
        args,
        train_hand_actions.mean(dim=0),
        bootstrap_samples=args.bootstrap_samples,
    )
    metrics["history"] = history
    metrics["best_epoch"] = int(checkpoint["epoch"])
    metrics["best_validation_loss"] = best_val
    metrics["args"] = vars(args)
    metrics["input_hashes"] = {
        "train_npz": hashlib.sha256(
            Path(args.train_npz).read_bytes()
        ).hexdigest(),
        "val_npz": hashlib.sha256(
            Path(args.val_npz).read_bytes()
        ).hexdigest(),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "best_epoch": metrics["best_epoch"],
        "best_validation_loss": metrics["best_validation_loss"],
        "horizons": metrics["horizons"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    train(parse_args())
