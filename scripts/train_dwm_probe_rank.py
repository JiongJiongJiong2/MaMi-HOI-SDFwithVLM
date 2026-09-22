#!/usr/bin/env python3
"""Train one DWM P0 passive probe ranking arm."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from manip.world_model.dwm.probe_data import (
    candidate_delta,
    centered_candidate_delta,
    condition_mask,
    load_split,
    oracle_response_context,
    split_indices,
)
from manip.world_model.dwm.probe_model import DWMProbeRankingModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=("listwise", "quotient", "absolute", "oracle"),
        default="listwise",
    )
    parser.add_argument(
        "--condition",
        default="all",
        help="all, none, or fixed:2 / random:4",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--response-dim", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--max-train-groups", type=int, default=0)
    parser.add_argument("--max-val-groups", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_condition(text):
    if text == "all":
        return None, None
    if text == "none":
        return "none", 0
    if ":" not in text:
        raise ValueError(f"unsupported condition: {text}")
    mode, length = text.split(":", 1)
    return mode, int(length)


def select_condition(split, condition):
    mode, length = parse_condition(condition)
    if mode is None:
        return split
    mask = condition_mask(split, mode, length)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError(f"condition has no rows: {condition}")
    return split_indices(split, indices)


def compute_normalization(train):
    state = np.concatenate([
        train["initial_state"],
        train["post_probe_state"],
    ], axis=0)
    state_mean = state.mean(axis=0)
    state_std = state.std(axis=0).clip(min=1e-6)
    valid_probe = train["probe_mask"].reshape(-1)
    probe_actions = train["probe_action"].reshape(-1, train["probe_action"].shape[-1])
    probe_actions = probe_actions[valid_probe]
    action = np.concatenate([
        train["candidate_action"].reshape(
            -1,
            train["candidate_action"].shape[-1],
        ),
        probe_actions,
    ], axis=0)
    action_mean = action.mean(axis=0)
    action_std = action.std(axis=0).clip(min=1e-6)
    delta = centered_candidate_delta(train).reshape(
        -1,
        centered_candidate_delta(train).shape[-1],
    )
    delta_mean = delta.mean(axis=0)
    delta_std = delta.std(axis=0).clip(min=1e-6)
    return {
        "state_mean": torch.as_tensor(state_mean, dtype=torch.float32),
        "state_std": torch.as_tensor(state_std, dtype=torch.float32),
        "action_mean": torch.as_tensor(action_mean, dtype=torch.float32),
        "action_std": torch.as_tensor(action_std, dtype=torch.float32),
        "delta_mean": torch.as_tensor(delta_mean, dtype=torch.float32),
        "delta_std": torch.as_tensor(delta_std, dtype=torch.float32),
    }


def tensor_split(split, device):
    result = {}
    for key in (
        "initial_state",
        "probe_action",
        "probe_state",
        "probe_mask",
        "post_probe_state",
        "candidate_action",
        "candidate_contact_mode",
        "target_translation",
        "utility",
        "object_id",
        "reset_id",
        "probe_mode",
        "probe_length",
    ):
        value = split[key]
        if key in ("object_id", "probe_mode"):
            result[key] = value
        elif key == "probe_mask":
            result[key] = torch.as_tensor(value, dtype=torch.bool, device=device)
        elif key == "utility":
            result[key] = torch.as_tensor(value, dtype=torch.float32, device=device)
        elif key == "candidate_contact_mode":
            result[key] = torch.as_tensor(value, dtype=torch.long, device=device)
        else:
            result[key] = torch.as_tensor(value, device=device)
    result["candidate_delta"] = torch.as_tensor(
        centered_candidate_delta(split),
        dtype=torch.float32,
        device=device,
    )
    result["oracle_context"] = torch.as_tensor(
        oracle_response_context(split),
        dtype=torch.float32,
        device=device,
    )
    return result


def normalize_model_inputs(batch, normalization):
    state_mean = normalization["state_mean"]
    state_std = normalization["state_std"]
    action_mean = normalization["action_mean"]
    action_std = normalization["action_std"]
    normalized = dict(batch)
    normalized["initial_state"] = (
        batch["initial_state"] - state_mean
    ) / state_std
    normalized["probe_state"] = (
        batch["probe_state"] - state_mean
    ) / state_std
    normalized["post_probe_state"] = (
        batch["post_probe_state"] - state_mean
    ) / state_std
    normalized["probe_action"] = (
        batch["probe_action"] - action_mean
    ) / action_std
    normalized["candidate_action"] = (
        batch["candidate_action"] - action_mean
    ) / action_std
    return normalized


def ranking_loss(scores, utility, mode, temperature=0.002):
    if mode == "listwise":
        target = F.softmax(utility / temperature, dim=1)
        return -(target * F.log_softmax(scores, dim=1)).sum(dim=1).mean()
    if mode == "quotient":
        left = scores[:, :, None]
        right = scores[:, None, :]
        difference = utility[:, :, None] - utility[:, None, :]
        weight = (difference.abs() / temperature).clamp(max=1.0)
        active = difference > 1e-6
        pair = F.softplus(-(left - right))
        return (pair * weight * active).sum() / active.sum().clamp_min(1)
    raise ValueError(f"unsupported ranking mode: {mode}")


def forward_batch(model, batch, normalization):
    normalized = normalize_model_inputs(batch, normalization)
    kwargs = {}
    if model.oracle_context_dim > 0:
        kwargs["oracle_context"] = batch["oracle_context"]
    return model(
        normalized["initial_state"],
        normalized["probe_action"],
        normalized["probe_state"],
        normalized["probe_mask"],
        normalized["post_probe_state"],
        normalized["candidate_action"],
        batch["target_translation"] / 0.02,
        **kwargs,
    )


def evaluate(model, split, normalization, objective, device):
    model.eval()
    scores = []
    utilities = []
    chosen_modes = []
    batch_size = 128
    with torch.no_grad():
        for start in range(0, split["initial_state"].shape[0], batch_size):
            stop = start + batch_size
            batch = {
                key: value[start:stop]
                for key, value in split.items()
                if torch.is_tensor(value)
            }
            batch["object_id"] = split["object_id"][start:stop]
            batch["probe_mode"] = split["probe_mode"][start:stop]
            batch["probe_length"] = split["probe_length"][start:stop]
            output = forward_batch(model, batch, normalization)
            if objective == "absolute":
                predicted_delta = output["predicted_object_delta"]
                predicted = (
                    predicted_delta[..., :3]
                    * normalization["delta_std"][:3]
                    + normalization["delta_mean"][:3]
                )
                target = batch["target_translation"]
                score = -torch.abs(
                    predicted - target[:, None, :]
                ).sum(dim=-1)
            else:
                score = output["candidate_score"]
            scores.append(score.cpu())
            utilities.append(batch["utility"].cpu())
            chosen_modes.append(
                split["candidate_contact_mode"][start:stop]
            )
    score = torch.cat(scores, dim=0)
    utility = torch.cat(utilities, dim=0)
    chosen = score.argmax(dim=1)
    oracle = utility.argmax(dim=1)
    chosen_utility = utility.gather(1, chosen[:, None]).squeeze(1)
    oracle_utility = utility.gather(1, oracle[:, None]).squeeze(1)
    regret = oracle_utility - chosen_utility
    metrics = {
        "groups": int(score.shape[0]),
        "top1": float((chosen == oracle).float().mean().item()),
        "regret_mean": float(regret.mean().item()),
        "regret_median": float(regret.median().item()),
    }
    chosen_modes = torch.cat(chosen_modes, dim=0)
    chosen_modes = chosen_modes[
        torch.arange(chosen_modes.shape[0]),
        chosen,
    ]
    metrics["slip_rate"] = float(
        (chosen_modes[:, 1:] == 2).any(dim=1).float().mean().item()
    )
    metrics["release_rate"] = float(
        (chosen_modes[:, 1:] == 3).any(dim=1).float().mean().item()
    )
    return metrics, {
        "score": score.numpy(),
        "utility": utility.numpy(),
        "chosen": chosen.numpy(),
        "oracle": oracle.numpy(),
    }


def save_checkpoint(path, model, args, normalization):
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "normalization": {
            key: value.cpu()
            for key, value in normalization.items()
        },
    }, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = select_condition(load_split(args.dataset_dir / "train.npz"), args.condition)
    val = select_condition(load_split(args.dataset_dir / "val.npz"), args.condition)
    if args.max_train_groups > 0:
        train = split_indices(train, np.arange(args.max_train_groups))
    if args.max_val_groups > 0:
        val = split_indices(val, np.arange(args.max_val_groups))
    normalization = compute_normalization(train)
    normalization = {
        key: value.to(args.device)
        for key, value in normalization.items()
    }
    train_device = tensor_split(train, args.device)
    val_device = tensor_split(val, args.device)
    oracle_dim = 5 if args.objective == "oracle" else 0
    model = DWMProbeRankingModel(
        hidden_size=args.hidden_size,
        response_dim=args.response_dim,
        oracle_context_dim=oracle_dim,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    train_size = train_device["initial_state"].shape[0]
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(train_size, device=args.device)
        losses = []
        for start in range(0, train_size, args.batch_size):
            indices = permutation[start:start + args.batch_size]
            batch = {
                key: value[indices]
                for key, value in train_device.items()
                if torch.is_tensor(value)
            }
            output = forward_batch(model, batch, normalization)
            delta_target = (
                batch["candidate_delta"] - normalization["delta_mean"]
            ) / normalization["delta_std"]
            delta_loss = F.smooth_l1_loss(
                output["predicted_object_delta"],
                delta_target,
            )
            if args.objective == "absolute":
                loss = delta_loss
            else:
                loss = ranking_loss(
                    output["candidate_score"],
                    batch["utility"],
                    "listwise" if args.objective == "oracle" else args.objective,
                )
                loss = loss + 0.1 * delta_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        val_metrics, _ = evaluate(
            model,
            val_device,
            normalization,
            args.objective,
            args.device,
        )
        val_score = val_metrics["regret_mean"]
        history.append({
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val": val_metrics,
        })
        if val_score < best_val:
            best_val = val_score
            best_epoch = epoch
            stale = 0
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                args,
                normalization,
            )
        else:
            stale += 1
            if stale >= args.patience:
                break
        print(
            f"epoch={epoch + 1} loss={np.mean(losses):.6f} "
            f"val_top1={val_metrics['top1']:.4f} "
            f"val_regret={val_metrics['regret_mean']:.6f}",
            flush=True,
        )
    checkpoint = torch.load(
        args.output_dir / "best.pt",
        map_location=args.device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    val_metrics, val_predictions = evaluate(
        model,
        val_device,
        normalization,
        args.objective,
        args.device,
    )
    result = {
        "objective": args.objective,
        "condition": args.condition,
        "seed": args.seed,
        "best_epoch": best_epoch + 1,
        "best_val_regret": best_val,
        "train_groups": int(train_size),
        "val": val_metrics,
        "history": history,
    }
    np.savez_compressed(
        args.output_dir / "val_predictions.npz",
        **val_predictions,
        object_id=val["object_id"],
        reset_id=val["reset_id"],
        probe_mode=val["probe_mode"],
        probe_length=val["probe_length"],
    )
    if args.allow_test:
        test = select_condition(
            load_split(args.dataset_dir / "test.npz"),
            args.condition,
        )
        test_device = tensor_split(test, args.device)
        test_metrics, test_predictions = evaluate(
            model,
            test_device,
            normalization,
            args.objective,
            args.device,
        )
        result["test"] = test_metrics
        np.savez_compressed(
            args.output_dir / "test_predictions.npz",
            **test_predictions,
            object_id=test["object_id"],
            reset_id=test["reset_id"],
            probe_mode=test["probe_mode"],
            probe_length=test["probe_length"],
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
