#!/usr/bin/env python3
"""Train one DWM action-conditioned response arm."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from manip.world_model.dwm.model import DWMTransitionModel
from manip.world_model.dwm.schema import CONTACT_MODE_LABELS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-npz", type=Path, required=True)
    parser.add_argument("--val-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--action-mode",
        choices=("true", "shuffled", "zero"),
        default="true",
    )
    parser.add_argument(
        "--target-mode",
        choices=("absolute", "state_centered"),
        default="absolute",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--object-loss-weight", type=float, default=10.0)
    parser.add_argument("--mode-loss-weight", type=float, default=0.1)
    parser.add_argument("--impulse-loss-weight", type=float, default=0.01)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path, maximum=0):
    with np.load(path, allow_pickle=False) as data:
        result = {
            "states": torch.from_numpy(
                np.asarray(data["states"], dtype=np.float32)
            ),
            "actions": torch.from_numpy(
                np.asarray(data["actions"], dtype=np.float32)
            ),
            "object_pose": torch.from_numpy(
                np.asarray(data["object_pose"], dtype=np.float32)
            ),
            "contact_mode": torch.from_numpy(
                np.asarray(data["contact_mode"], dtype=np.int64)
            ),
            "contact_forces": torch.from_numpy(
                np.asarray(data["contact_forces"], dtype=np.float32)
            ),
            "object_id": np.asarray(data["object_id"]),
            "reset_id": np.asarray(data["reset_id"], dtype=np.int64),
        }
    if maximum > 0:
        result = {
            key: value[:maximum] for key, value in result.items()
        }
    return result


def prepare_targets(split, target_mode):
    initial_pose = split["object_pose"][:, 0]
    object_delta = split["object_pose"][:, 1:] - initial_pose[:, None]
    contact_mode = split["contact_mode"][:, 1:]
    contact_impulse = split["contact_forces"][:, 1:].sum(dim=2)
    if target_mode == "state_centered":
        groups = {}
        for index, key in enumerate(zip(
            split["object_id"],
            split["reset_id"].tolist(),
        )):
            groups.setdefault((str(key[0]), int(key[1])), []).append(index)
        object_centered = object_delta.clone()
        impulse_centered = contact_impulse.clone()
        for indices in groups.values():
            object_centered[indices] -= object_delta[indices].mean(
                dim=0,
                keepdim=True,
            )
            impulse_centered[indices] -= contact_impulse[indices].mean(
                dim=0,
                keepdim=True,
            )
        object_delta = object_centered
        contact_impulse = impulse_centered
    elif target_mode != "absolute":
        raise ValueError(f"Unsupported target mode: {target_mode}")
    return object_delta, contact_mode, contact_impulse


def action_arm(actions, mode, seed):
    if mode == "true":
        return actions
    if mode == "zero":
        return torch.zeros_like(actions)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(actions.shape[0], generator=generator)
    return actions[permutation]


def macro_f1(prediction, target, class_count):
    scores = []
    for class_index in range(class_count):
        truth = target == class_index
        predicted = prediction == class_index
        tp = int((truth & predicted).sum().item())
        fp = int((~truth & predicted).sum().item())
        fn = int((truth & ~predicted).sum().item())
        denominator = 2 * tp + fp + fn
        scores.append(2.0 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def evaluate(model, split, normalization, horizon, device, target_mode):
    model.eval()
    batch_size = 512
    predictions = []
    targets = []
    mode_predictions = []
    mode_targets = []
    with torch.no_grad():
        for start in range(0, split["states"].shape[0], batch_size):
            stop = start + batch_size
            initial = split["states"][start:stop, 0].to(device)
            actions = split["actions"][start:stop, :horizon].to(device)
            initial = (
                initial - normalization["state_mean"]
            ) / normalization["state_std"]
            actions = (
                actions - normalization["action_mean"]
            ) / normalization["action_std"]
            output = model(initial, actions)
            predictions.append(output["object_delta"].cpu())
            mode_predictions.append(
                output["contact_mode_logits"].argmax(dim=-1).cpu()
            )
            object_delta, contact_mode, _ = prepare_targets({
                key: value[start:stop]
                for key, value in split.items()
            }, target_mode)
            targets.append(object_delta[:, :horizon])
            mode_targets.append(contact_mode[:, :horizon])
    prediction = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    mode_prediction = torch.cat(mode_predictions, dim=0)
    mode_target = torch.cat(mode_targets, dim=0)
    metrics = {}
    for index in range(horizon):
        metrics[f"h{index + 1}"] = {
            "object_translation_l1": float(
                F.l1_loss(
                    prediction[:, index, :3],
                    target[:, index, :3],
                ).item()
            ),
            "object_rotation_l1": float(
                F.l1_loss(
                    prediction[:, index, 3:9],
                    target[:, index, 3:9],
                ).item()
            ),
            "contact_mode_accuracy": float(
                (mode_prediction[:, index] == mode_target[:, index])
                .float()
                .mean()
                .item()
            ),
            "contact_mode_macro_f1": macro_f1(
                mode_prediction[:, index],
                mode_target[:, index],
                len(CONTACT_MODE_LABELS),
            ),
        }
    return metrics


def save_checkpoint(path, model, args, normalization):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "normalization": {
                key: value.cpu()
                for key, value in normalization.items()
            },
        },
        path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    train_split = load_split(args.train_npz, args.max_train_samples)
    val_split = load_split(args.val_npz, args.max_val_samples)
    horizon = train_split["actions"].shape[1]
    train_actions = action_arm(
        train_split["actions"],
        args.action_mode,
        args.seed,
    )
    val_actions = action_arm(
        val_split["actions"],
        args.action_mode,
        args.seed + 1,
    )
    train_split["actions"] = train_actions
    val_split["actions"] = val_actions
    state_mean = train_split["states"][:, 0].mean(dim=0)
    state_std = train_split["states"][:, 0].std(dim=0).clamp_min(1e-6)
    action_mean = train_actions.mean(dim=(0, 1))
    action_std = train_actions.std(dim=(0, 1)).clamp_min(1e-6)
    train_impulse_raw = prepare_targets(
        train_split,
        args.target_mode,
    )[2]
    impulse_mean = train_impulse_raw.mean(dim=(0, 1))
    impulse_std = train_impulse_raw.std(dim=(0, 1)).clamp_min(1e-6)
    normalization = {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "impulse_mean": impulse_mean,
        "impulse_std": impulse_std,
    }
    gpu_normalization = {
        key: value.to(args.device) for key, value in normalization.items()
    }
    model = DWMTransitionModel(
        hidden_size=args.hidden_size,
        max_horizon=horizon,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_object_delta, train_mode, train_impulse = prepare_targets(
        train_split,
        args.target_mode,
    )
    train_size = train_split["states"].shape[0]
    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(train_size)
        losses = []
        for start in range(0, train_size, args.batch_size):
            indices = permutation[start:start + args.batch_size]
            initial = train_split["states"][indices, 0].to(args.device)
            actions = train_split["actions"][indices].to(args.device)
            initial = (
                initial - gpu_normalization["state_mean"]
            ) / gpu_normalization["state_std"]
            actions = (
                actions - gpu_normalization["action_mean"]
            ) / gpu_normalization["action_std"]
            output = model(initial, actions)
            target_delta = train_object_delta[indices].to(args.device)
            target_mode = train_mode[indices].to(args.device)
            target_impulse = (
                train_impulse[indices].to(args.device)
                - gpu_normalization["impulse_mean"]
            ) / gpu_normalization["impulse_std"]
            object_loss = F.smooth_l1_loss(
                output["object_delta"],
                target_delta,
            )
            mode_loss = F.cross_entropy(
                output["contact_mode_logits"].reshape(-1, 4),
                target_mode.reshape(-1),
            )
            impulse_loss = F.mse_loss(
                output["contact_impulse_normalized"],
                target_impulse,
            )
            loss = (
                args.object_loss_weight * object_loss
                + args.mode_loss_weight * mode_loss
                + args.impulse_loss_weight * impulse_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))
        val_metrics = evaluate(
            model,
            val_split,
            gpu_normalization,
            horizon,
            args.device,
            args.target_mode,
        )
        val_loss = float(np.mean([
            metrics["object_translation_l1"]
            + 0.1 * metrics["object_rotation_l1"]
            + 0.1 * (1.0 - metrics["contact_mode_macro_f1"])
            for metrics in val_metrics.values()
        ]))
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": val_loss,
        })
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                args,
                normalization,
            )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={np.mean(losses):.6f} val_loss={val_loss:.6f}",
            flush=True,
        )
    checkpoint = torch.load(
        args.output_dir / "best.pt",
        map_location=args.device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate(
        model,
        val_split,
        gpu_normalization,
        horizon,
        args.device,
        args.target_mode,
    )
    metrics["history"] = history
    metrics["best_val"] = best_val
    metrics["action_mode"] = args.action_mode
    metrics["target_mode"] = args.target_mode
    metrics["seed"] = args.seed
    metrics["train_samples"] = int(train_size)
    metrics["val_samples"] = int(val_split["states"].shape[0])
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "action_mode": args.action_mode,
        "best_val": best_val,
        "h8": metrics.get(f"h{horizon}"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
