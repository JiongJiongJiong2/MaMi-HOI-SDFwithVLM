#!/usr/bin/env python3
"""Train and evaluate the action-conditioned contact/response transition."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from manip.world_model.contact_action.features import (
    ACTION_DIM,
    CONTACT_SLICE,
    NONCONTACT_DIM,
    STATE_DIM,
)
from manip.world_model.contact_action.geometry import build_geometry_bank
from manip.world_model.contact_action.model import ContactActionTransition
from manip.world_model.contact_action.model_geometry import (
    ContactActionGeometryTransition,
)
from scripts.build_contact_action_dataset import (
    TRAIN_OBJECTS,
    VALIDATION_OBJECTS,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--val_npz", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--train_mode",
        choices=("teacher", "mixed", "free"),
        default="mixed",
    )
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--event_pos_weight", type=float, default=4.0)
    parser.add_argument("--contact_calibration_weight", type=float, default=0.2)
    parser.add_argument("--residual_scale", type=float, default=0.05)
    parser.add_argument("--zero_actions", action="store_true")
    parser.add_argument("--max_eval_horizon", type=int, default=8)
    parser.add_argument(
        "--geometry_mode",
        choices=("none", "normal", "zero", "shuffle"),
        default="none",
    )
    parser.add_argument("--geometry_data_root", default="")
    parser.add_argument("--geometry_patch_grid", type=int, default=5)
    parser.add_argument(
        "--geometry_radius_normalized",
        type=float,
        default=0.08,
    )
    parser.add_argument("--geometry_output_size", type=int, default=64)
    parser.add_argument("--contrastive_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument(
        "--contrastive_objective",
        choices=("phase", "predictive"),
        default="phase",
    )
    parser.add_argument("--contrastive_top_k", type=int, default=8)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path):
    with np.load(path) as data:
        return {
            key: torch.from_numpy(data[key])
            for key in (
                "state_history",
                "action_history",
                "future_actions",
                "future_states",
                "sequence_index",
                "anchor_index",
                "object_index",
            )
        }


def model_rollout(
    model,
    split,
    indices,
    horizon,
    teacher_forcing_ratio,
    zero_actions=False,
    device="cuda",
    geometry_bank=None,
    geometry_mode="none",
    perturb_palms=False,
):
    states = split["state_history"][indices].to(device)
    action_history = split["action_history"][indices].to(device)
    future_actions = split["future_actions"][indices, :horizon].to(device)
    targets = split["future_states"][indices, :horizon].to(device)
    if zero_actions:
        future_actions = torch.zeros_like(future_actions)
    if perturb_palms:
        future_actions = future_actions.clone()
        future_actions[:, :, 0:6] *= 0.5
    rollout_kwargs = {
        "teacher_states": targets if teacher_forcing_ratio > 0 else None,
        "teacher_forcing_ratio": teacher_forcing_ratio,
    }
    if geometry_mode != "none":
        rollout_kwargs.update(
            geometry_bank=geometry_bank,
            object_indices=split["object_index"][indices].to(device),
            geometry_mode=geometry_mode,
        )
    output = model.rollout(
        states,
        action_history,
        future_actions,
        **rollout_kwargs,
    )
    return output, targets, future_actions


def collect_rollout(
    model,
    split,
    indices,
    horizon,
    teacher_forcing_ratio,
    zero_actions=False,
    perturb_palms=False,
    device="cuda",
    geometry_bank=None,
    geometry_mode="none",
    batch_size=256,
):
    outputs = {
        "states": [],
        "contact_logits": [],
        "onset_logits": [],
        "release_logits": [],
    }
    targets = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        output, batch_targets, _ = model_rollout(
            model,
            split,
            batch_indices,
            horizon,
            teacher_forcing_ratio,
            zero_actions=zero_actions,
            perturb_palms=perturb_palms,
            device=device,
            geometry_bank=geometry_bank,
            geometry_mode=geometry_mode,
        )
        for key in outputs:
            outputs[key].append(output[key])
        targets.append(batch_targets)
    return {
        key: torch.cat(values, dim=0)
        for key, values in outputs.items()
    }, torch.cat(targets, dim=0)


def state_extrapolation(split, indices, horizon, device="cuda"):
    current = split["state_history"][indices, -1].to(device).clone()
    targets = split["future_states"][indices, :horizon].to(device)
    predicted = []
    for _ in range(horizon):
        next_state = current.clone()
        next_state[:, 0:3] = current[:, 0:3] + current[:, 21:24]
        next_state[:, 9:15] = current[:, 9:15] + current[:, 15:21]
        next_state[:, 15:21] = current[:, 15:21]
        next_state[:, 21:24] = current[:, 21:24]
        next_state[:, 24:32] = current[:, 24:32]
        predicted.append(next_state)
        current = next_state
    return torch.stack(predicted, dim=1), targets


def micro_f1(logits, targets):
    prediction = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    tp = int((prediction & truth).sum().item())
    fp = int((prediction & ~truth).sum().item())
    fn = int((~prediction & truth).sum().item())
    denominator = 2 * tp + fp + fn
    return None if denominator == 0 else 2.0 * tp / denominator


def binary_auc(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive_count = int(labels.sum())
    negative_count = int((~labels).sum())
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(scores), dtype=np.float64)
    return float(
        (ranks[labels].mean() - (positive_count - 1) / 2.0)
        / negative_count
    )


def geodesic_rotation_error(predicted_6d, target_6d):
    predicted = rotation_6d_to_matrix(predicted_6d.reshape(-1, 6))
    target = rotation_6d_to_matrix(target_6d.reshape(-1, 6))
    relative = predicted.transpose(-1, -2) @ target
    trace = torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1)
    angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-6, 1.0 - 1e-6))
    return angle.mean()


def rotation_6d_to_matrix(rotation_6d):
    """Convert first-two-rows 6D rotations back to matrices."""
    rotation_6d = rotation_6d.reshape(-1, 6)
    row1 = F.normalize(rotation_6d[:, 0:3], dim=-1)
    row2 = rotation_6d[:, 3:6]
    row2 = F.normalize(
        row2 - (row2 * row1).sum(dim=-1, keepdim=True) * row1,
        dim=-1,
    )
    row3 = torch.cross(row1, row2, dim=-1)
    return torch.stack([row1, row2, row3], dim=-2)


def prediction_metrics(predicted_states, contact_logits, targets):
    horizon = targets.shape[1]
    metrics = {}
    for index in range(horizon):
        prediction = predicted_states[:, index]
        target = targets[:, index]
        metrics[f"h{index + 1}"] = {
            "contact_bce": F.binary_cross_entropy_with_logits(
                contact_logits[:, index],
                target[:, CONTACT_SLICE],
            ).item(),
            "contact_f1": micro_f1(
                contact_logits[:, index],
                target[:, CONTACT_SLICE],
            ),
            "object_translation_l1": F.l1_loss(
                prediction[:, 0:3],
                target[:, 0:3],
            ).item(),
            "object_rotation_l1": F.l1_loss(
                prediction[:, 3:9],
                target[:, 3:9],
            ).item(),
            "object_rotation_geodesic": geodesic_rotation_error(
                prediction[:, 3:9],
                target[:, 3:9],
            ).item(),
            "palm_l1": F.l1_loss(
                prediction[:, 9:15],
                target[:, 9:15],
            ).item(),
            "palm_velocity_l1": F.l1_loss(
                prediction[:, 15:21],
                target[:, 15:21],
            ).item(),
            "object_velocity_l1": F.l1_loss(
                prediction[:, 21:24],
                target[:, 21:24],
            ).item(),
        }
    return metrics


def contact_event_metrics(
    current_contact,
    target_contact,
    onset_logits,
    release_logits,
):
    current_contact = current_contact.detach().cpu().numpy()
    target_contact = target_contact.detach().cpu().numpy()
    onset_probability = (
        torch.sigmoid(onset_logits).detach().cpu().numpy()
    )
    release_probability = (
        torch.sigmoid(release_logits).detach().cpu().numpy()
    )
    current_binary = current_contact >= 0.5
    target_binary = target_contact >= 0.5
    metrics = {}
    for horizon in range(target_binary.shape[1]):
        for hand_index, hand_name in enumerate(("left", "right")):
            onset_mask = (
                (~current_binary[:, hand_index])
                & target_binary[:, horizon, hand_index]
            )
            stable_off_mask = (
                (~current_binary[:, hand_index])
                & (~target_binary[:, horizon, hand_index])
            )
            release_mask = (
                current_binary[:, hand_index]
                & (~target_binary[:, horizon, hand_index])
            )
            stable_on_mask = (
                current_binary[:, hand_index]
                & target_binary[:, horizon, hand_index]
            )
            onset_union = onset_mask | stable_off_mask
            release_union = release_mask | stable_on_mask
            onset_scores = onset_probability[
                onset_union, horizon, hand_index
            ]
            release_scores = release_probability[
                release_union, horizon, hand_index
            ]
            prefix = f"h{horizon + 1}_{hand_name}"
            metrics[f"{prefix}_onset_auc"] = binary_auc(
                onset_scores,
                onset_mask[onset_union],
            )
            metrics[f"{prefix}_release_auc"] = binary_auc(
                release_scores,
                release_mask[release_union],
            )
            metrics[f"{prefix}_onset_recall"] = (
                float(
                    (
                        onset_probability[
                            onset_mask, horizon, hand_index
                        ]
                        >= 0.5
                    ).mean()
                )
                if onset_mask.any()
                else None
            )
            metrics[f"{prefix}_release_recall"] = (
                float(
                    (
                        release_probability[
                            release_mask, horizon, hand_index
                        ]
                        >= 0.5
                    ).mean()
                )
                if release_mask.any()
                else None
            )
            metrics[f"{prefix}_onset_false_positive"] = (
                float(
                    (
                        onset_probability[
                            stable_off_mask, horizon, hand_index
                        ]
                        >= 0.5
                    ).mean()
                )
                if stable_off_mask.any()
                else None
            )
            metrics[f"{prefix}_release_false_positive"] = (
                float(
                    (
                        release_probability[
                            stable_on_mask, horizon, hand_index
                        ]
                        >= 0.5
                    ).mean()
                )
                if stable_on_mask.any()
                else None
            )
    return metrics


def aggregate_horizon_metrics(per_horizon):
    keys = next(iter(per_horizon.values())).keys()
    return {
        key: float(np.mean([
            horizon[key]
            for horizon in per_horizon.values()
            if horizon[key] is not None
        ]))
        for key in keys
    }


def serializable_args(args):
    values = dict(vars(args))
    values.pop("geometry_bank", None)
    return values


def geometry_supervised_contrastive_loss(
    embeddings,
    contact_phase,
    object_indices,
    temperature,
):
    """Pull same-object, same-contact-phase geometry embeddings together."""

    if embeddings.ndim != 3:
        raise ValueError("geometry embeddings must be [B, T, D]")
    if contact_phase.shape != embeddings.shape[:2]:
        raise ValueError("contact_phase must align with [B, T]")
    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    batch_size, steps, hidden_size = embeddings.shape
    normalized = F.normalize(embeddings, dim=-1)
    flat = normalized.reshape(batch_size * steps, hidden_size)
    phase = contact_phase.reshape(batch_size * steps)
    objects = object_indices[:, None].expand(-1, steps).reshape(-1)
    similarity = flat @ flat.transpose(0, 1) / temperature
    identity = torch.eye(
        flat.shape[0],
        device=flat.device,
        dtype=torch.bool,
    )
    positive = (phase[:, None] == phase[None, :]) & (
        objects[:, None] == objects[None, :]
    ) & ~identity
    valid_anchor = positive.any(dim=1)
    if not valid_anchor.any():
        return embeddings.new_zeros(())
    log_denominator = torch.logsumexp(
        similarity.masked_fill(identity, float("-inf")),
        dim=1,
    )
    log_positive = torch.logsumexp(
        similarity.masked_fill(~positive, float("-inf")),
        dim=1,
    )
    return (
        -(log_positive - log_denominator)[valid_anchor]
    ).mean()


def predictive_geometry_contrastive_loss(
    model,
    predicted_states,
    target_states,
    object_indices,
    geometry_bank,
    geometry_mode,
    temperature,
    top_k,
):
    """Match action-conditioned predicted geometry to the true future geometry."""

    if temperature <= 0:
        raise ValueError("contrastive temperature must be positive")
    losses = []
    for horizon in range(predicted_states.shape[1]):
        predicted_hands = model.encode_geometry_hands(
            predicted_states[:, horizon],
            object_indices,
            geometry_bank,
            geometry_mode,
        )
        target_hands = model.encode_geometry_hands(
            target_states[:, horizon],
            object_indices,
            geometry_bank,
            geometry_mode,
        )
        batch_size = predicted_hands.shape[0]
        if batch_size < 2:
            continue
        for hand_index in range(2):
            anchor = F.normalize(predicted_hands[:, hand_index], dim=-1)
            target = F.normalize(target_hands[:, hand_index], dim=-1)
            logits = anchor @ target.transpose(0, 1) / temperature
            identity = torch.eye(
                batch_size,
                device=logits.device,
                dtype=torch.bool,
            )
            negative_logits = logits.masked_fill(
                identity,
                float("-inf"),
            )
            negative_count = min(top_k, batch_size - 1)
            hard_values, hard_indices = torch.topk(
                negative_logits,
                k=negative_count,
                dim=1,
            )
            selected_logits = torch.full_like(logits, float("-inf"))
            selected_logits.scatter_(1, hard_indices, hard_values)
            diagonal = torch.arange(batch_size, device=logits.device)
            selected_logits[diagonal, diagonal] = logits[
                diagonal,
                diagonal,
            ]
            losses.append(
                F.cross_entropy(
                    selected_logits,
                    diagonal,
                )
            )
    if not losses:
        return predicted_states.new_zeros(())
    return torch.stack(losses).mean()


def evaluate_model(model, split, args, max_horizon):
    model.eval()
    indices = torch.arange(split["state_history"].shape[0])
    with torch.no_grad():
        teacher_output, targets = collect_rollout(
            model,
            split,
            indices,
            max_horizon,
            teacher_forcing_ratio=1.0,
            zero_actions=args.zero_actions,
            device=args.device,
            geometry_bank=args.geometry_bank,
            geometry_mode=args.geometry_mode,
            batch_size=args.eval_batch_size,
        )
        free_output, _ = collect_rollout(
            model,
            split,
            indices,
            max_horizon,
            teacher_forcing_ratio=0.0,
            zero_actions=args.zero_actions,
            device=args.device,
            geometry_bank=args.geometry_bank,
            geometry_mode=args.geometry_mode,
            batch_size=args.eval_batch_size,
        )
        zero_output, _ = collect_rollout(
            model,
            split,
            indices,
            max_horizon,
            teacher_forcing_ratio=0.0,
            zero_actions=True,
            device=args.device,
            geometry_bank=args.geometry_bank,
            geometry_mode=args.geometry_mode,
            batch_size=args.eval_batch_size,
        )
        extrapolated, _ = state_extrapolation(
            split,
            indices,
            max_horizon,
            device=args.device,
        )

        perturbed, _ = collect_rollout(
            model,
            split,
            indices,
            max_horizon,
            teacher_forcing_ratio=0.0,
            perturb_palms=True,
            device=args.device,
            geometry_bank=args.geometry_bank,
            geometry_mode=args.geometry_mode,
            batch_size=args.eval_batch_size,
        )
        sensitivity = {
            "contact_probability_change": torch.abs(
                torch.sigmoid(perturbed["contact_logits"])
                - torch.sigmoid(free_output["contact_logits"])
            ).mean().item(),
            "object_translation_change": torch.abs(
                perturbed["states"][:, :, 0:3]
                - free_output["states"][:, :, 0:3]
            ).mean().item(),
            "palm_change": torch.abs(
                perturbed["states"][:, :, 9:15]
                - free_output["states"][:, :, 9:15]
            ).mean().item(),
        }

    return {
        "teacher_forced": prediction_metrics(
            teacher_output["states"],
            teacher_output["contact_logits"],
            targets,
        ),
        "free_running": prediction_metrics(
            free_output["states"],
            free_output["contact_logits"],
            targets,
        ),
        "zero_action": prediction_metrics(
            zero_output["states"],
            zero_output["contact_logits"],
            targets,
        ),
        "state_extrapolation": prediction_metrics(
            extrapolated,
            torch.logit(
                extrapolated[:, :, CONTACT_SLICE].clamp(1e-4, 1 - 1e-4)
            ),
            targets,
        ),
        "action_sensitivity": sensitivity,
        "contact_events": contact_event_metrics(
            split["state_history"][indices, -1, CONTACT_SLICE].to(
                args.device
            ),
            targets[:, :, CONTACT_SLICE],
            free_output["onset_logits"],
            free_output["release_logits"],
        ),
    }


def train(args):
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_split = load_split(args.train_npz)
    val_split = load_split(args.val_npz)
    if args.max_train_samples > 0:
        train_split = {
            key: value[: args.max_train_samples]
            for key, value in train_split.items()
        }
    if args.max_val_samples > 0:
        val_split = {
            key: value[: args.max_val_samples]
            for key, value in val_split.items()
        }
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    args.geometry_bank = None
    if args.geometry_mode != "none":
        if not args.geometry_data_root:
            raise ValueError(
                "--geometry_data_root is required when geometry is enabled"
            )
        args.geometry_bank = build_geometry_bank(
            args.geometry_data_root,
            TRAIN_OBJECTS + VALIDATION_OBJECTS,
            args.device,
        )
        val_split["object_index"] = val_split["object_index"] + len(
            TRAIN_OBJECTS
        )

    train_state_history = train_split["state_history"].to(args.device)
    train_action_history = train_split["action_history"].to(args.device)
    train_future_actions = train_split["future_actions"].to(args.device)
    train_future_states = train_split["future_states"].to(args.device)
    train_object_indices = train_split["object_index"].to(args.device)
    train_size = train_state_history.shape[0]

    if args.geometry_mode == "none":
        model = ContactActionTransition(
            hidden_size=args.hidden_size,
            residual_scale=args.residual_scale,
        ).to(args.device)
    else:
        model = ContactActionGeometryTransition(
            hidden_size=args.hidden_size,
            residual_scale=args.residual_scale,
            geometry_output_size=args.geometry_output_size,
            geometry_patch_grid=args.geometry_patch_grid,
            geometry_radius_normalized=args.geometry_radius_normalized,
            geometry_mode=args.geometry_mode,
        ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_val = math.inf
    history = []

    if args.train_mode == "teacher":
        training_ratios = [1.0]
    elif args.train_mode == "free":
        training_ratios = [0.0]
    else:
        training_ratios = [args.teacher_forcing_ratio]

    for epoch in range(args.epochs):
        model.train()
        losses = []
        permutation = torch.randperm(train_size, device=args.device)
        for batch_start in range(0, train_size, args.batch_size):
            batch_indices = permutation[
                batch_start : batch_start + args.batch_size
            ]
            state_history = train_state_history[batch_indices]
            action_history = train_action_history[batch_indices]
            future_actions = train_future_actions[batch_indices]
            future_states = train_future_states[batch_indices]
            object_indices = (
                train_object_indices[batch_indices]
                if args.geometry_mode != "none"
                else None
            )
            if args.zero_actions:
                future_actions = torch.zeros_like(future_actions)
            rollout_kwargs = {
                "teacher_states": future_states,
                "teacher_forcing_ratio": training_ratios[0],
            }
            if args.geometry_mode != "none":
                rollout_kwargs.update(
                    geometry_bank=args.geometry_bank,
                    object_indices=object_indices,
                    geometry_mode=args.geometry_mode,
                )
            output = model.rollout(
                state_history,
                action_history,
                future_actions,
                **rollout_kwargs,
            )
            if args.contrastive_weight > 0:
                if args.geometry_mode != "normal":
                    raise ValueError(
                        "contrastive auxiliary requires geometry_mode=normal"
                    )
                if args.contrastive_objective == "phase":
                    future_input_contacts = torch.cat(
                        [
                            state_history[:, -1:, CONTACT_SLICE],
                            future_states[:, :-1, CONTACT_SLICE],
                        ],
                        dim=1,
                    )
                    contact_sequence = torch.cat(
                        [
                            state_history[:, :, CONTACT_SLICE],
                            future_input_contacts,
                        ],
                        dim=1,
                    )
                    contact_phase = (
                        (contact_sequence[..., 0] >= 0.5).long()
                        + 2
                        * (contact_sequence[..., 1] >= 0.5).long()
                    )
                    contrastive_loss = (
                        geometry_supervised_contrastive_loss(
                            output["geometry_embeddings"],
                            contact_phase,
                            object_indices,
                            args.contrastive_temperature,
                        )
                    )
                else:
                    contrastive_loss = (
                        predictive_geometry_contrastive_loss(
                            model,
                            output["states"],
                            future_states,
                            object_indices,
                            args.geometry_bank,
                            args.geometry_mode,
                            args.contrastive_temperature,
                            args.contrastive_top_k,
                        )
                    )
            else:
                contrastive_loss = None
            noncontact_loss = F.smooth_l1_loss(
                output["states"][:, :, :NONCONTACT_DIM],
                future_states[:, :, :NONCONTACT_DIM],
            )
            current_contact = state_history[:, -1, CONTACT_SLICE]
            target_contact = future_states[:, :, CONTACT_SLICE]
            currently_off = (current_contact < 0.5)[:, None].expand_as(
                target_contact
            )
            currently_on = ~currently_off
            onset_labels = (
                currently_off & (target_contact >= 0.5)
            ).to(target_contact.dtype)
            release_labels = (
                currently_on & (target_contact < 0.5)
            ).to(target_contact.dtype)
            pos_weight = torch.tensor(
                args.event_pos_weight,
                device=target_contact.device,
                dtype=target_contact.dtype,
            )

            def masked_bce(logits, labels, mask):
                elementwise = F.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                    reduction="none",
                    pos_weight=pos_weight,
                )
                return (elementwise * mask).sum() / mask.sum().clamp_min(1.0)

            onset_loss = masked_bce(
                output["onset_logits"],
                onset_labels,
                currently_off,
            )
            release_loss = masked_bce(
                output["release_logits"],
                release_labels,
                currently_on,
            )
            calibration_loss = F.binary_cross_entropy_with_logits(
                output["contact_logits"],
                target_contact,
            )
            contact_loss = (
                onset_loss
                + release_loss
                + args.contact_calibration_weight * calibration_loss
            )
            loss = noncontact_loss + contact_loss
            if contrastive_loss is not None:
                loss = loss + args.contrastive_weight * contrastive_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().item()))

        val_metrics = evaluate_model(
            model,
            val_split,
            args,
            min(args.max_eval_horizon, val_split["future_states"].shape[1]),
        )
        val_loss = aggregate_horizon_metrics(
            val_metrics["free_running"]
        )["object_translation_l1"] + aggregate_horizon_metrics(
            val_metrics["free_running"]
        )["contact_bce"]
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": float(val_loss),
        })
        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": serializable_args(args),
                    "epoch": epoch,
                    "best_val": best_val,
                },
                output_dir / "best.pt",
            )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={np.mean(losses):.6f} "
            f"val_loss={val_loss:.6f}"
        )

    checkpoint = torch.load(output_dir / "best.pt", map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = evaluate_model(
        model,
        val_split,
        args,
        min(args.max_eval_horizon, val_split["future_states"].shape[1]),
    )
    metrics["epochs"] = history
    metrics["best_epoch"] = int(checkpoint["epoch"])
    metrics["args"] = serializable_args(args)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics["action_sensitivity"], indent=2, sort_keys=True))


if __name__ == "__main__":
    train(parse_args())
