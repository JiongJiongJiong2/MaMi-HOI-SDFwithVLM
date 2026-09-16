#!/usr/bin/env python3
"""Select among MaMi-generated candidates with the contact world model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from manip.world_model.contact_action.features import (
    CLEARANCE_SLICE,
    CONTACT_SLICE,
    build_window_features,
    make_training_samples,
)
from manip.world_model.contact_action.model import ContactActionTransition
from scripts.build_contact_action_dataset import (
    load_object_scale,
    load_object_sdf,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_dir", action="append", required=True)
    parser.add_argument("--world_model_checkpoint", required=True)
    parser.add_argument("--data_root_folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--contact_threshold_norm", type=float, default=0.02)
    parser.add_argument("--penetration_weight", type=float, default=2.0)
    parser.add_argument(
        "--score_mode",
        choices=("contact_probability", "zero_clearance"),
        default="contact_probability",
    )
    parser.add_argument("--max_sequences", type=int, default=0)
    return parser.parse_args()


def candidate_files(candidate_dir):
    files = sorted(
        (Path(candidate_dir) / "res_npz_files").rglob("*.npz")
    )
    if not files:
        raise FileNotFoundError(
            f"No res_npz_files/*.npz under {candidate_dir}"
        )
    return {path.stem: path for path in files}


def load_candidates(candidate_dirs, max_sequences=0):
    directories = [candidate_files(path) for path in candidate_dirs]
    common = set(directories[0])
    for mapping in directories[1:]:
        common &= set(mapping)
    sequence_names = sorted(common)
    if max_sequences:
        sequence_names = sequence_names[:max_sequences]
    if not sequence_names:
        raise ValueError("Candidate directories have no common sequences")
    return {
        sequence_name: [
            np.load(mapping[sequence_name], allow_pickle=True)
            for mapping in directories
        ]
        for sequence_name in sequence_names
    }


def validate_candidate(candidate):
    required = {
        "global_jpos",
        "obj_com_pos",
        "obj_rot_mat",
        "start_frame_idx",
        "end_frame_idx",
    }
    missing = required.difference(candidate.files)
    if missing:
        raise KeyError(f"Candidate is missing keys: {sorted(missing)}")
    global_jpos = np.asarray(candidate["global_jpos"], dtype=np.float32)
    object_pos = np.asarray(candidate["obj_com_pos"], dtype=np.float32)
    object_rot = np.asarray(candidate["obj_rot_mat"], dtype=np.float32)
    if global_jpos.ndim != 3 or global_jpos.shape[1:] != (24, 3):
        raise ValueError(f"Unexpected global_jpos shape: {global_jpos.shape}")
    if object_pos.shape != (len(global_jpos), 3):
        raise ValueError(f"Unexpected object position shape: {object_pos.shape}")
    if object_rot.shape != (len(global_jpos), 3, 3):
        raise ValueError(f"Unexpected object rotation shape: {object_rot.shape}")
    return global_jpos, object_pos, object_rot


def candidate_features(
    candidate,
    data_root,
    object_scale_cache,
    object_sdf_cache,
    contact_threshold_norm,
    history,
    horizon,
    stride,
):
    global_jpos, object_pos, object_rot = validate_candidate(candidate)
    sequence_name = str(candidate["seq_name"])
    if "object_name" in candidate.files:
        object_name = str(candidate["object_name"])
    else:
        object_name = sequence_name.split("_")[1]
    object_scale = load_object_scale(data_root, object_name, object_scale_cache)
    object_sdf = load_object_sdf(data_root, object_name, object_sdf_cache)
    motion = global_jpos.reshape(len(global_jpos), 24 * 3)
    placeholder_contact = np.zeros((len(global_jpos), 2), dtype=np.float32)
    dummy_joint_stats = np.zeros(72, dtype=np.float32)
    dummy_joint_stats_max = np.ones(72, dtype=np.float32)
    geometry_features = build_window_features(
        motion=motion,
        contact_labels=placeholder_contact,
        object_pos=object_pos,
        object_rot=object_rot,
        jpos_minimum=dummy_joint_stats,
        jpos_maximum=dummy_joint_stats_max,
        object_scale=object_scale,
        object_sdf=object_sdf,
    )
    proxy_contact = (
        geometry_features.states[:, CLEARANCE_SLICE]
        <= contact_threshold_norm
    ).astype(np.float32)
    features = build_window_features(
        motion=motion,
        contact_labels=proxy_contact,
        object_pos=object_pos,
        object_rot=object_rot,
        jpos_minimum=dummy_joint_stats,
        jpos_maximum=dummy_joint_stats_max,
        object_scale=object_scale,
        object_sdf=object_sdf,
    )
    windows = make_training_samples(
        features.states,
        features.actions,
        history=history,
        horizon=horizon,
        stride=stride,
        max_samples=None,
    )
    if windows is None:
        raise ValueError(f"Sequence {sequence_name} is too short")
    return {
        "sequence_name": sequence_name,
        "start_frame_idx": int(candidate["start_frame_idx"]),
        "end_frame_idx": int(candidate["end_frame_idx"]),
        "states": features.states,
        "windows": windows,
    }


def score_windows(
    model,
    windows,
    batch_size,
    device,
    penetration_weight,
    score_mode,
):
    state_history = torch.from_numpy(windows["state_history"])
    action_history = torch.from_numpy(windows["action_history"])
    future_actions = torch.from_numpy(windows["future_actions"])
    contact_scores = []
    clearance_scores = []
    penetration_scores = []
    with torch.no_grad():
        for start in range(0, len(state_history), batch_size):
            end = start + batch_size
            output = model.rollout(
                state_history[start:end].to(device),
                action_history[start:end].to(device),
                future_actions[start:end].to(device),
                teacher_states=None,
                teacher_forcing_ratio=0.0,
            )
            contact_probability = torch.sigmoid(
                output["contact_logits"]
            )
            clearance = output["states"][..., CLEARANCE_SLICE]
            penetration = torch.relu(-clearance).mean(dim=(1, 2))
            contact_scores.append(
                contact_probability.mean(dim=(1, 2)).cpu()
            )
            clearance_scores.append(
                clearance.abs().mean(dim=(1, 2)).cpu()
            )
            penetration_scores.append(penetration.cpu())
    contact_score = torch.cat(contact_scores).mean().item()
    clearance_score = torch.cat(clearance_scores).mean().item()
    penetration_score = torch.cat(penetration_scores).mean().item()
    if score_mode == "contact_probability":
        world_score = contact_score - penetration_weight * penetration_score
    elif score_mode == "zero_clearance":
        world_score = -clearance_score - penetration_weight * penetration_score
    else:
        raise ValueError(f"Unknown score mode: {score_mode}")
    return {
        "world_score": float(world_score),
        "contact_score": float(contact_score),
        "clearance_score": float(clearance_score),
        "penetration_score": float(penetration_score),
    }


def proxy_contact_metrics(
    states,
    ground_truth_contact,
    contact_threshold_norm,
):
    frame_count = min(len(states), len(ground_truth_contact))
    prediction = states[:frame_count, CLEARANCE_SLICE] <= contact_threshold_norm
    truth = ground_truth_contact[:frame_count] >= 0.5
    true_positive = int((prediction & truth).sum())
    false_positive = int((prediction & ~truth).sum())
    false_negative = int((~prediction & truth).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    return {
        "contact_f1": (
            2.0 * true_positive / denominator
            if denominator
            else None
        ),
        "contact_precision": (
            true_positive / precision_denominator
            if precision_denominator
            else None
        ),
        "contact_recall": (
            true_positive / recall_denominator
            if recall_denominator
            else None
        ),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "min_clearance_norm": float(
            states[:frame_count, CLEARANCE_SLICE].min()
        ),
    }


def mean_metric(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def main():
    args = parse_args()
    data_root = Path(args.data_root_folder)
    contact_root = data_root / "contact_labels_w_semantics_npy_files"
    candidates_by_sequence = load_candidates(
        args.candidate_dir,
        args.max_sequences,
    )
    checkpoint = torch.load(
        args.world_model_checkpoint,
        map_location=args.device,
    )
    model = ContactActionTransition(
        hidden_size=checkpoint["args"]["hidden_size"],
        residual_scale=checkpoint["args"].get("residual_scale", 0.0),
        residual_mask=checkpoint["args"].get("residual_mask", "palm"),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    object_scale_cache = {}
    object_sdf_cache = {}
    selected_rows = []
    candidate_rows = []
    for sequence_name, candidates in candidates_by_sequence.items():
        ground_truth_path = contact_root / f"{sequence_name}.npy"
        if not ground_truth_path.is_file():
            raise FileNotFoundError(ground_truth_path)
        full_contact = np.load(ground_truth_path)[:, :2]
        per_candidate = []
        for candidate_index, candidate in enumerate(candidates):
            features = candidate_features(
                candidate,
                data_root,
                object_scale_cache,
                object_sdf_cache,
                args.contact_threshold_norm,
                args.history,
                args.horizon,
                args.stride,
            )
            start = features["start_frame_idx"]
            ground_truth_contact = full_contact[
                start : start + len(features["states"])
            ]
            metrics = proxy_contact_metrics(
                features["states"],
                ground_truth_contact,
                args.contact_threshold_norm,
            )
            metrics.update(score_windows(
                model,
                features["windows"],
                args.batch_size,
                args.device,
                args.penetration_weight,
                args.score_mode,
            ))
            metrics["candidate_index"] = candidate_index
            per_candidate.append(metrics)

        selected_index = int(np.argmax([
            row["world_score"] for row in per_candidate
        ]))
        selected = per_candidate[selected_index]
        selected_rows.append({
            "sequence_name": sequence_name,
            "selected_candidate_index": selected_index,
            **selected,
        })
        for row in per_candidate:
            candidate_rows.append({
                "sequence_name": sequence_name,
                **row,
            })

    baseline_rows = [
        row for row in candidate_rows if row["candidate_index"] == 0
    ]
    candidate_count = len(
        {row["candidate_index"] for row in candidate_rows}
    )
    summary = {
        "sequence_count": len(selected_rows),
        "candidate_count": candidate_count,
        "baseline_contact_f1": mean_metric(
            baseline_rows,
            "contact_f1",
        ),
        "selected_contact_f1": mean_metric(
            selected_rows,
            "contact_f1",
        ),
        "oracle_contact_f1": float(np.mean([
            max(row["contact_f1"] for row in candidate_rows
                if row["sequence_name"] == sequence_name
                and row["contact_f1"] is not None)
            for sequence_name in candidates_by_sequence
        ])),
        "baseline_min_clearance_norm": mean_metric(
            baseline_rows,
            "min_clearance_norm",
        ),
        "selected_min_clearance_norm": mean_metric(
            selected_rows,
            "min_clearance_norm",
        ),
        "selected_world_score": mean_metric(
            selected_rows,
            "world_score",
        ),
    }
    output = {
        "args": vars(args),
        "summary": summary,
        "selected": selected_rows,
        "candidates": candidate_rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
