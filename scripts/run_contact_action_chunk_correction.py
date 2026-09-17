#!/usr/bin/env python3
"""L1 pilot: rank bounded palm action chunks with the contact world model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from manip.world_model.contact_action.features import (
    CONTACT_SLICE,
    NORMAL_SLICE,
)
from manip.world_model.contact_action.geometry import build_geometry_bank
from manip.world_model.contact_action.episodes import (
    aggregate_episode_metrics,
    contact_episode_metrics,
)
from manip.world_model.contact_action.model import ContactActionTransition
from manip.world_model.contact_action.model_geometry import (
    ContactActionGeometryTransition,
)
from scripts.evaluate_contact_protocol_v0_dense import (
    dense_contact_distances,
)
from scripts.select_mami_candidates_with_world_model import (
    candidate_features,
    candidate_files,
)
from scripts.build_contact_action_dataset import (
    TRAIN_OBJECTS,
    VALIDATION_OBJECTS,
    load_object_sdf,
)
from scripts.train_contact_action_world_model import (
    rotation_6d_to_matrix,
)


HAND_NAMES = ("left", "right")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_root", required=True)
    parser.add_argument("--world_model_checkpoint", required=True)
    parser.add_argument("--data_root_folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate_seed", type=int, default=1)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--num_random", type=int, default=8)
    parser.add_argument(
        "--proxy_contact_threshold_norm",
        type=float,
        default=0.02,
    )
    parser.add_argument("--contact_threshold_m", type=float, default=0.05)
    parser.add_argument("--penetration_weight", type=float, default=0.0)
    parser.add_argument(
        "--scorer_mode",
        choices=("geom_only", "learned_state", "hybrid_event"),
        default="geom_only",
    )
    parser.add_argument(
        "--rollout_mode",
        choices=("analytic", "learned"),
        default="analytic",
    )
    parser.add_argument("--event_weight", type=float, default=0.25)
    parser.add_argument("--stable_min_frames", type=int, default=3)
    parser.add_argument("--geometry_data_root", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_visual_sequence", default="")
    parser.add_argument(
        "--save_visual_hand",
        choices=("", "left", "right"),
        default="",
    )
    parser.add_argument("--save_visual_output", default="")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_world_model(args, checkpoint):
    model_args = checkpoint["args"]
    geometry_mode = model_args.get("geometry_mode", "none")
    residual_mask = model_args.get("residual_mask", "palm")
    if geometry_mode == "none":
        model = ContactActionTransition(
            hidden_size=model_args["hidden_size"],
            residual_scale=model_args.get("residual_scale", 0.0),
            residual_mask=residual_mask,
        )
        geometry_bank = None
        return model, geometry_bank, geometry_mode

    if not args.geometry_data_root:
        raise ValueError(
            "--geometry_data_root is required by a geometry checkpoint"
        )
    model = ContactActionGeometryTransition(
        hidden_size=model_args["hidden_size"],
        residual_scale=model_args.get("residual_scale", 0.0),
        geometry_output_size=model_args.get("geometry_output_size", 64),
        geometry_patch_grid=model_args.get("geometry_patch_grid", 5),
        geometry_radius_normalized=model_args.get(
            "geometry_radius_normalized",
            0.08,
        ),
        geometry_mode=geometry_mode,
        residual_mask=residual_mask,
    )
    geometry_bank = build_geometry_bank(
        Path(args.geometry_data_root),
        TRAIN_OBJECTS + VALIDATION_OBJECTS,
        args.device,
    )
    return model, geometry_bank, geometry_mode


def smooth_random_residual(horizon, rng, scale, knot_count=3):
    knot_times = np.linspace(0.0, 1.0, knot_count)
    times = np.linspace(0.0, 1.0, horizon)
    knots = rng.normal(0.0, scale, size=(knot_count, 3))
    residual = np.stack([
        np.interp(times, knot_times, knots[:, axis])
        for axis in range(3)
    ], axis=-1)
    return residual.astype(np.float32)


def make_action_candidates(
    base_actions,
    normal,
    hand_index,
    alpha,
    num_random,
    rng,
):
    horizon = base_actions.shape[0]
    channel = slice(hand_index * 3, hand_index * 3 + 3)
    candidates = [
        ("base", base_actions.copy()),
    ]
    for name, sign in (("inward", -1.0), ("outward", 1.0)):
        action = base_actions.copy()
        action[:, channel] += sign * alpha * normal[hand_index]
        candidates.append((name, action))
    action = base_actions.copy()
    action[:, channel] = 0.0
    candidates.append(("zero_palm", action))
    for index in range(num_random):
        action = base_actions.copy()
        action[:, channel] += smooth_random_residual(
            horizon,
            rng,
            scale=alpha,
        )
        candidates.append((f"random_{index}", action))
    return candidates


def action_residual_penalty(candidate_actions, base_actions, hand_index):
    channel = slice(hand_index * 3, hand_index * 3 + 3)
    residual = candidate_actions[..., channel] - base_actions[..., channel]
    magnitude = residual.abs().mean(dim=(1, 2))
    smoothness = (
        residual[:, 1:] - residual[:, :-1]
    ).abs().mean(dim=(1, 2)) if residual.shape[1] > 1 else torch.zeros_like(
        magnitude
    )
    return magnitude, smoothness


def world_model_cost(
    predicted_states,
    contact_logits,
    candidate_actions,
    base_actions,
    hand_index,
):
    clearance = predicted_states[..., 24 + hand_index]
    contact_probability = torch.sigmoid(
        contact_logits[..., hand_index]
    )
    penetration = torch.relu(-clearance).mean(dim=1)
    magnitude, smoothness = action_residual_penalty(
        candidate_actions,
        base_actions,
        hand_index,
    )
    clearance_abs = clearance.abs().mean(dim=1)
    contact_mean = contact_probability.mean(dim=1)
    cost = (
        clearance_abs
        + 0.5 * penetration
        + 0.02 * magnitude
        + 0.02 * smoothness
        - 0.05 * contact_mean
    )
    return cost, {
        "clearance_abs": clearance_abs,
        "penetration": penetration,
        "contact_probability": contact_mean,
        "action_magnitude": magnitude,
        "action_smoothness": smoothness,
    }


def candidate_selection_score(
    scorer_mode,
    frame_contact_fraction,
    mean_penetration,
    event_probability,
    penetration_weight,
    event_weight,
):
    score = (
        frame_contact_fraction
        - penetration_weight * mean_penetration
    )
    if scorer_mode == "hybrid_event":
        score = score + event_weight * event_probability
    return score


def state_to_palm_world(
    states,
    anchor_object_pos,
    anchor_object_rot,
    object_scale,
):
    rotation = rotation_6d_to_matrix(states[..., 3:9].reshape(-1, 6))
    rotation = rotation.reshape(*states.shape[:-1], 3, 3)
    object_rotation = rotation @ anchor_object_rot
    object_translation = (
        anchor_object_pos
        + (states[..., 0:3] * object_scale)
        @ anchor_object_rot.transpose(-1, -2)
    )
    palms = (
        states[..., 9:15].reshape(*states.shape[:-1], 2, 3)
        * object_scale
    )
    return (
        palms @ object_rotation.transpose(-1, -2)
        + object_translation.unsqueeze(-2)
    )


def contact_f1(predicted_contact, ground_truth_contact):
    prediction = np.asarray(predicted_contact, dtype=bool)
    truth = np.asarray(ground_truth_contact, dtype=bool)
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    return 2.0 * tp / max(2 * tp + fp + fn, 1)


def bootstrap_interval(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def first_onset_window(windows, ground_truth_contact, hand_index):
    current_indices = windows["anchor_indices"]
    future_states = windows["future_states"]
    for window_index, current_index in enumerate(current_indices):
        frame_indices = np.arange(
            current_index + 1,
            current_index + 1 + future_states.shape[1],
        )
        if frame_indices[-1] >= len(ground_truth_contact):
            continue
        if (
            ground_truth_contact[current_index, hand_index] < 0.5
            and (ground_truth_contact[frame_indices, hand_index] >= 0.5).any()
        ):
            return window_index
    return None


def main():
    args = parse_args()
    if args.scorer_mode == "geom_only" and args.rollout_mode != "analytic":
        raise ValueError(
            "geom_only requires --rollout_mode analytic"
        )
    if args.scorer_mode != "geom_only" and args.rollout_mode != "learned":
        raise ValueError(
            f"{args.scorer_mode} requires --rollout_mode learned"
        )
    data_root = Path(args.data_root_folder)
    rng = np.random.default_rng(args.seed)
    candidate_dir = (
        Path(args.candidate_root)
        / f"candidate_seed_{args.candidate_seed}"
    )
    mapping = candidate_files(candidate_dir)
    checkpoint = torch.load(
        args.world_model_checkpoint,
        map_location=args.device,
    )
    model, geometry_bank, geometry_mode = load_world_model(
        args,
        checkpoint,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    object_scale_cache = {}
    object_sdf_cache = {}
    events = []
    for sequence_name, candidate_path in sorted(mapping.items()):
        candidate = np.load(candidate_path, allow_pickle=True)
        features = candidate_features(
            candidate,
            data_root,
            object_scale_cache,
            object_sdf_cache,
            contact_threshold_norm=args.proxy_contact_threshold_norm,
            history=args.history,
            horizon=args.horizon,
            stride=args.stride,
        )
        ground_truth = np.load(
            data_root
            / "contact_labels_w_semantics_npy_files"
            / f"{sequence_name}.npy"
        )[:, :2]
        for hand_index, hand_name in enumerate(HAND_NAMES):
            window_index = first_onset_window(
                features["windows"],
                ground_truth,
                hand_index,
            )
            if window_index is None:
                continue
            current_index = int(
                features["windows"]["anchor_indices"][window_index]
            )
            frame_indices = np.arange(
                current_index + 1,
                current_index + 1 + args.horizon,
            )
            state_history = torch.from_numpy(
                features["windows"]["state_history"][window_index]
            )[None].to(args.device)
            action_history = torch.from_numpy(
                features["windows"]["action_history"][window_index]
            )[None].to(args.device)
            base_actions_np = features["windows"]["future_actions"][
                window_index
            ]
            normal = features["windows"]["state_history"][
                window_index, -1, NORMAL_SLICE
            ].reshape(2, 3)
            candidates = make_action_candidates(
                base_actions_np,
                normal,
                hand_index,
                args.alpha,
                args.num_random,
                rng,
            )
            candidate_actions = torch.from_numpy(
                np.stack([action for _, action in candidates])
            ).to(args.device)
            batch_size = len(candidates)
            object_name = str(candidate["object_name"])
            rollout_kwargs = {
                "teacher_states": None,
                "teacher_forcing_ratio": 0.0,
                "use_learned_residual": args.rollout_mode == "learned",
            }
            if geometry_mode != "none":
                object_index = geometry_bank.names.index(object_name)
                rollout_kwargs.update(
                    geometry_bank=geometry_bank,
                    object_indices=torch.full(
                        (batch_size,),
                        object_index,
                        dtype=torch.long,
                        device=args.device,
                    ),
                    geometry_mode=geometry_mode,
                )
            with torch.no_grad():
                output = model.rollout(
                    state_history.repeat(batch_size, 1, 1),
                    action_history.repeat(batch_size, 1, 1),
                    candidate_actions,
                    **rollout_kwargs,
                )
                base_actions = candidate_actions[0:1]
                costs, cost_components = world_model_cost(
                    output["states"],
                    output["contact_logits"],
                    candidate_actions,
                    base_actions,
                    hand_index,
                )
            predicted_states = output["states"].detach().cpu()
            anchor_object_pos = torch.as_tensor(
                candidate["obj_com_pos"][0],
                dtype=torch.float32,
            )
            anchor_object_rot = torch.as_tensor(
                candidate["obj_rot_mat"][0],
                dtype=torch.float32,
            )
            object_scale = float(object_scale_cache[object_name])

            predicted_palms = state_to_palm_world(
                predicted_states,
                anchor_object_pos,
                anchor_object_rot,
                object_scale,
            ).numpy()
            base_hand_verts = np.asarray(
                candidate[f"pred_{hand_name}_hand_verts"],
                dtype=np.float32,
            )
            base_jpos = np.asarray(candidate["global_jpos"], dtype=np.float32)
            base_palms = base_jpos[frame_indices][:, [22, 23]]
            object_pos = np.asarray(
                candidate["obj_com_pos"][frame_indices],
                dtype=np.float32,
            )
            object_rot = np.asarray(
                candidate["obj_rot_mat"][frame_indices],
                dtype=np.float32,
            )
            modified = (
                base_hand_verts[frame_indices][None]
                + (
                    predicted_palms[:, :, hand_index]
                    - base_palms[:, hand_index]
                )[:, :, None, :]
            ).reshape(batch_size * args.horizon, -1, 3)
            object_pos_batch = np.repeat(
                object_pos[None],
                batch_size,
                axis=0,
            ).reshape(batch_size * args.horizon, 3)
            object_rot_batch = np.repeat(
                object_rot[None],
                batch_size,
                axis=0,
            ).reshape(batch_size * args.horizon, 3, 3)
            object_sdf = load_object_sdf(
                data_root,
                str(candidate["object_name"]),
                object_sdf_cache,
            )
            distances = dense_contact_distances(
                modified,
                object_pos_batch,
                object_rot_batch,
                object_sdf,
            ).reshape(batch_size, args.horizon, -1)
            predicted_contact = (
                distances.min(axis=-1) <= args.contact_threshold_m
            )
            frame_contact_fraction = predicted_contact.mean(axis=-1)
            vertex_contact_fraction = (
                distances <= args.contact_threshold_m
            ).mean(
                axis=(1, 2)
            )
            min_distance = distances.min(axis=(1, 2))
            mean_id_penetration = np.maximum(
                -distances,
                0.0,
            ).mean(axis=(1, 2))
            ground_truth_window = ground_truth[frame_indices, hand_index]
            candidate_f1 = [
                contact_f1(predicted_contact[index], ground_truth_window)
                for index in range(batch_size)
            ]
            candidate_episode_metrics = [
                contact_episode_metrics(
                    predicted_contact[index],
                    ground_truth_window,
                    stable_min_frames=args.stable_min_frames,
                )
                for index in range(batch_size)
            ]
            event_probability = torch.sigmoid(
                output["contact_logits"][..., hand_index]
            ).mean(dim=1).detach().cpu().numpy()
            selection_score = candidate_selection_score(
                args.scorer_mode,
                frame_contact_fraction,
                mean_id_penetration,
                event_probability,
                args.penetration_weight,
                args.event_weight,
            )
            selected_index = int(
                selection_score.argmax().item()
            )
            random_indices = [
                index
                for index, (name, _) in enumerate(candidates)
                if name.startswith("random_")
            ]
            if (
                args.save_visual_output
                and sequence_name == args.save_visual_sequence
                and hand_name == args.save_visual_hand
            ):
                visual_path = Path(args.save_visual_output)
                visual_path.parent.mkdir(parents=True, exist_ok=True)
                hand_vertices = modified.reshape(
                    batch_size,
                    args.horizon,
                    -1,
                    3,
                )
                hand_distances = distances.reshape(
                    batch_size,
                    args.horizon,
                    -1,
                )
                np.savez_compressed(
                    visual_path,
                    sequence_name=np.asarray(sequence_name),
                    hand=np.asarray(hand_name),
                    object_name=np.asarray(str(candidate["object_name"])),
                    frame_indices=frame_indices,
                    base_hand_vertices=hand_vertices[0],
                    selected_hand_vertices=hand_vertices[selected_index],
                    base_hand_distances=hand_distances[0],
                    selected_hand_distances=hand_distances[selected_index],
                    object_vertices=np.asarray(
                        candidate["pred_object_verts"],
                        dtype=np.float32,
                    )[frame_indices],
                    object_faces=np.asarray(
                        candidate["object_faces"],
                        dtype=np.int64,
                    ),
                    object_pos=object_pos,
                    object_rot=object_rot,
                    ground_truth_contact=ground_truth_window,
                    base_predicted_contact=predicted_contact[0],
                    selected_predicted_contact=predicted_contact[
                        selected_index
                    ],
                    base_f1=np.asarray(candidate_f1[0]),
                    selected_f1=np.asarray(
                        candidate_f1[selected_index]
                    ),
                    oracle_f1=np.asarray(max(candidate_f1)),
                    selected_name=np.asarray(
                        candidates[selected_index][0]
                    ),
                    base_actions=np.asarray(
                        candidates[0][1],
                        dtype=np.float32,
                    ),
                    selected_actions=np.asarray(
                        candidates[selected_index][1],
                        dtype=np.float32,
                    ),
                )
            events.append({
                "sequence": sequence_name,
                "hand": hand_name,
                "current_index": current_index,
                "frame_indices": frame_indices.tolist(),
                "selected_name": candidates[selected_index][0],
                "selected_index": selected_index,
                "selected_f1": candidate_f1[selected_index],
                "base_f1": candidate_f1[0],
                "random_mean_f1": float(np.mean([
                    candidate_f1[index] for index in random_indices
                ])),
                "oracle_f1": float(np.max(candidate_f1)),
                "candidate_costs": costs.detach().cpu().tolist(),
                "candidate_f1": candidate_f1,
                "candidate_names": [
                    name for name, _ in candidates
                ],
                "candidate_contact_sequences": (
                    predicted_contact.astype(np.int8).tolist()
                ),
                "candidate_episode_metrics": candidate_episode_metrics,
                "candidate_clearance_abs": cost_components[
                    "clearance_abs"
                ].detach().cpu().tolist(),
                "candidate_penetration": cost_components[
                    "penetration"
                ].detach().cpu().tolist(),
                "candidate_contact_probability": cost_components[
                    "contact_probability"
                ].detach().cpu().tolist(),
                "candidate_action_magnitude": cost_components[
                    "action_magnitude"
                ].detach().cpu().tolist(),
                "candidate_action_smoothness": cost_components[
                    "action_smoothness"
                ].detach().cpu().tolist(),
                "candidate_frame_contact_fraction": (
                    frame_contact_fraction.tolist()
                ),
                "candidate_vertex_contact_fraction": (
                    vertex_contact_fraction.tolist()
                ),
                "candidate_min_distance": min_distance.tolist(),
                "candidate_mean_penetration_distance": (
                    mean_id_penetration.tolist()
                ),
            })

    result = {
        "sequence_count": len({event["sequence"] for event in events}),
        "event_count": len(events),
        "selected_mean_f1": float(np.mean([
            event["selected_f1"] for event in events
        ])) if events else None,
        "base_mean_f1": float(np.mean([
            event["base_f1"] for event in events
        ])) if events else None,
        "random_mean_f1": float(np.mean([
            event["random_mean_f1"] for event in events
        ])) if events else None,
        "oracle_mean_f1": float(np.mean([
            event["oracle_f1"] for event in events
        ])) if events else None,
        "scorer_mode": args.scorer_mode,
        "rollout_mode": args.rollout_mode,
        "event_weight": args.event_weight,
        "event_head_used": args.scorer_mode == "hybrid_event",
        "checkpoint_hash": sha256_file(args.world_model_checkpoint),
        "residual_scale": checkpoint["args"].get("residual_scale", 0.0),
        "residual_mask": checkpoint["args"].get("residual_mask", "palm"),
        "geometry_mode": geometry_mode,
        "penetration_weight": args.penetration_weight,
        "episode_protocol": {
            "stable_min_frames": int(args.stable_min_frames),
        },
        "events": events,
    }
    if events:
        result["episode_metrics"] = {
            "base": aggregate_episode_metrics([
                event["candidate_episode_metrics"][0]
                for event in events
            ]),
            "selected": aggregate_episode_metrics([
                event["candidate_episode_metrics"][
                    event["selected_index"]
                ]
                for event in events
            ]),
            "random": aggregate_episode_metrics([
                metric
                for event in events
                for name, metric in zip(
                    event["candidate_names"],
                    event["candidate_episode_metrics"],
                )
                if name.startswith("random_")
            ]),
        }
        result["selected_minus_base"] = {
            "mean": float(np.mean([
                event["selected_f1"] - event["base_f1"]
                for event in events
            ])),
            "ci95": bootstrap_interval(
                [
                    event["selected_f1"] - event["base_f1"]
                    for event in events
                ],
                5000,
                args.seed,
            ),
        }
        result["selected_minus_random"] = {
            "mean": float(np.mean([
                event["selected_f1"] - event["random_mean_f1"]
                for event in events
            ])),
            "ci95": bootstrap_interval(
                [
                    event["selected_f1"] - event["random_mean_f1"]
                    for event in events
                ],
                5000,
                args.seed + 1,
            ),
        }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: value
        for key, value in result.items()
        if key != "events"
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
