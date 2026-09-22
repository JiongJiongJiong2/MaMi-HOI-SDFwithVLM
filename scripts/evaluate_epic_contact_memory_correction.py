#!/usr/bin/env python3
"""Evaluate memory-guided correction on deterministic perturbed EPIC data."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from train_epic_contact_memory import (
    ACTION_TO_ID,
    ACTIONS,
    OBJECT_TO_ID,
    build_model,
    causal_window,
    macro_f1,
    reference_tracks,
)


POLICIES = (
    "no_correction",
    "ordinary_smoothing",
    "per_frame_nearest",
    "sticky",
    "hysteresis",
    "oracle",
    "learned",
)
SUPPORTED_OBJECT_FOLDS = ("bowl", "plate", "pan", "bottle")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--perturbations",
        default="2mm,5mm,10mm",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--max-pose-delta-l2", type=float, default=1.0)
    parser.add_argument(
        "--max-translation-delta-m",
        type=float,
        default=0.02,
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--manopth-root", type=Path)
    parser.add_argument("--mano-root", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--held-out-object",
        choices=SUPPORTED_OBJECT_FOLDS,
    )
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def parse_perturbations(value):
    values = []
    for item in value.split(","):
        text = item.strip().lower()
        if not text:
            continue
        if text.endswith("mm"):
            scale = 0.001
            text = text[:-2]
        elif text.endswith("m"):
            scale = 1.0
            text = text[:-1]
        else:
            raise ValueError("perturbation must end in m or mm")
        values.append(float(text) * scale)
    if not values:
        raise ValueError("at least one perturbation is required")
    return tuple(values)


def load_index(manifest_dir):
    rows = []
    with (manifest_dir / "index.jsonl").open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("relative_path"):
                    row["path"] = str(
                        manifest_dir
                        / row["relative_path"].replace("\\", "/")
                    )
                rows.append(row)
    return rows


def load_arrays(row):
    with np.load(row["path"], allow_pickle=False) as archive:
        return {
            key: np.asarray(archive[key])
            for key in archive.files
        }


def gaussian_kernel(sigma):
    if sigma <= 0:
        return np.asarray([1.0], dtype=np.float64)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    grid = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (grid / sigma) ** 2)
    return kernel / kernel.sum()


def lowpass_noise(length, channels, sigma, rng):
    raw = rng.normal(size=(length, channels))
    kernel = gaussian_kernel(sigma)
    radius = len(kernel) // 2
    padded = np.pad(
        raw,
        ((radius, radius), (0, 0)),
        mode="edge",
    )
    output = np.zeros_like(raw)
    for offset, weight in enumerate(kernel):
        output += weight * padded[offset : offset + length]
    return output.astype(np.float32)


def normalize_noise(noise):
    rms = float(np.sqrt(np.mean(noise ** 2)))
    if rms <= 1e-12:
        return np.zeros_like(noise)
    return noise / rms


def perturb_sequence(
    arrays,
    amplitude_m,
    seed,
):
    pose = np.asarray(arrays["mano_pose"], dtype=np.float32).copy()
    translation = np.asarray(
        arrays["mano_translation"],
        dtype=np.float32,
    ).copy()
    diameter = float(np.asarray(arrays["object_diameter_m"]))
    rng = np.random.default_rng(seed)
    pose_noise = normalize_noise(
        lowpass_noise(len(pose), pose.shape[1], 5.0, rng)
    )
    translation_noise = normalize_noise(
        lowpass_noise(len(translation), 3, 5.0, rng)
    )
    pose_scale = 0.5 * amplitude_m / max(diameter, 1e-3)
    pose += pose_scale * pose_noise
    translation += amplitude_m * translation_noise
    return {
        "pose": pose,
        "translation": translation,
        "pose_noise": pose_noise,
        "translation_noise": translation_noise,
    }


def smooth_pose(pose, kernel=(1.0, 4.0, 6.0, 4.0, 1.0)):
    values = np.asarray(pose, dtype=np.float32).copy()
    radius = len(kernel) // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    coefficients = np.asarray(kernel, dtype=np.float32)
    coefficients /= coefficients.sum()
    output = np.zeros_like(values)
    for offset, weight in enumerate(coefficients):
        output += weight * padded[offset : offset + len(values)]
    return output


def object_world(arrays, object_index, time_index):
    canonical = np.asarray(arrays["object_vertices"], dtype=np.float64)
    rotation = np.asarray(
        arrays["object_rotation"],
        dtype=np.float64,
    )
    translation = np.asarray(
        arrays["object_translation"],
        dtype=np.float64,
    )
    return (
        canonical[int(object_index)]
        @ rotation[time_index].T
        + translation[time_index]
    )


def fixed_policy_sequences(
    arrays,
    object_name,
    hand,
    policy,
    checkpoint,
    history,
):
    selected = checkpoint["reference"]["selected"]
    if policy == "oracle":
        config = dict(selected)
    elif policy == "sticky":
        config = {
            "hold_ratio": selected["hold_ratio"],
            "update_ratio": 1e9,
            "update_persistence": 1,
            "close_persistence": 1,
        }
    elif policy == "hysteresis":
        config = {
            "hold_ratio": 0.01,
            "update_ratio": 0.02,
            "update_persistence": 3,
            "close_persistence": 1,
        }
    else:
        raise ValueError(f"invalid fixed policy: {policy}")
    return reference_tracks(
        arrays,
        object_name=object_name,
        hand=hand,
        history=history,
        **config,
    )


def per_frame_sequences(arrays):
    contact = np.asarray(arrays["contact"], dtype=bool)
    top_hand = np.asarray(
        arrays["top_hand_indices"],
        dtype=np.int64,
    )
    top_object = np.asarray(
        arrays["top_object_indices"],
        dtype=np.int64,
    )
    top_distance = np.asarray(
        arrays["top_distances_m"],
        dtype=np.float64,
    )
    sequences = []
    for time_index in np.flatnonzero(contact):
        if not len(top_hand[time_index]):
            continue
        sequences.append({
            "times": np.asarray([time_index], dtype=np.int64),
            "hand_indices": np.asarray(
                [top_hand[time_index, 0]],
                dtype=np.int64,
            ),
            "object_indices": np.asarray(
                [top_object[time_index, 0]],
                dtype=np.int64,
            ),
            "labels": np.asarray(
                [ACTION_TO_ID["hold"]],
                dtype=np.int64,
            ),
            "features": np.asarray(
                [[
                    float(contact[time_index]),
                    float(top_distance[time_index, 0])
                    / float(np.asarray(arrays["object_diameter_m"])),
                ]],
                dtype=np.float32,
            ),
            "object_name": "unknown",
            "hand": "left",
        })
    return sequences


def learned_policy_sequences(
    arrays,
    object_name,
    hand,
    checkpoint,
    model,
    device,
):
    import torch

    selected = checkpoint["reference"]["selected"]
    sequences = reference_tracks(
        arrays,
        object_name=object_name,
        hand=hand,
        history=int(checkpoint["history"]),
        **selected,
    )
    if not sequences:
        return []
    histories = np.asarray([
        causal_window(
            sequence["features"],
            int(checkpoint["history"]),
        )
        for sequence in sequences
    ], dtype=np.float32)
    previous = np.asarray([
        causal_window(
            np.concatenate([
                np.asarray([ACTION_TO_ID["unknown"]]),
                sequence["labels"][:-1],
            ]),
            int(checkpoint["history"]),
        )
        for sequence in sequences
    ], dtype=np.int64)
    object_ids = np.asarray([
        OBJECT_TO_ID[sequence["object_name"]]
        for sequence in sequences
    ], dtype=np.int64)
    hand_ids = np.asarray([
        0 if sequence["hand"] == "left" else 1
        for sequence in sequences
    ], dtype=np.int64)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(sequences), 512):
            logits = model(
                torch.as_tensor(
                    histories[start : start + 512],
                    device=device,
                ),
                torch.as_tensor(
                    previous[start : start + 512],
                    device=device,
                ),
                torch.as_tensor(
                    object_ids[start : start + 512],
                    device=device,
                ),
                torch.as_tensor(
                    hand_ids[start : start + 512],
                    device=device,
                ),
            )
            predictions.append(
                logits.argmax(dim=1).cpu().numpy()
            )
    predictions = np.concatenate(predictions)
    output = []
    for sequence, predicted in zip(sequences, predictions):
        sequence = dict(sequence)
        sequence["predicted_labels"] = np.asarray(
            predicted,
            dtype=np.int64,
        )
        output.append(sequence)
    return output


def policy_sequences(
    arrays,
    object_name,
    hand,
    policy,
    checkpoint,
    model,
    device,
):
    if policy == "per_frame_nearest":
        return per_frame_sequences(arrays)
    if policy in ("sticky", "hysteresis", "oracle"):
        return fixed_policy_sequences(
            arrays,
            object_name,
            hand,
            policy,
            checkpoint,
            int(checkpoint["history"]),
        )
    if policy == "learned":
        return learned_policy_sequences(
            arrays,
            object_name,
            hand,
            checkpoint,
            model,
            device,
        )
    raise ValueError(f"invalid correction policy: {policy}")


def combine_policy_sequences(sequences):
    if not sequences:
        return None
    labels = []
    for sequence in sequences:
        labels.append(np.atleast_1d(sequence.get(
            "predicted_labels",
            sequence["labels"],
        )))
    return {
        "times": np.concatenate([
            np.atleast_1d(sequence["times"])
            for sequence in sequences
        ]),
        "hand_indices": np.concatenate([
            np.atleast_1d(sequence["hand_indices"])
            for sequence in sequences
        ]),
        "object_indices": np.concatenate([
            np.atleast_1d(sequence["object_indices"])
            for sequence in sequences
        ]),
        "labels": np.concatenate(labels),
    }


def build_mano_layer(args, side):
    import sys

    import torch

    if args.manopth_root is not None:
        sys.path.insert(0, str(args.manopth_root))
    from manopth.manolayer import ManoLayer

    if args.mano_root is None:
        raise ValueError("--mano-root is required for correction")
    return ManoLayer(
        mano_root=str(args.mano_root),
        use_pca=False,
        ncomps=0,
        side=side,
        flat_hand_mean=False,
    ).to(args.device)


def forward_hand_vertices(
    mano_layer,
    pose,
    beta,
    translation,
    device=None,
):
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    vertices, _ = mano_layer(
        torch.as_tensor(
            pose,
            dtype=torch.float32,
            device=device,
        ),
        torch.as_tensor(
            beta,
            dtype=torch.float32,
            device=device,
        ),
    )
    return (
        vertices * 1e-3
        + torch.as_tensor(
            translation,
            dtype=torch.float32,
            device=device,
        )[:, None]
    )


def sequence_temporal_loss(vertices):
    import torch

    if len(vertices) < 3:
        return torch.zeros((), device=vertices.device)
    acceleration = torch.linalg.vector_norm(
        vertices[2:] - 2.0 * vertices[1:-1] + vertices[:-2],
        dim=2,
    ).mean()
    if len(vertices) < 4:
        return acceleration
    jerk = torch.linalg.vector_norm(
        vertices[3:]
        - 3.0 * vertices[2:-1]
        + 3.0 * vertices[1:-2]
        - vertices[:-3],
        dim=2,
    ).mean()
    return acceleration + jerk


def correct_sequence(
    args,
    arrays,
    perturbed,
    policy_sequence,
    mano_layer,
):
    import torch

    perturbed_pose_full = torch.as_tensor(
        perturbed["pose"],
        dtype=torch.float32,
        device=args.device,
    )
    perturbed_translation = torch.as_tensor(
        perturbed["translation"],
        dtype=torch.float32,
        device=args.device,
    )
    pose = torch.nn.Parameter(
        perturbed_pose_full[:, 3:].clone()
    )
    translation = torch.nn.Parameter(
        torch.as_tensor(
            perturbed["translation"],
            dtype=torch.float32,
            device=args.device,
        )
    )
    beta = torch.as_tensor(
        np.asarray(arrays["mano_beta"], dtype=np.float32),
        dtype=torch.float32,
        device=args.device,
    )
    active = []
    labels = policy_sequence.get(
        "predicted_labels",
        policy_sequence["labels"],
    )
    for time_index, hand_index, object_index, label in zip(
        policy_sequence["times"],
        policy_sequence["hand_indices"],
        policy_sequence["object_indices"],
        labels,
    ):
        if int(label) not in (
            ACTION_TO_ID["hold"],
            ACTION_TO_ID["update"],
        ):
            continue
        world = object_world(
            arrays,
            object_index,
            time_index,
        )
        active.append((
            int(time_index),
            int(hand_index),
            torch.as_tensor(
                world,
                dtype=torch.float32,
                device=args.device,
            ),
        ))
    optimizer = torch.optim.Adam(
        [pose, translation],
        lr=args.learning_rate,
    )
    observed_pose_delta = torch.linalg.vector_norm(
        perturbed_pose_full[:, 3:]
        - torch.as_tensor(
            arrays["mano_pose"],
            dtype=torch.float32,
            device=args.device,
        )[:, 3:],
        dim=1,
    ).max()
    observed_translation_delta = torch.linalg.vector_norm(
        perturbed_translation
        - torch.as_tensor(
            arrays["mano_translation"],
            dtype=torch.float32,
            device=args.device,
        ),
        dim=1,
    ).max()
    pose_cap = min(
        args.max_pose_delta_l2,
        max(0.05, float(2.0 * observed_pose_delta)),
    )
    translation_cap = min(
        args.max_translation_delta_m,
        max(0.002, float(2.0 * observed_translation_delta)),
    )
    for _ in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        full_pose = torch.cat([
            perturbed_pose_full[:, :3],
            pose,
        ], dim=1)
        vertices = forward_hand_vertices(
            mano_layer,
            full_pose,
            beta,
            translation,
            device=args.device,
        )
        loss = (
            0.05 * torch.nn.functional.smooth_l1_loss(
                pose,
                perturbed_pose_full[:, 3:],
                beta=0.01,
            )
            + 0.05 * torch.nn.functional.smooth_l1_loss(
                translation,
                perturbed_translation,
                beta=0.002,
            )
            + 2.0 * sequence_temporal_loss(vertices)
        )
        if active:
            anchor_losses = []
            for time_index, hand_index, target in active:
                anchor_losses.append(torch.linalg.vector_norm(
                    vertices[time_index, hand_index] - target
                ))
            loss = loss + 10.0 * torch.stack(anchor_losses).mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            pose_delta = pose - perturbed_pose_full[:, 3:]
            pose_norm = torch.linalg.vector_norm(
                pose_delta,
                dim=1,
                keepdim=True,
            )
            pose_scale = torch.clamp(
                pose_cap
                / torch.clamp(pose_norm, min=1e-12),
                max=1.0,
            )
            pose.copy_(
                perturbed_pose_full[:, 3:]
                + pose_delta * pose_scale
            )
            translation_delta = translation - perturbed_translation
            translation_norm = torch.linalg.vector_norm(
                translation_delta,
                dim=1,
                keepdim=True,
            )
            translation_scale = torch.clamp(
                translation_cap
                / torch.clamp(translation_norm, min=1e-12),
                max=1.0,
            )
            translation.copy_(
                perturbed_translation
                + translation_delta * translation_scale
            )
    with torch.no_grad():
        full_pose = torch.cat([
            perturbed_pose_full[:, :3],
            pose,
        ], dim=1)
        vertices = forward_hand_vertices(
            mano_layer,
            full_pose,
            beta,
            translation,
            device=args.device,
        )
    return {
        "vertices": vertices.detach().cpu().numpy(),
        "pose": full_pose.detach().cpu().numpy(),
        "translation": translation.detach().cpu().numpy(),
    }


def trajectory_stat(vertices):
    vertices = np.asarray(vertices, dtype=np.float64)
    if len(vertices) < 2:
        return {"acceleration_mm": 0.0, "jerk_mm": 0.0}
    acceleration = (
        np.linalg.norm(np.diff(vertices, n=2, axis=0), axis=2).mean()
        * 1000.0
        if len(vertices) >= 3
        else 0.0
    )
    jerk = (
        np.linalg.norm(np.diff(vertices, n=3, axis=0), axis=2).mean()
        * 1000.0
        if len(vertices) >= 4
        else 0.0
    )
    return {
        "acceleration_mm": float(acceleration),
        "jerk_mm": float(jerk),
    }


def sequence_metrics(
    arrays,
    corrected_vertices,
    policy_sequence,
    oracle_sequence=None,
):
    gt_vertices = np.asarray(
        arrays["hand_vertices"],
        dtype=np.float64,
    )
    corrected = np.asarray(corrected_vertices, dtype=np.float64)
    contact = np.asarray(arrays["contact"], dtype=bool)
    labels = policy_sequence.get(
        "predicted_labels",
        policy_sequence["labels"],
    )
    times = np.asarray(policy_sequence["times"])
    labels = np.asarray(labels)
    hand_indices = np.asarray(policy_sequence["hand_indices"])
    object_indices = np.asarray(policy_sequence["object_indices"])
    aligned = min(
        len(times),
        len(labels),
        len(hand_indices),
        len(object_indices),
    )
    if aligned == 0:
        return {
            "vertex_error_mm": float(
                np.linalg.norm(
                    corrected - gt_vertices,
                    axis=2,
                ).mean()
                * 1000.0
            ),
            "hold_anchor_error_mm": None,
            "post_release_attraction_mm": None,
            "false_break_rate": 0.0,
            "switch_delay_frames": None,
            **trajectory_stat(corrected),
        }
    times = times[:aligned]
    labels = labels[:aligned]
    hand_indices = hand_indices[:aligned]
    object_indices = object_indices[:aligned]
    anchor_errors = []
    post_release = []
    false_break = 0
    for time_index, hand_index, object_index, label in zip(
        times,
        hand_indices,
        object_indices,
        labels,
    ):
        target = object_world(arrays, object_index, time_index)
        distance = np.linalg.norm(
            corrected[time_index, hand_index] - target
        )
        if (
            contact[time_index]
            and int(label)
            in (ACTION_TO_ID["hold"], ACTION_TO_ID["update"])
        ):
            anchor_errors.append(distance)
        if (
            not contact[time_index]
            and int(label)
            in (ACTION_TO_ID["hold"], ACTION_TO_ID["update"])
        ):
            post_release.append(distance)
        if (
            contact[time_index]
            and int(label)
            in (ACTION_TO_ID["close"], ACTION_TO_ID["unknown"])
        ):
            false_break += 1
    temporal = trajectory_stat(corrected)
    switch_delay = None
    if oracle_sequence is not None:
        oracle_labels = oracle_sequence["labels"]
        oracle_updates = np.asarray(
            oracle_sequence["times"]
        )[oracle_labels == ACTION_TO_ID["update"]]
        policy_updates = times[
            np.asarray(labels) == ACTION_TO_ID["update"]
        ]
        delays = []
        for update_time in oracle_updates:
            later = policy_updates[policy_updates >= update_time]
            delays.append(
                int(later[0] - update_time)
                if len(later)
                else int(
                    arrays["frames"][-1]
                    - update_time
                    + 1
                )
            )
        if delays:
            switch_delay = float(np.mean(delays))
    return {
        "vertex_error_mm": float(
            np.linalg.norm(
                corrected - gt_vertices,
                axis=2,
            ).mean()
            * 1000.0
        ),
        "hold_anchor_error_mm": (
            float(np.mean(anchor_errors) * 1000.0)
            if anchor_errors
            else None
        ),
        "post_release_attraction_mm": (
            float(np.mean(post_release) * 1000.0)
            if post_release
            else None
        ),
        "false_break_rate": (
            false_break / max(int(contact.sum()), 1)
        ),
        "switch_delay_frames": switch_delay,
        "acceleration_mm": temporal["acceleration_mm"],
        "jerk_mm": temporal["jerk_mm"],
    }


def cluster_bootstrap_mean(values, clusters, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(clusters)
    unique, inverse = np.unique(clusters, return_inverse=True)
    groups = [
        values[inverse == index]
        for index in range(len(unique))
    ]
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        selected = rng.integers(
            0,
            len(unique),
            size=len(unique),
        )
        draws[draw] = np.concatenate([
            groups[index] for index in selected
        ]).mean()
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def aggregate(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["split"], row["policy"])].append(row)
    output = {}
    for (split, policy), rows in sorted(grouped.items()):
        output.setdefault(split, {})[policy] = {
            "sequence_count": len(rows),
            "mean_vertex_error_mm": float(np.mean([
                row["vertex_error_mm"] for row in rows
            ])),
            "mean_hold_anchor_error_mm": float(np.mean([
                row["hold_anchor_error_mm"]
                for row in rows
                if row["hold_anchor_error_mm"] is not None
            ])),
            "mean_post_release_attraction_mm": float(np.mean([
                row["post_release_attraction_mm"]
                for row in rows
                if row["post_release_attraction_mm"] is not None
            ])),
            "mean_false_break_rate": float(np.mean([
                row["false_break_rate"] for row in rows
            ])),
            "mean_switch_delay_frames": float(np.mean([
                row["switch_delay_frames"]
                for row in rows
                if row["switch_delay_frames"] is not None
            ])),
            "mean_acceleration_mm": float(np.mean([
                row["acceleration_mm"] for row in rows
            ])),
            "mean_jerk_mm": float(np.mean([
                row["jerk_mm"] for row in rows
            ])),
        }
    return output


def improvement_bootstrap(
    learned,
    baseline_rows,
    metric,
    samples,
    seed,
):
    paired = []
    for row in learned:
        key = (row["segment_id"], row["perturbation_m"])
        baseline = baseline_rows.get(key)
        if baseline is None:
            continue
        learned_value = row.get(metric)
        baseline_value = baseline.get(metric)
        if learned_value is None or baseline_value is None:
            continue
        paired.append((
            float(baseline_value - learned_value),
            row["participant_id"],
        ))
    if not paired:
        return None
    values = np.asarray([item[0] for item in paired])
    clusters = [item[1] for item in paired]
    return {
        "mean": float(values.mean()),
        "ci95": cluster_bootstrap_mean(
            values,
            clusters,
            samples,
            seed,
        ),
    }


def build_gate(
    aggregate_metrics,
    bootstrap,
    object_folds,
    checkpoint,
    held_out_object,
):
    metrics = (
        "vertex_error_mm",
        "hold_anchor_error_mm",
        "post_release_attraction_mm",
        "false_break_rate",
        "switch_delay_frames",
        "acceleration_mm",
        "jerk_mm",
    )
    checks = {}
    dev = aggregate_metrics.get("dev", {})
    test = aggregate_metrics.get("test", {})

    def has_headroom(policy):
        if policy not in dev or "hysteresis" not in dev:
            return False
        return (
            dev[policy]["mean_vertex_error_mm"]
            < dev["hysteresis"]["mean_vertex_error_mm"]
            and dev[policy]["mean_post_release_attraction_mm"]
            < dev["hysteresis"]["mean_post_release_attraction_mm"]
        )

    checks["oracle_headroom_dev"] = has_headroom("oracle")
    for baseline in ("hysteresis", "sticky"):
        item = bootstrap.get(
            f"test:learned_vs_{baseline}:vertex_error_mm"
        )
        checks[f"learned_vertex_better_than_{baseline}"] = (
            item is not None and item["ci95"][0] > 0
        )
    for baseline in ("sticky", "per_frame_nearest"):
        item = bootstrap.get(
            f"test:learned_vs_{baseline}:post_release_attraction_mm"
        )
        checks[f"post_release_better_than_{baseline}"] = (
            item is not None and item["ci95"][0] > 0
        )
    if "hysteresis" in test and "learned" in test:
        baseline = test["hysteresis"]
        learned = test["learned"]
        checks["hold_drift_nonregression"] = (
            learned["mean_hold_anchor_error_mm"]
            <= 1.05 * baseline["mean_hold_anchor_error_mm"]
        )
        checks["switch_delay_nonregression"] = (
            learned["mean_switch_delay_frames"]
            <= 1.05 * baseline["mean_switch_delay_frames"]
        )
        checks["acceleration_nonregression"] = (
            learned["mean_acceleration_mm"]
            <= 1.05 * baseline["mean_acceleration_mm"]
        )
        checks["jerk_nonregression"] = (
            learned["mean_jerk_mm"]
            <= 1.05 * baseline["mean_jerk_mm"]
        )
    model_metrics = checkpoint.get("metrics", {})
    split_metrics = model_metrics.get("test")
    checks["event_baseline_nonregression"] = bool(
        split_metrics is not None
        and split_metrics["release_auprc"] is not None
        and split_metrics["release_auprc"] >= 0.2972
        and split_metrics["per_class_f1"]["close"] >= 0.3784
    )
    if held_out_object is not None:
        fold = object_folds.get(held_out_object, {})
        checks["object_fold_direction"] = bool(
            "learned" in fold
            and "hysteresis" in fold
            and fold["learned"]["mean_vertex_error_mm"]
            < fold["hysteresis"]["mean_vertex_error_mm"]
        )
    checks["pass"] = all(
        value for key, value in checks.items() if key != "pass"
    )
    return checks


def main():
    import torch

    args = parse_args()
    amplitudes = parse_perturbations(args.perturbations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_index(args.manifest_dir)
    if args.held_out_object is not None:
        rows = [
            row for row in rows
            if row["object_name"] == args.held_out_object
            and row["split"] == "train"
        ]
    if args.max_sequences:
        rows = rows[: args.max_sequences]
    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
    )
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    model_args = SimpleNamespace(
        hidden_size=checkpoint["hidden_size"],
        layers=checkpoint["layers"],
        dropout=checkpoint["dropout"],
    )
    model = build_model(
        model_args,
        checkpoint["feature_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    records = []
    mano_layers = {}
    for sequence_index, row in enumerate(rows):
        arrays = load_arrays(row)
        oracle_sequence = combine_policy_sequences(
            fixed_policy_sequences(
                arrays,
                row["object_name"],
                row["hand"],
                "oracle",
                checkpoint,
                int(checkpoint["history"]),
            )
        )
        side = row["hand"]
        if side not in mano_layers:
            mano_layers[side] = build_mano_layer(args, side)
        mano_layer = mano_layers[side]
        for amplitude_index, amplitude in enumerate(amplitudes):
            seed = (
                args.seed
                + sequence_index * 1009
                + amplitude_index
            )
            perturbed = perturb_sequence(
                arrays,
                amplitude,
                seed,
            )
            for policy in POLICIES:
                if policy == "no_correction":
                    vertices = (
                        forward_hand_vertices(
                            mano_layer,
                            perturbed["pose"],
                            arrays["mano_beta"],
                            perturbed["translation"],
                            device=args.device,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    policy_sequence = {
                        "times": np.asarray([], dtype=np.int64),
                        "hand_indices": np.asarray([], dtype=np.int64),
                        "object_indices": np.asarray([], dtype=np.int64),
                        "labels": np.asarray([], dtype=np.int64),
                    }
                elif policy == "ordinary_smoothing":
                    smoothed_translation = smooth_pose(
                        perturbed["translation"]
                    )
                    vertices = (
                        forward_hand_vertices(
                            mano_layer,
                            smooth_pose(perturbed["pose"]),
                            arrays["mano_beta"],
                            smoothed_translation,
                            device=args.device,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    policy_sequence = {
                        "times": np.asarray([], dtype=np.int64),
                        "hand_indices": np.asarray([], dtype=np.int64),
                        "object_indices": np.asarray([], dtype=np.int64),
                        "labels": np.asarray([], dtype=np.int64),
                    }
                else:
                    policy_sequence_list = policy_sequences(
                        arrays,
                        row["object_name"],
                        row["hand"],
                        policy,
                        checkpoint,
                        model,
                        device,
                    )
                    policy_sequence = combine_policy_sequences(
                        policy_sequence_list
                    )
                    if policy_sequence is None:
                        continue
                    corrected = correct_sequence(
                        args,
                        arrays,
                        perturbed,
                        policy_sequence,
                        mano_layer,
                    )
                    vertices = corrected["vertices"]
                if policy in ("no_correction", "ordinary_smoothing"):
                    combined_sequence = {
                        "times": np.flatnonzero(
                            arrays["contact"]
                        ),
                        "hand_indices": np.asarray([
                            arrays["top_hand_indices"][time, 0]
                            for time in np.flatnonzero(arrays["contact"])
                        ], dtype=np.int64),
                        "object_indices": np.asarray([
                            arrays["top_object_indices"][time, 0]
                            for time in np.flatnonzero(arrays["contact"])
                        ], dtype=np.int64),
                        "labels": np.full(
                            int(np.asarray(arrays["contact"]).sum()),
                            ACTION_TO_ID["hold"],
                            dtype=np.int64,
                        ),
                    }
                else:
                    combined_sequence = policy_sequence
                metrics = sequence_metrics(
                    arrays,
                    vertices,
                    combined_sequence,
                    oracle_sequence=oracle_sequence,
                )
                records.append({
                    "segment_id": row["segment_id"],
                    "participant_id": row["participant_id"],
                    "object_name": row["object_name"],
                    "split": row["split"],
                    "hand": row["hand"],
                    "policy": policy,
                    "perturbation_m": float(amplitude),
                    **metrics,
                })
        print(
            f"[{sequence_index + 1}/{len(rows)}] {row['segment_id']}",
            flush=True,
        )

    aggregate_metrics = aggregate(records)
    bootstrap = {}
    by_policy = defaultdict(list)
    for row in records:
        by_policy[(row["split"], row["policy"])].append(row)
    for split in ("train", "dev", "test"):
        learned = by_policy.get((split, "learned"), [])
        for baseline in ("hysteresis", "sticky", "per_frame_nearest"):
            baseline_rows = {
                (row["segment_id"], row["perturbation_m"]): row
                for row in by_policy.get((split, baseline), [])
            }
            for metric in (
                "vertex_error_mm",
                "hold_anchor_error_mm",
                "post_release_attraction_mm",
                "false_break_rate",
                "switch_delay_frames",
                "acceleration_mm",
                "jerk_mm",
            ):
                result = improvement_bootstrap(
                    learned,
                    baseline_rows,
                    metric,
                    5000,
                    args.seed,
                )
                if result is not None:
                    bootstrap[
                        f"{split}:learned_vs_{baseline}:{metric}"
                    ] = result
    object_folds = {}
    for object_name in SUPPORTED_OBJECT_FOLDS:
        object_folds[object_name] = aggregate([
            row for row in records
            if row["object_name"] == object_name
            and row["split"] == "train"
        ])
    gate = build_gate(
        aggregate_metrics,
        bootstrap,
        object_folds,
        checkpoint,
        args.held_out_object,
    )
    summary = {
        "method": (
            "deterministic low-pass perturbation and matched MANO "
            "correction under fixed material-point memory policies"
        ),
        "perturbations_m": list(amplitudes),
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "held_out_object": args.held_out_object,
        "aggregate": aggregate_metrics,
        "bootstrap_vertex_error": bootstrap,
        "object_held_out_descriptive": object_folds,
        "gate": gate,
        "decision": "GO" if gate["pass"] else "NO-GO",
        "record_count": len(records),
        "test_note": (
            "strict test split is reused from the EPIC event baseline"
        ),
    }
    output_path = args.output_dir / "summary.json"
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "summary": str(output_path),
        "aggregate": aggregate_metrics,
        "bootstrap": bootstrap,
        "gate": gate,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
