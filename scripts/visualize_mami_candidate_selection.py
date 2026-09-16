#!/usr/bin/env python3
"""Render intuitive MaMi candidate-selection figures and GIFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from scripts.select_mami_candidates_with_world_model import (
    candidate_features,
    candidate_files,
    load_candidates,
)


SMPL24_PARENTS = (
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--candidate_root", required=True)
    parser.add_argument("--data_root_folder", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--contact_threshold_norm", type=float, default=0.02)
    parser.add_argument("--frame_stride", type=int, default=2)
    return parser.parse_args()


def load_official_metrics(candidate_root, sequence_names):
    candidate_dirs = sorted(Path(candidate_root).glob("candidate_seed_*"))
    metrics = []
    for candidate_dir in candidate_dirs:
        mapping = {}
        for path in sorted(
            (candidate_dir / "evaluation_metrics_json").rglob("*.json")
        ):
            matched = [
                sequence
                for sequence in sequence_names
                if path.name.startswith(sequence + "_")
            ]
            if not matched:
                continue
            if len(matched) > 1:
                raise ValueError(f"Could not map metric file: {path}")
            mapping[matched[0]] = json.loads(
                path.read_text(encoding="utf-8")
            )
        metrics.append(mapping)
    return candidate_dirs, metrics


def load_candidate_data(candidate_dir, sequence_name):
    mapping = candidate_files(candidate_dir)
    return np.load(mapping[sequence_name], allow_pickle=True)


def clearance_for_candidate(
    candidate,
    data_root,
    object_scale_cache,
    object_sdf_cache,
    contact_threshold,
):
    return candidate_features(
        candidate,
        data_root,
        object_scale_cache,
        object_sdf_cache,
        contact_threshold,
        history=4,
        horizon=8,
        stride=4,
    )


def plot_metric_overview(
    output_path,
    sequences,
    candidate_metrics,
    selected_indices,
):
    baseline_f1 = np.asarray([
        candidate_metrics[0][sequence]["mean_contact_f1_score"]
        for sequence in sequences
    ])
    selected_f1 = np.asarray([
        candidate_metrics[selected_indices[sequence]][sequence][
            "mean_contact_f1_score"
        ]
        for sequence in sequences
    ])
    baseline = candidate_metrics[0]
    selected = [
        candidate_metrics[selected_indices[sequence]][sequence]
        for sequence in sequences
    ]
    metric_keys = (
        ("F1", "mean_contact_f1_score"),
        ("Precision", "mean_contact_precision"),
        ("Recall", "mean_contact_recall"),
    )
    baseline_means = [
        np.mean([baseline[sequence][key] for sequence in sequences])
        for _, key in metric_keys
    ]
    selected_means = [
        np.mean([row[key] for row in selected])
        for _, key in metric_keys
    ]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(sequences))
    axes[0].bar(x - 0.2, baseline_f1, width=0.4, label="baseline")
    axes[0].bar(x + 0.2, selected_f1, width=0.4, label="selected")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [sequence.replace("sub16_", "") for sequence in sequences],
        rotation=55,
        ha="right",
        fontsize=8,
    )
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Contact F1")
    axes[0].set_title("Per-sequence official Contact F1")
    axes[0].legend()

    labels = [label for label, _ in metric_keys]
    x_metrics = np.arange(len(labels))
    axes[1].bar(x_metrics - 0.2, baseline_means, width=0.4, label="baseline")
    axes[1].bar(x_metrics + 0.2, selected_means, width=0.4, label="selected")
    axes[1].set_xticks(x_metrics)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Mean official contact metrics")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def transform_object_points(points, rotation, translation):
    return points @ rotation.T + translation


def draw_skeleton(axis, joints, color):
    axis.scatter(
        joints[:, 0],
        joints[:, 1],
        joints[:, 2],
        s=12,
        c=color,
        depthshade=False,
    )
    for child, parent in enumerate(SMPL24_PARENTS):
        if parent < 0:
            continue
        axis.plot(
            [joints[parent, 0], joints[child, 0]],
            [joints[parent, 1], joints[child, 1]],
            [joints[parent, 2], joints[child, 2]],
            color=color,
            linewidth=1.2,
        )
    axis.scatter(
        joints[[22, 23], 0],
        joints[[22, 23], 1],
        joints[[22, 23], 2],
        s=60,
        c=["tab:red", "tab:blue"],
        depthshade=False,
    )


def set_equal_axes(axis, points):
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    radius = 0.5 * float(np.max(upper - lower)) + 1e-6
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])


def render_sequence_gif(
    output_path,
    sequence_name,
    baseline_candidate,
    selected_candidate,
    object_points,
    selected_index,
    frame_stride,
):
    baseline_joints = np.asarray(
        baseline_candidate["global_jpos"], dtype=np.float64
    )
    selected_joints = np.asarray(
        selected_candidate["global_jpos"], dtype=np.float64
    )
    baseline_rotation = np.asarray(
        baseline_candidate["obj_rot_mat"], dtype=np.float64
    )
    baseline_translation = np.asarray(
        baseline_candidate["obj_com_pos"], dtype=np.float64
    )
    selected_rotation = np.asarray(
        selected_candidate["obj_rot_mat"], dtype=np.float64
    )
    selected_translation = np.asarray(
        selected_candidate["obj_com_pos"], dtype=np.float64
    )
    frame_count = min(
        len(baseline_joints),
        len(selected_joints),
        len(baseline_rotation),
        len(selected_rotation),
    )
    frames = []
    for frame_index in range(0, frame_count, frame_stride):
        figure = plt.figure(figsize=(10, 4.5))
        axes = [
            figure.add_subplot(1, 2, 1, projection="3d"),
            figure.add_subplot(1, 2, 2, projection="3d"),
        ]
        panels = (
            (
                axes[0],
                "baseline",
                baseline_joints[frame_index],
                baseline_rotation[frame_index],
                baseline_translation[frame_index],
            ),
            (
                axes[1],
                f"selected candidate {selected_index + 1}",
                selected_joints[frame_index],
                selected_rotation[frame_index],
                selected_translation[frame_index],
            ),
        )
        all_points = []
        for (
            axis,
            title,
            joints,
            rotation,
            translation,
        ) in panels:
            object_world = transform_object_points(
                object_points,
                rotation,
                translation,
            )
            axis.scatter(
                object_world[:, 0],
                object_world[:, 1],
                object_world[:, 2],
                s=2,
                c="0.55",
                alpha=0.35,
                depthshade=False,
            )
            draw_skeleton(axis, joints, "black")
            axis.set_title(title)
            all_points.extend([object_world, joints])
        combined = np.concatenate(all_points, axis=0)
        for axis in axes:
            set_equal_axes(axis, combined)
        figure.suptitle(
            f"{sequence_name}  frame {frame_index}",
            fontsize=11,
        )
        figure.tight_layout()
        figure.canvas.draw()
        image = Image.fromarray(
            np.asarray(figure.canvas.buffer_rgba())
        ).convert("P", palette=Image.ADAPTIVE)
        frames.append(image)
        plt.close(figure)
    if not frames:
        raise ValueError("No frames were rendered")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=max(40, int(1000 / 15)),
        loop=0,
        optimize=False,
    )


def plot_clearance_pair(
    output_path,
    sequence_name,
    baseline_features,
    selected_features,
    ground_truth_contact,
    selected_index,
    contact_threshold,
):
    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    frame_count = min(
        len(baseline_features["states"]),
        len(selected_features["states"]),
        len(ground_truth_contact),
    )
    time = np.arange(frame_count)
    for hand_index, hand_name in enumerate(("left", "right")):
        axis = axes[hand_index]
        baseline_clearance = baseline_features["states"][
            :frame_count, 24 + hand_index
        ]
        selected_clearance = selected_features["states"][
            :frame_count, 24 + hand_index
        ]
        axis.plot(
            time,
            baseline_clearance,
            label="baseline",
            color="tab:gray",
        )
        axis.plot(
            time,
            selected_clearance,
            label=f"selected {selected_index + 1}",
            color="tab:green",
        )
        axis.axhline(
            contact_threshold,
            color="tab:red",
            linestyle="--",
            linewidth=1,
            label="contact threshold",
        )
        contact_frames = np.flatnonzero(
            ground_truth_contact[:frame_count, hand_index] >= 0.5
        )
        for start in np.split(
            contact_frames,
            np.where(np.diff(contact_frames) > 1)[0] + 1,
        ):
            if len(start):
                axis.axvspan(
                    start[0],
                    start[-1],
                    color="tab:blue",
                    alpha=0.08,
                )
        axis.set_ylabel(f"{hand_name} clearance / scale")
        axis.grid(alpha=0.2)
        if hand_index == 0:
            axis.legend(loc="upper right")
    axes[-1].set_xlabel("frame")
    figure.suptitle(
        f"{sequence_name}: proxy clearance, shaded = GT contact"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root_folder)
    selection = json.loads(
        Path(args.selection).read_text(encoding="utf-8")
    )
    selected_indices = {
        row["sequence_name"]: int(row["selected_candidate_index"])
        for row in selection["selected"]
    }
    sequences = sorted(selected_indices)
    candidate_dirs, candidate_metrics = load_official_metrics(
        args.candidate_root,
        set(sequences),
    )
    baseline_f1 = np.asarray([
        candidate_metrics[0][sequence]["mean_contact_f1_score"]
        for sequence in sequences
    ])
    selected_f1 = np.asarray([
        candidate_metrics[selected_indices[sequence]][sequence][
            "mean_contact_f1_score"
        ]
        for sequence in sequences
    ])
    differences = selected_f1 - baseline_f1
    best_sequence = sequences[int(np.argmax(differences))]
    worst_sequence = sequences[int(np.argmin(differences))]

    plot_metric_overview(
        output_dir / "selection_metric_overview.png",
        sequences,
        candidate_metrics,
        selected_indices,
    )

    object_scale_cache = {}
    object_sdf_cache = {}
    for sequence_name in (best_sequence, worst_sequence):
        selected_index = selected_indices[sequence_name]
        baseline_candidate = load_candidate_data(
            candidate_dirs[0],
            sequence_name,
        )
        selected_candidate = load_candidate_data(
            candidate_dirs[selected_index],
            sequence_name,
        )
        baseline_features = clearance_for_candidate(
            baseline_candidate,
            data_root,
            object_scale_cache,
            object_sdf_cache,
            args.contact_threshold_norm,
        )
        selected_features = clearance_for_candidate(
            selected_candidate,
            data_root,
            object_scale_cache,
            object_sdf_cache,
            args.contact_threshold_norm,
        )
        start = baseline_features["start_frame_idx"]
        full_contact = np.load(
            data_root
            / "contact_labels_w_semantics_npy_files"
            / f"{sequence_name}.npy"
        )[:, :2]
        ground_truth_contact = full_contact[
            start : start + len(baseline_features["states"])
        ]
        plot_clearance_pair(
            output_dir / f"{sequence_name}_clearance.png",
            sequence_name,
            baseline_features,
            selected_features,
            ground_truth_contact,
            selected_index,
            args.contact_threshold_norm,
        )
        if sequence_name == best_sequence:
            object_name = str(baseline_candidate["object_name"])
            object_points = np.load(
                data_root / "rest_object_geo" / f"{object_name}.npy"
            ).reshape(-1, 3)
            object_points = object_points[::4]
            render_sequence_gif(
                output_dir / f"{sequence_name}_baseline_vs_selected.gif",
                sequence_name,
                baseline_candidate,
                selected_candidate,
                object_points,
                selected_index,
                args.frame_stride,
            )

    manifest = {
        "best_sequence": best_sequence,
        "best_f1_delta": float(differences.max()),
        "worst_sequence": worst_sequence,
        "worst_f1_delta": float(differences.min()),
        "outputs": sorted(
            path.name for path in output_dir.iterdir() if path.is_file()
        ),
    }
    (output_dir / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
