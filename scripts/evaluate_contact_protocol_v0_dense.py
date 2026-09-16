#!/usr/bin/env python3
"""Dense external-first contact evaluation for exported MaMi candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from manip.model.sdf_utils import sample_object_sdf_at_points
from scripts.build_contact_action_dataset import load_object_sdf
from scripts.select_mami_candidates_with_world_model import candidate_files


HAND_NAMES = ("left", "right")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True)
    parser.add_argument("--candidate_root", required=True)
    parser.add_argument("--data_root_folder", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--contact_threshold_m", type=float, default=0.05)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def dense_contact_distances(
    hand_verts_world,
    object_pos,
    object_rot,
    object_sdf,
):
    grid, centroid, extents = object_sdf
    hand_verts_world = torch.as_tensor(
        hand_verts_world,
        dtype=torch.float32,
    )
    object_pos = torch.as_tensor(object_pos, dtype=torch.float32)
    object_rot = torch.as_tensor(object_rot, dtype=torch.float32)
    relative = hand_verts_world - object_pos[:, None, :]
    canonical = torch.einsum(
        "tnj,tji->tni",
        relative,
        object_rot,
    )
    frame_count, vertex_count, _ = canonical.shape
    flattened = canonical.reshape(1, frame_count * vertex_count, 3)
    with torch.no_grad():
        distances = sample_object_sdf_at_points(
            grid,
            flattened,
            centroid,
            extents,
        )
    return distances.reshape(frame_count, vertex_count).numpy()


def contact_counts_from_min_distances(
    min_distances,
    ground_truth_contact,
    threshold_m,
):
    frame_count = min(len(min_distances), len(ground_truth_contact))
    prediction = min_distances[:frame_count] <= threshold_m
    truth = np.asarray(ground_truth_contact[:frame_count]) >= 0.5
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_contact_frames": int(prediction.sum()),
        "gt_contact_frames": int(truth.sum()),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2.0 * tp / max(2 * tp + fp + fn, 1),
        "min_clearance_mm": float(np.min(min_distances[:frame_count])) * 1000.0,
        "mean_hand_mesh_clearance_mm": (
            float(np.mean(min_distances[:frame_count])) * 1000.0
        ),
    }


def evaluate_npz(candidate_path, data_root, threshold_m, sdf_cache):
    payload = np.load(candidate_path, allow_pickle=True)
    required = {
        "pred_left_hand_verts",
        "pred_right_hand_verts",
        "obj_com_pos",
        "obj_rot_mat",
        "start_frame_idx",
    }
    missing = required.difference(payload.files)
    if missing:
        raise KeyError(
            f"{candidate_path} is missing dense v0 fields: {sorted(missing)}"
        )
    sequence = str(payload["seq_name"])
    object_name = str(payload["object_name"])
    start = int(payload["start_frame_idx"])
    ground_truth = np.load(
        data_root
        / "contact_labels_w_semantics_npy_files"
        / f"{sequence}.npy"
    )[:, :2]
    object_sdf = load_object_sdf(data_root, object_name, sdf_cache)
    rows = {
        "sequence": sequence,
        "object": object_name,
        "start_frame_idx": start,
    }
    for hand_index, hand_name in enumerate(HAND_NAMES):
        distances = dense_contact_distances(
            payload[f"pred_{hand_name}_hand_verts"],
            payload["obj_com_pos"],
            payload["obj_rot_mat"],
            object_sdf,
        )
        min_distances = distances.min(axis=1)
        gt_contact = ground_truth[
            start : start + len(min_distances),
            hand_index,
        ]
        rows[hand_name] = contact_counts_from_min_distances(
            min_distances,
            gt_contact,
            threshold_m,
        )
        rows[hand_name]["mean_contact_vertices"] = float(
            np.mean((distances <= threshold_m).sum(axis=1))
        )
        rows[hand_name]["mean_penetration_mm"] = (
            float(np.mean(np.maximum(-distances, 0.0))) * 1000.0
        )
    return rows


def pool_hand(rows, hand_name):
    tp = sum(row[hand_name]["tp"] for row in rows)
    fp = sum(row[hand_name]["fp"] for row in rows)
    fn = sum(row[hand_name]["fn"] for row in rows)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2.0 * tp / max(2 * tp + fp + fn, 1),
    }


def summarize_arm(rows, name):
    summary = {"arm": name, "sequence_count": len(rows), "pooled": {}}
    for hand_name in HAND_NAMES:
        summary["pooled"][hand_name] = pool_hand(rows, hand_name)
        summary[f"{hand_name}_macro_f1"] = float(np.mean([
            row[hand_name]["f1"] for row in rows
        ]))
        summary[f"{hand_name}_mean_hand_mesh_clearance_mm"] = float(
            np.mean([
                row[hand_name]["mean_hand_mesh_clearance_mm"]
                for row in rows
            ])
        )
        summary[f"{hand_name}_mean_penetration_mm"] = float(
            np.mean([
                row[hand_name]["mean_penetration_mm"]
                for row in rows
            ])
        )
    return summary


def bootstrap_interval(values, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def main():
    args = parse_args()
    data_root = Path(args.data_root_folder)
    selection = json.loads(
        Path(args.selection).read_text(encoding="utf-8")
    )
    selected_index = {
        row["sequence_name"]: int(row["selected_candidate_index"])
        for row in selection["selected"]
    }
    candidate_dirs = sorted(
        Path(args.candidate_root).glob("candidate_seed_*")
    )
    if not candidate_dirs:
        raise FileNotFoundError("No candidate_seed_* directories found")
    mappings = [candidate_files(path) for path in candidate_dirs]
    sequences = sorted(set(selected_index).intersection(mappings[0]))
    if not sequences:
        raise ValueError("No common candidate sequences")

    sdf_cache = {}
    all_rows = []
    for candidate_index, mapping in enumerate(mappings):
        rows = []
        for sequence in sequences:
            rows.append(
                evaluate_npz(
                    mapping[sequence],
                    data_root,
                    args.contact_threshold_m,
                    sdf_cache,
                )
            )
        all_rows.append(rows)

    baseline = all_rows[0]
    selected = [
        all_rows[selected_index[sequence]][index]
        for index, sequence in enumerate(sequences)
    ]
    oracle = []
    random_mean = []
    for index, sequence in enumerate(sequences):
        candidates = [rows[index] for rows in all_rows]
        oracle_row = {"sequence": sequence}
        oracle_row.update({
            hand_name: max(
                (row[hand_name] for row in candidates),
                key=lambda item: item["f1"],
            )
            for hand_name in HAND_NAMES
        })
        oracle.append(oracle_row)
        random_row = {"sequence": sequence}
        random_row.update({
            hand_name: {
                "tp": float(np.mean([
                    row[hand_name]["tp"] for row in candidates
                ])),
                "fp": float(np.mean([
                    row[hand_name]["fp"] for row in candidates
                ])),
                "fn": float(np.mean([
                    row[hand_name]["fn"] for row in candidates
                ])),
                "f1": float(np.mean([
                    row[hand_name]["f1"] for row in candidates
                ])),
                "mean_hand_mesh_clearance_mm": float(np.mean([
                    row[hand_name]["mean_hand_mesh_clearance_mm"]
                    for row in candidates
                ])),
                "mean_penetration_mm": float(np.mean([
                    row[hand_name]["mean_penetration_mm"]
                    for row in candidates
                ])),
            }
            for hand_name in HAND_NAMES
        })
        random_mean.append(random_row)

    summary = {
        "baseline": summarize_arm(baseline, "baseline"),
        "random": summarize_arm(random_mean, "random"),
        "selected": summarize_arm(selected, "selected"),
        "oracle": summarize_arm(oracle, "oracle"),
        "selected_minus_baseline": {},
        "selected_minus_random": {},
    }
    for hand_index, hand_name in enumerate(HAND_NAMES):
        baseline_diff = np.asarray([
            selected[row][hand_name]["f1"] - baseline[row][hand_name]["f1"]
            for row in range(len(sequences))
        ])
        random_diff = np.asarray([
            selected[row][hand_name]["f1"]
            - random_mean[row][hand_name]["f1"]
            for row in range(len(sequences))
        ])
        summary["selected_minus_baseline"][hand_name] = {
            "mean": float(baseline_diff.mean()),
            "ci95": bootstrap_interval(
                baseline_diff,
                args.bootstrap_samples,
                args.seed + hand_index,
            ),
        }
        summary["selected_minus_random"][hand_name] = {
            "mean": float(random_diff.mean()),
            "ci95": bootstrap_interval(
                random_diff,
                args.bootstrap_samples,
                args.seed + 10 + hand_index,
            ),
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "args": vars(args),
                "summary": summary,
                "baseline": baseline,
                "selected": selected,
                "random_mean": random_mean,
                "oracle": oracle,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
