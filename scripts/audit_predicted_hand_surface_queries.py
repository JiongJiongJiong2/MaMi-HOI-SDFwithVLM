"""Audit stored predicted hand queries against canonical object triangles."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from manip.model.sdf_utils import point_to_triangle_unsigned_distance


HAND_COLUMN = {"left": 0, "right": 1}
METRIC_VERSION = "predicted_proxy_to_triangle_v2"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--data_root_folder", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--contact_threshold_m", type=float, default=0.05)
    parser.add_argument("--point_chunk", type=int, default=1024)
    parser.add_argument("--triangle_chunk", type=int, default=512)
    parser.add_argument(
        "--disable_triangle_kdtree",
        action="store_true",
        help="Use exhaustive triangle evaluation instead of bounded KNN pruning.",
    )
    parser.add_argument("--kdtree_neighbors", type=int, default=64)
    parser.add_argument("--max_sequences", type=int, default=0)
    parser.add_argument(
        "--sequence",
        action="append",
        default=[],
        help="Restrict analysis to one or more sequence names.",
    )
    return parser.parse_args()


def load_query_rows(path, max_sequences=0, sequences=()):
    requested_sequences = set(sequences)
    grouped = {}
    selected_sequences = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sequence = row["sequence"]
            if requested_sequences and sequence not in requested_sequences:
                continue
            if max_sequences and sequence not in selected_sequences:
                if len(selected_sequences) >= max_sequences:
                    continue
                selected_sequences.add(sequence)
            elif not max_sequences:
                selected_sequences.add(sequence)
            object_name = row["object"]
            grouped.setdefault(object_name, []).append(row)
    if not grouped:
        raise ValueError(f"No query rows found in {path}")
    return grouped


def load_triangles(data_root_folder, object_name):
    import trimesh

    mesh_path = (
        Path(data_root_folder)
        / "rest_object_geo"
        / f"{object_name}.ply"
    )
    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    triangles = np.asarray(mesh.triangles, dtype=np.float32)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(
            f"Expected [F, 3, 3] triangles from {mesh_path}, "
            f"got {triangles.shape}"
        )
    return torch.from_numpy(triangles)


def point_to_triangle_distance_kdtree(
    points,
    triangles,
    point_chunk=1024,
    neighbors=64,
):
    """Clearance using a bounded number of nearest triangle centroids."""
    from scipy.spatial import cKDTree

    centers = triangles.mean(dim=1).cpu().numpy()
    tree = cKDTree(centers)
    neighbor_count = min(int(neighbors), len(centers))

    distances = []
    for start in range(0, len(points), point_chunk):
        query = points[start : start + point_chunk]
        query_numpy = query.cpu().numpy()
        _, candidate_lists = tree.query(
            query_numpy,
            k=neighbor_count,
        )
        if neighbor_count == 1:
            candidate_lists = candidate_lists[:, None]
        for point, candidates in zip(query, candidate_lists):
            candidate_triangles = triangles[
                torch.as_tensor(candidates, dtype=torch.long)
            ]
            point_distance, _ = point_to_triangle_unsigned_distance(
                point[None, :],
                candidate_triangles,
                point_chunk=1,
                triangle_chunk=max(1, neighbor_count),
            )
            distances.append(point_distance[0])
    return torch.stack(distances)


def _empty_hand_stats():
    return {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "contact_clearances_mm": [],
    }


def _finalize_hand_stats(stats):
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    clearances = stats["contact_clearances_mm"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "gt_contact_frame_count": tp + fn,
        "predicted_contact_frame_count": tp + fp,
        "tp": tp,
        "fp": fp,
        "tn": stats["tn"],
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_clearance_on_gt_contact_mm": (
            float(np.mean(clearances)) if clearances else None
        ),
    }


def summarize_predictions(predictions, contact_threshold_m):
    sequence_stats = {}
    for prediction in predictions:
        key = (prediction["sequence"], prediction["hand"])
        stats = sequence_stats.setdefault(key, _empty_hand_stats())
        gt_contact = prediction["gt_contact"]
        pred_contact = (
            prediction["distance_m"] < contact_threshold_m
        )
        if gt_contact and pred_contact:
            stats["tp"] += 1
        elif (not gt_contact) and pred_contact:
            stats["fp"] += 1
        elif (not gt_contact) and (not pred_contact):
            stats["tn"] += 1
        else:
            stats["fn"] += 1
        if gt_contact:
            stats["contact_clearances_mm"].append(
                prediction["distance_m"] * 1000.0
            )

    rows = []
    for (sequence, hand), stats in sorted(sequence_stats.items()):
        summary = _finalize_hand_stats(stats)
        summary["sequence"] = sequence
        summary["hand"] = hand
        rows.append(summary)
    return rows


def aggregate_sequence_rows(rows):
    aggregate = {}
    for hand in ("left", "right"):
        hand_rows = [row for row in rows if row["hand"] == hand]
        macro_values = [
            row["mean_clearance_on_gt_contact_mm"]
            for row in hand_rows
            if row["mean_clearance_on_gt_contact_mm"] is not None
        ]
        totals = {
            key: sum(int(row[key]) for row in hand_rows)
            for key in ("tp", "fp", "tn", "fn")
        }
        precision = (
            totals["tp"] / (totals["tp"] + totals["fp"])
            if totals["tp"] + totals["fp"]
            else 0.0
        )
        recall = (
            totals["tp"] / (totals["tp"] + totals["fn"])
            if totals["tp"] + totals["fn"]
            else 0.0
        )
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        aggregate[hand] = {
            "sequence_count_with_gt_contact": len(macro_values),
            "mean_clearance_on_gt_contact_macro_mm": (
                float(np.mean(macro_values)) if macro_values else None
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            **totals,
        }
    return aggregate


def compute_predictions(
    grouped_rows,
    data_root_folder,
    contact_threshold_m,
    point_chunk,
    triangle_chunk,
    use_triangle_kdtree,
    kdtree_neighbors,
):
    predictions = []
    contact_cache = {}
    for object_name, rows in sorted(grouped_rows.items()):
        triangles = load_triangles(data_root_folder, object_name)
        points = torch.tensor(
            [
                [
                    row["canonical_x"],
                    row["canonical_y"],
                    row["canonical_z"],
                ]
                for row in rows
            ],
            dtype=torch.float32,
        )
        if use_triangle_kdtree:
            distances = point_to_triangle_distance_kdtree(
                points,
                triangles,
                point_chunk=point_chunk,
                neighbors=kdtree_neighbors,
            )
        else:
            distances, _ = point_to_triangle_unsigned_distance(
                points,
                triangles,
                point_chunk=point_chunk,
                triangle_chunk=triangle_chunk,
            )
        for row, distance in zip(rows, distances.tolist()):
            sequence = row["sequence"]
            if sequence not in contact_cache:
                contact_path = (
                    Path(data_root_folder)
                    / "contact_labels_w_semantics_npy_files"
                    / f"{sequence}.npy"
                )
                if not contact_path.is_file():
                    raise FileNotFoundError(contact_path)
                contact_cache[sequence] = np.load(contact_path)
            contact_labels = contact_cache[sequence]
            frame = int(row["frame"])
            if frame < 0 or frame >= len(contact_labels):
                raise IndexError(
                    f"{sequence}: frame {frame} outside "
                    f"contact array length {len(contact_labels)}"
                )
            hand = row["hand"]
            predictions.append(
                {
                    "sequence": sequence,
                    "hand": hand,
                    "frame": frame,
                    "object": object_name,
                    "distance_m": float(distance),
                    "gt_contact": bool(
                        contact_labels[frame, HAND_COLUMN[hand]] > 0.5
                    ),
                }
            )
    return predictions


def write_outputs(output_dir, rows, aggregate, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    csv_path = output_dir / "predicted_proxy_metrics_v2_sequences.csv"
    fieldnames = [
        "sequence",
        "hand",
        "gt_contact_frame_count",
        "predicted_contact_frame_count",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_clearance_on_gt_contact_mm",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "metric_version": METRIC_VERSION,
        "metadata": metadata,
        "aggregate": aggregate,
        "sequence_count": len({
            row["sequence"] for row in rows
        }),
    }
    summary_path = output_dir / "predicted_proxy_metrics_v2_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def main():
    args = parse_args()
    if args.contact_threshold_m <= 0:
        raise ValueError("--contact_threshold_m must be positive")
    grouped_rows = load_query_rows(
        args.queries,
        max_sequences=args.max_sequences,
        sequences=args.sequence,
    )
    predictions = compute_predictions(
        grouped_rows,
        args.data_root_folder,
        args.contact_threshold_m,
        args.point_chunk,
        args.triangle_chunk,
        not args.disable_triangle_kdtree,
        args.kdtree_neighbors,
    )
    rows = summarize_predictions(
        predictions,
        args.contact_threshold_m,
    )
    aggregate = aggregate_sequence_rows(rows)
    metadata = {
        "queries": str(Path(args.queries).resolve()),
        "data_root_folder": str(Path(args.data_root_folder).resolve()),
        "contact_threshold_m": args.contact_threshold_m,
        "max_sequences": args.max_sequences,
        "sequence_filter": sorted(args.sequence),
        "distance_definition": (
            "stored canonical hand proxy point to canonical object triangle"
        ),
        "triangle_pruning": (
            f"bounded_knn_centroids_k{args.kdtree_neighbors}"
            if not args.disable_triangle_kdtree
            else "exhaustive"
        ),
        "ground_truth_grouping": (
            "contact_labels_w_semantics_npy_files columns 0/1"
        ),
    }
    write_outputs(args.output_dir, rows, aggregate, metadata)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
