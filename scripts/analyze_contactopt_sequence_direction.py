#!/usr/bin/env python3
"""Decompose E5-to-E5T finger updates into normal and tangential motion."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np


METRICS = (
    "near_signed_normal_mm",
    "near_outward_normal_mm",
    "near_inward_normal_mm",
    "near_tangential_mm",
    "near_total_motion_mm",
    "near_outward_fraction",
    "near_inward_fraction",
    "near_distance_change_mm",
    "near_vertex_count",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument("--e5-root", type=Path, required=True)
    parser.add_argument("--e5t-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mano-root", type=Path)
    parser.add_argument("--manopth-root", type=Path)
    parser.add_argument("--near-threshold-m", type=float, default=0.02)
    parser.add_argument("--split", nargs="+", default=("train", "dev"))
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else None


def nearest_object_vertices(points, objects):
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        distances, indices = cKDTree(objects).query(points, k=1)
        return distances, indices

    distances = np.empty(len(points), dtype=np.float64)
    indices = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), 64):
        stop = start + 64
        squared = np.sum(
            (
                points[start:stop, None, :]
                - objects[None, :, :]
            )
            ** 2,
            axis=2,
        )
        local_indices = np.argmin(squared, axis=1)
        indices[start:stop] = local_indices
        distances[start:stop] = np.sqrt(
            squared[np.arange(len(local_indices)), local_indices]
        )
    return distances, indices


def direction_statistics(
    base_vertices,
    candidate_vertices,
    object_vertices,
    near_threshold_m,
):
    """Measure candidate motion relative to the nearest object vertex."""
    base = np.asarray(base_vertices, dtype=np.float64)
    candidate = np.asarray(candidate_vertices, dtype=np.float64)
    objects = np.asarray(object_vertices, dtype=np.float64)
    if base.shape != candidate.shape:
        raise ValueError("base and candidate vertices must align")
    if objects.ndim != 2 or objects.shape[1] != 3:
        raise ValueError("object vertices must have shape [N, 3]")
    if near_threshold_m <= 0:
        raise ValueError("near_threshold_m must be positive")

    distances, nearest_indices = nearest_object_vertices(base, objects)
    nearest = objects[nearest_indices]
    vectors = base - nearest
    vector_norms = np.linalg.norm(vectors, axis=1)
    valid_normal = vector_norms > 1e-9
    normals = np.zeros_like(vectors)
    normals[valid_normal] = (
        vectors[valid_normal] / vector_norms[valid_normal, None]
    )

    near = distances <= near_threshold_m
    selected = near & valid_normal
    if not np.any(selected):
        return {
            "near_signed_normal_mm": None,
            "near_outward_normal_mm": None,
            "near_inward_normal_mm": None,
            "near_tangential_mm": None,
            "near_total_motion_mm": None,
            "near_outward_fraction": None,
            "near_inward_fraction": None,
            "near_distance_change_mm": None,
            "near_vertex_count": int(near.sum()),
            "valid_normal_vertex_count": 0,
        }

    displacement = candidate - base
    normal_displacement = np.einsum(
        "ij,ij->i",
        displacement,
        normals,
    )
    tangent_displacement = (
        displacement
        - normal_displacement[:, None] * normals
    )
    candidate_distances = nearest_object_vertices(candidate, objects)[0]

    signed_normal = normal_displacement[selected] * 1000.0
    return {
        "near_signed_normal_mm": float(signed_normal.mean()),
        "near_outward_normal_mm": float(
            np.maximum(signed_normal, 0.0).mean()
        ),
        "near_inward_normal_mm": float(
            np.maximum(-signed_normal, 0.0).mean()
        ),
        "near_tangential_mm": float(
            np.linalg.norm(
                tangent_displacement[selected],
                axis=1,
            ).mean()
            * 1000.0
        ),
        "near_total_motion_mm": float(
            np.linalg.norm(
                displacement[selected],
                axis=1,
            ).mean()
            * 1000.0
        ),
        "near_outward_fraction": float(
            np.mean(signed_normal > 0.0)
        ),
        "near_inward_fraction": float(
            np.mean(signed_normal < 0.0)
        ),
        "near_distance_change_mm": float(
            (
                candidate_distances[selected]
                - distances[selected]
            ).mean()
            * 1000.0
        ),
        "near_vertex_count": int(near.sum()),
        "valid_normal_vertex_count": int(selected.sum()),
    }


def aggregate_frames(frames, metrics=METRICS):
    frames = list(frames)
    if not frames:
        raise ValueError("cannot aggregate an empty frame list")
    return {
        metric: finite_mean([frame[metric] for frame in frames])
        for metric in metrics
    }


def cluster_bootstrap_mean(values, groups, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    if len(unique_groups) == 0:
        return None
    group_values = [
        values[inverse == index]
        for index in range(len(unique_groups))
    ]
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        selected = rng.integers(
            0,
            len(unique_groups),
            size=len(unique_groups),
        )
        draws[draw] = np.concatenate(
            [group_values[index] for index in selected]
        ).mean()
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def group_difference(
    first,
    second,
    metric,
    bootstrap_samples,
    seed,
):
    first_values = np.asarray(
        [row[metric] for row in first if row[metric] is not None],
        dtype=np.float64,
    )
    second_values = np.asarray(
        [row[metric] for row in second if row[metric] is not None],
        dtype=np.float64,
    )
    if not len(first_values) or not len(second_values):
        return None

    observed = float(first_values.mean() - second_values.mean())
    if bootstrap_samples <= 0:
        return {"mean": observed, "ci95": None}

    first_groups = [
        row["sequence"] for row in first if row[metric] is not None
    ]
    second_groups = [
        row["sequence"] for row in second if row[metric] is not None
    ]
    rng = np.random.default_rng(seed)
    first_unique = np.unique(first_groups)
    second_unique = np.unique(second_groups)
    first_group_values = [
        first_values[np.asarray(first_groups) == group]
        for group in first_unique
    ]
    second_group_values = [
        second_values[np.asarray(second_groups) == group]
        for group in second_unique
    ]
    draws = np.empty(bootstrap_samples, dtype=np.float64)
    for draw in range(bootstrap_samples):
        first_selected = rng.integers(
            0,
            len(first_unique),
            size=len(first_unique),
        )
        second_selected = rng.integers(
            0,
            len(second_unique),
            size=len(second_unique),
        )
        first_draw = np.concatenate(
            [first_group_values[index] for index in first_selected]
        ).mean()
        second_draw = np.concatenate(
            [second_group_values[index] for index in second_selected]
        ).mean()
        draws[draw] = first_draw - second_draw
    return {
        "mean": observed,
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }


def summarize_group(rows):
    return {
        "window_count": len(rows),
        "sequence_count": len({row["sequence"] for row in rows}),
        **{
            metric: finite_mean([row[metric] for row in rows])
            for metric in METRICS
        },
    }


def install_hand_object_pickle_compatibility():
    package = types.ModuleType("contactopt")
    package.__path__ = []
    module = types.ModuleType("contactopt.hand_object")

    class HandObject:
        pass

    module.HandObject = HandObject
    sys.modules["contactopt"] = package
    sys.modules["contactopt.hand_object"] = module


def load_runs(path):
    install_hand_object_pickle_compatibility()
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def numpy_legacy_aliases():
    aliases = {
        "bool": bool,
        "object": object,
        "str": str,
        "unicode": str,
        "int": int,
        "float": float,
        "complex": complex,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def make_mano_layer(mano_root, manopth_root):
    numpy_legacy_aliases()
    if manopth_root is not None:
        sys.path.insert(0, str(manopth_root))
    from manopth.manolayer import ManoLayer

    return ManoLayer(
        mano_root=str(mano_root),
        use_pca=True,
        ncomps=15,
        side="right",
        flat_hand_mean=False,
    )


def mano_vertices(poses, betas, transforms, layer, batch_size=16):
    import torch

    poses = np.ascontiguousarray(poses, dtype=np.float32)
    betas = np.ascontiguousarray(betas, dtype=np.float32)
    transforms = np.ascontiguousarray(transforms, dtype=np.float32)
    if len(poses) != len(betas) or len(poses) != len(transforms):
        raise ValueError("MANO inputs must have equal frame counts")

    output = []
    with torch.no_grad():
        for start in range(0, len(poses), batch_size):
            stop = start + batch_size
            pose = torch.from_numpy(poses[start:stop])
            beta = torch.from_numpy(betas[start:stop])
            transform = torch.from_numpy(transforms[start:stop])
            vertices, _ = layer(pose, beta)
            homogeneous = torch.cat(
                [
                    vertices / 1000.0,
                    torch.ones(
                        len(vertices),
                        vertices.shape[1],
                        1,
                    ),
                ],
                dim=2,
            )
            transformed = torch.bmm(
                transform,
                homogeneous.permute(0, 2, 1),
            ).permute(0, 2, 1)
            output.append(transformed[:, :, :3].cpu().numpy())
    return np.concatenate(output, axis=0)


def validate_mano_reconstruction(runs, layer):
    indices = sorted({0, len(runs) - 1})
    poses = np.asarray(
        [runs[index]["out_ho"].hand_pose for index in indices],
        dtype=np.float32,
    )
    betas = np.asarray(
        [runs[index]["out_ho"].hand_beta for index in indices],
        dtype=np.float32,
    )
    transforms = np.asarray(
        [runs[index]["out_ho"].hand_mTc for index in indices],
        dtype=np.float32,
    )
    expected = np.asarray(
        [runs[index]["out_ho"].hand_verts for index in indices],
        dtype=np.float32,
    )
    actual = mano_vertices(poses, betas, transforms, layer)
    return float(np.max(np.abs(actual - expected)))


def frame_metrics_for_chunk(
    chunk_id,
    frames,
    base_poses,
    candidate_poses,
    runs,
    object_vertices,
    layer,
    near_threshold_m,
):
    betas = np.asarray(
        [run["out_ho"].hand_beta for run in runs],
        dtype=np.float32,
    )
    transforms = np.asarray(
        [run["out_ho"].hand_mTc for run in runs],
        dtype=np.float32,
    )
    base_vertices = mano_vertices(
        base_poses,
        betas,
        transforms,
        layer,
    )
    candidate_vertices = mano_vertices(
        candidate_poses,
        betas,
        transforms,
        layer,
    )
    if len(base_vertices) != len(frames):
        raise ValueError(f"{chunk_id} frame count mismatch")
    if object_vertices.shape[0] != len(frames):
        raise ValueError(f"{chunk_id} object frame count mismatch")

    return [
        {
            "frame": int(frame),
            **direction_statistics(
                base_vertices[index],
                candidate_vertices[index],
                object_vertices[index],
                near_threshold_m,
            ),
        }
        for index, frame in enumerate(frames)
    ]


def load_frame_cache(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "frames" not in payload or "frame_metrics" not in payload:
        raise ValueError(f"invalid frame cache: {path}")
    return payload


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    args = parse_args()
    if args.near_threshold_m <= 0:
        raise ValueError("near_threshold_m must be positive")
    if args.mano_root is None:
        raise ValueError("--mano-root is required")

    batch = json.loads(
        args.batch_summary.read_text(encoding="utf-8")
    )
    if batch["completed_chunk_count"] != batch["chunk_count"]:
        raise ValueError("batch summary is incomplete")
    e5_result_path = args.e5_root / "result.json"
    e5t_result_path = args.e5t_root / "result.json"
    e5_result = json.loads(e5_result_path.read_text(encoding="utf-8"))
    e5t_result = json.loads(e5t_result_path.read_text(encoding="utf-8"))

    window_key = lambda row: (
        row["chunk_id"],
        tuple(int(frame) for frame in row["frames"]),
    )
    e5_windows = {window_key(row): row for row in e5_result["windows"]}
    e5t_windows = {window_key(row): row for row in e5t_result["windows"]}
    if set(e5_windows) != set(e5t_windows):
        raise ValueError("E5 and E5-T window sets differ")

    layer = make_mano_layer(args.mano_root, args.manopth_root)
    frame_cache_dir = args.output_dir / "frames"
    frame_cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    reconstruction_errors = {}
    selected_splits = set(args.split)

    for index, chunk in enumerate(batch["rows"], start=1):
        if chunk["split"] not in selected_splits:
            continue
        chunk_id = chunk["chunk_id"]
        cache_path = frame_cache_dir / f"{chunk_id}.json"
        if args.resume and cache_path.is_file():
            cached = load_frame_cache(cache_path)
            frames = [int(frame) for frame in cached["frames"]]
            frame_metrics = cached["frame_metrics"]
            reconstruction_errors[chunk_id] = cached.get(
                "mano_reconstruction_max_error_m"
            )
        else:
            frames = [int(frame) for frame in chunk["frames"]]
            with np.load(args.e5_root / "optimized" / (
                f"{chunk_id}_optimized.npz"
            )) as archive:
                base_poses = np.asarray(
                    archive["hand_pose"],
                    dtype=np.float32,
                )
                base_frames = [
                    int(frame) for frame in archive["frames"]
                ]
            with np.load(args.e5t_root / "optimized" / (
                f"{chunk_id}_optimized.npz"
            )) as archive:
                candidate_poses = np.asarray(
                    archive["hand_pose"],
                    dtype=np.float32,
                )
                candidate_frames = [
                    int(frame) for frame in archive["frames"]
                ]
            if frames != base_frames or frames != candidate_frames:
                raise ValueError(f"{chunk_id} optimized frame mismatch")

            runs = load_runs(chunk["optimized_pkl"])
            geometry_path = args.geometry_dir / f"{chunk_id}.npz"
            with np.load(geometry_path, allow_pickle=True) as archive:
                object_vertices = np.asarray(
                    archive["object_vertices"],
                    dtype=np.float32,
                )
            reconstruction_error = validate_mano_reconstruction(
                runs,
                layer,
            )
            if reconstruction_error > 1e-6:
                raise ValueError(
                    f"{chunk_id} MANO reconstruction error "
                    f"{reconstruction_error}"
                )
            frame_metrics = frame_metrics_for_chunk(
                chunk_id,
                frames,
                base_poses,
                candidate_poses,
                runs,
                object_vertices,
                layer,
                args.near_threshold_m,
            )
            reconstruction_errors[chunk_id] = reconstruction_error
            write_json(
                cache_path,
                {
                    "chunk_id": chunk_id,
                    "frames": frames,
                    "mano_reconstruction_max_error_m": (
                        reconstruction_error
                    ),
                    "frame_metrics": frame_metrics,
                },
            )

        frame_by_index = {
            int(frame["frame"]): frame
            for frame in frame_metrics
        }
        for window_frames in chunk["windows"]:
            key = (
                chunk_id,
                tuple(int(frame) for frame in window_frames),
            )
            selected_frames = [
                frame_by_index[int(frame)]
                for frame in window_frames
            ]
            e5 = e5_windows[key]
            e5t = e5t_windows[key]
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "sequence": chunk["sequence"],
                    "split": chunk["split"],
                    "object_name": chunk["object_name"],
                    "frames": list(key[1]),
                    "e5_contact_pass": bool(
                        e5["optimized_contact_gate"]["passed"]
                    ),
                    "e5t_contact_pass": bool(
                        e5t["optimized_contact_gate"]["passed"]
                    ),
                    "e5_temporal_pass": bool(
                        e5["optimized_temporal_gate"]["passed"]
                    ),
                    "e5t_temporal_pass": bool(
                        e5t["optimized_temporal_gate"]["passed"]
                    ),
                    "e5_contact_improved_frames": int(
                        e5["optimized_contact_gate"][
                            "contact_improved_frames"
                        ]
                    ),
                    "e5t_contact_improved_frames": int(
                        e5t["optimized_contact_gate"][
                            "contact_improved_frames"
                        ]
                    ),
                    **aggregate_frames(selected_frames),
                }
            )
        print(
            f"[{index}/{len(batch['rows'])}] {chunk_id}",
            flush=True,
        )

    stable = [
        row for row in rows
        if row["e5_contact_pass"] and row["e5t_contact_pass"]
    ]
    regression = [
        row for row in rows
        if row["e5_contact_pass"] and not row["e5t_contact_pass"]
    ]
    gain = [
        row for row in rows
        if not row["e5_contact_pass"] and row["e5t_contact_pass"]
    ]
    common_failure = [
        row for row in rows
        if not row["e5_contact_pass"] and not row["e5t_contact_pass"]
    ]
    groups = {
        "all": rows,
        "e5_contact_pass": stable + regression,
        "contact_stable": stable,
        "contact_regression": regression,
        "contact_gain": gain,
        "contact_common_failure": common_failure,
    }
    differences = {
        metric: group_difference(
            regression,
            stable,
            metric,
            args.bootstrap_samples,
            args.seed + offset,
        )
        for offset, metric in enumerate(METRICS)
    }
    direction_gate = {
        "criterion": (
            "contact_regression_minus_stable "
            "near_signed_normal_mm ci95 lower bound > 0"
        ),
        "pass": bool(
            differences["near_signed_normal_mm"] is not None
            and differences["near_signed_normal_mm"]["ci95"] is not None
            and differences["near_signed_normal_mm"]["ci95"][0] > 0
        ),
    }
    summary = {
        "method": (
            "re-run MANO from frozen E5/E5-T poses; decompose each "
            "hand-vertex displacement using the unit vector from the "
            "nearest ContactOpt object vertex at E5"
        ),
        "near_threshold_m": args.near_threshold_m,
        "splits": sorted(selected_splits),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "inputs": {
            "batch_summary": str(args.batch_summary),
            "batch_summary_sha256": sha256_file(args.batch_summary),
            "e5_result": str(e5_result_path),
            "e5_result_sha256": sha256_file(e5_result_path),
            "e5t_result": str(e5t_result_path),
            "e5t_result_sha256": sha256_file(e5t_result_path),
            "geometry_dir": str(args.geometry_dir),
        },
        "mano_reconstruction_max_error_m": max(
            value for value in reconstruction_errors.values()
            if value is not None
        ),
        "window_count": len(rows),
        "groups": {
            name: summarize_group(group)
            for name, group in groups.items()
        },
        "contact_regression_minus_stable": differences,
        "direction_gate": direction_gate,
        "windows": rows,
        "decision": "GO" if direction_gate["pass"] else "NO-GO",
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({
        "decision": summary["decision"],
        "direction_gate": summary["direction_gate"],
        "contact_regression_minus_stable": (
            summary["contact_regression_minus_stable"]
        ),
        "groups": summary["groups"],
        "output_json": str(args.output_dir / "summary.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
