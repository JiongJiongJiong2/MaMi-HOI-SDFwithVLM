"""Run the minimal contact-conditioned sectional-prior diagnostic.

This is deliberately not a diffusion training entry point.  It asks a cheaper
question first: on frames where both palms contact the same object, does the
gravity/contact-aligned chord rank the other palm's surface region better than
random ordering or a simple central-antipode heuristic?

Synthetic mode has no third-party dependency:

    python scripts/run_sectional_prior_diagnostic.py --synthetic

Real OMOMO mode uses the repository's processed test windows and canonical
meshes, and therefore requires the normal project environment:

    python scripts/run_sectional_prior_diagnostic.py \
      --data_root_folder /path/to/processed_data \
      --split_manifest /path/to/eval_split.json \
      --eval_split validation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from manip.geometry.sectional_prior import (  # noqa: E402
    METHODS,
    evaluate_contact_pair,
    make_box_surface_candidates,
    summarize_results,
    world_to_object,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run a dependency-free box smoke experiment instead of OMOMO data.",
    )
    parser.add_argument(
        "--data_root_folder",
        default="",
        help="Processed-data root used by CanoObjectTrajDataset.",
    )
    parser.add_argument(
        "--split_manifest",
        default="",
        help="Sequence-disjoint manifest from create_experiment_split_manifest.py.",
    )
    parser.add_argument(
        "--eval_split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--candidate_count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top_k", type=int, nargs="+", default=(1, 5, 10))
    parser.add_argument("--hit_radius_ratio", type=float, default=0.08)
    parser.add_argument("--exclusion_ratio", type=float, default=0.08)
    parser.add_argument("--max_surface_distance_ratio", type=float, default=0.12)
    parser.add_argument("--normal_weight", type=float, default=0.25)
    parser.add_argument(
        "--min_top5_gain",
        type=float,
        default=0.05,
        help="Go/no-go margin over center_antipode on sequence-macro top-5 hit.",
    )
    parser.add_argument(
        "--max_median_chord_distance",
        type=float,
        default=0.12,
        help="Maximum normalized median target-to-chord distance for a PASS.",
    )
    parser.add_argument(
        "--output_dir",
        default="sectional_prior_eval/synthetic",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.synthetic and (not args.data_root_folder or not args.split_manifest):
        raise ValueError(
            "Real-data mode requires --data_root_folder and --split_manifest. "
            "Use --synthetic for the dependency-free smoke test."
        )
    if args.window < 30:
        raise ValueError("--window must be at least 30")
    if args.frame_stride <= 0:
        raise ValueError("--frame_stride must be positive")
    if args.max_samples <= 0:
        raise ValueError("--max_samples must be positive")
    if args.max_samples % 2 != 0:
        raise ValueError(
            "--max_samples must be even because each accepted frame contributes "
            "both left-to-right and right-to-left samples"
        )
    if args.candidate_count < 32:
        raise ValueError("--candidate_count must be at least 32")
    if any(top_k <= 0 for top_k in args.top_k):
        raise ValueError("--top_k values must be positive")
    for name in (
        "hit_radius_ratio",
        "max_surface_distance_ratio",
        "max_median_chord_distance",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name} must be positive")
    if not 0.0 <= args.exclusion_ratio < 1.0:
        raise ValueError("--exclusion_ratio must lie in [0, 1)")
    if args.normal_weight < 0.0:
        raise ValueError("--normal_weight must be non-negative")


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join((str(base_seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _surface_grid_values(half_extent: float, steps: int) -> List[float]:
    return [
        -half_extent + 2.0 * half_extent * index / (steps - 1)
        for index in range(steps)
    ]


def build_synthetic_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    half_extents = (1.0, 0.75, 0.5)
    points, normals = make_box_surface_candidates(half_extents, steps=9)
    centre = (0.0, 0.0, 0.0)
    extent = 2.0 * max(half_extents)
    up = (0.0, 0.0, 1.0)
    y_values = _surface_grid_values(half_extents[1], 9)[2:7:2]
    z_values = _surface_grid_values(half_extents[2], 9)[2:7:2]
    contact_pairs = []
    for y in y_values:
        for z in z_values:
            # Same-height opposing boundary point on the horizontal chord.
            contact_pairs.append(((half_extents[0], y, z), (-half_extents[0], -y, z)))

    rows: List[Dict[str, object]] = []
    for frame_index, (left, right) in enumerate(contact_pairs):
        for source_hand, source, target in (
            ("left", left, right),
            ("right", right, left),
        ):
            rows.append(
                evaluate_contact_pair(
                    sequence_name="synthetic_box_antipodal",
                    absolute_frame=frame_index,
                    source_hand=source_hand,
                    source_point=source,
                    target_point=target,
                    candidates=points,
                    normals=normals,
                    centre=centre,
                    up=up,
                    extent=extent,
                    seed=stable_seed(args.seed, frame_index, source_hand),
                    top_ks=args.top_k,
                    hit_radius_ratio=args.hit_radius_ratio,
                    exclusion_ratio=args.exclusion_ratio,
                    normal_weight=args.normal_weight,
                )
            )
    return rows, {"synthetic_pairs": len(contact_pairs), "accepted_ordered_samples": len(rows)}


def _require_real_dependencies():
    try:
        import joblib  # type: ignore
        import numpy as np  # type: ignore
        import trimesh  # type: ignore
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Real-data mode requires the normal MaMi-HOI environment with "
            "numpy, joblib, and trimesh. Install requirements in the project "
            "Conda environment, or run with --synthetic first."
        ) from error
    return joblib, np, trimesh


def _load_mesh(trimesh_module, mesh_path: Path):
    mesh = trimesh_module.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh_module.Scene):
        if not mesh.geometry:
            raise ValueError(f"Mesh scene is empty: {mesh_path}")
        mesh = trimesh_module.util.concatenate(tuple(mesh.geometry.values()))
    if len(mesh.vertices) < 3 or len(mesh.faces) < 1:
        raise ValueError(f"Mesh has insufficient geometry: {mesh_path}")
    return mesh


def _sample_surface_candidates(np, mesh, count: int, seed: int):
    """Deterministic area-weighted triangle sampling without rtree."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    cross_products = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_areas = np.linalg.norm(cross_products, axis=1)
    valid = double_areas > 1e-12
    if not bool(valid.any()):
        raise ValueError("Mesh contains no non-degenerate triangle")
    triangles = triangles[valid]
    cross_products = cross_products[valid]
    probabilities = double_areas[valid] / double_areas[valid].sum()
    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(triangles), size=count, replace=True, p=probabilities)
    selected = triangles[face_indices]
    first = rng.random(count)
    second = rng.random(count)
    sqrt_first = np.sqrt(first)
    barycentric = np.stack(
        (1.0 - sqrt_first, sqrt_first * (1.0 - second), sqrt_first * second),
        axis=1,
    )
    points = (selected * barycentric[:, :, None]).sum(axis=1)
    face_normals = cross_products / np.linalg.norm(cross_products, axis=1, keepdims=True)
    normals = face_normals[face_indices]
    return points, normals


def _rotation_vector_to_object(np, vector_world, rotation) -> Tuple[float, float, float]:
    result = np.asarray(vector_world, dtype=np.float64) @ np.asarray(rotation, dtype=np.float64)
    return tuple(float(value) for value in result)


def _iter_real_windows(data: Mapping[object, Mapping[str, object]]) -> Iterable[Mapping[str, object]]:
    return iter(
        sorted(
            data.values(),
            key=lambda item: (str(item["seq_name"]), int(item.get("start_t_idx", 0))),
        )
    )


def build_real_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    joblib, np, trimesh = _require_real_dependencies()
    data_root = Path(args.data_root_folder)
    manifest_path = Path(args.split_manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_key = f"{args.eval_split}_sequences"
    if split_key not in manifest:
        raise KeyError(f"{manifest_path} is missing {split_key}")
    allowed_sequences = set(str(name) for name in manifest[split_key])
    if not allowed_sequences:
        raise ValueError(f"{split_key} is empty in {manifest_path}")

    processed_path = data_root / f"cano_test_diffusion_manip_window_{args.window}_joints24.p"
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Missing {processed_path}. Run the normal MaMi-HOI preprocessing first."
        )
    windows = joblib.load(processed_path)
    if not isinstance(windows, Mapping):
        raise TypeError(f"Expected a mapping in {processed_path}, got {type(windows).__name__}")

    mesh_cache: Dict[str, Dict[str, object]] = {}
    contact_cache: Dict[str, object] = {}
    seen_frames = set()
    rows: List[Dict[str, object]] = []
    counts = {
        "windows_seen": 0,
        "frames_seen": 0,
        "duplicate_frames": 0,
        "non_bimanual_frames": 0,
        "surface_distance_rejections": 0,
        "degenerate_frame_rejections": 0,
        "accepted_ordered_samples": 0,
    }

    for item in _iter_real_windows(windows):
        sequence_name = str(item["seq_name"])
        if sequence_name not in allowed_sequences:
            continue
        counts["windows_seen"] += 1
        object_parts = sequence_name.split("_")
        if len(object_parts) < 2:
            raise ValueError(f"Cannot derive object name from sequence {sequence_name!r}")
        object_name = object_parts[1]

        if object_name not in mesh_cache:
            mesh_path = data_root / "rest_object_geo" / f"{object_name}.ply"
            if not mesh_path.exists():
                raise FileNotFoundError(f"Canonical object mesh is missing: {mesh_path}")
            mesh = _load_mesh(trimesh, mesh_path)
            sampled_points, sampled_normals = _sample_surface_candidates(
                np,
                mesh,
                args.candidate_count,
                stable_seed(args.seed, object_name),
            )
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            centre = 0.5 * (bounds[0] + bounds[1])
            extent = float((bounds[1] - bounds[0]).max())
            if not math.isfinite(extent) or extent <= 0.0:
                raise ValueError(f"Invalid mesh extent for {mesh_path}: {extent}")
            mesh_cache[object_name] = {
                "points": [tuple(float(value) for value in point) for point in sampled_points],
                "normals": [tuple(float(value) for value in normal) for normal in sampled_normals],
                "centre": tuple(float(value) for value in centre),
                "extent": extent,
            }
        surface = mesh_cache[object_name]

        if sequence_name not in contact_cache:
            contact_path = data_root / "contact_labels_w_semantics_npy_files" / f"{sequence_name}.npy"
            if not contact_path.exists():
                raise FileNotFoundError(f"Contact labels are missing: {contact_path}")
            contact_cache[sequence_name] = np.load(contact_path)
        all_contacts = contact_cache[sequence_name]

        motion = np.asarray(item["motion"])
        rotations = np.asarray(item["obj_rot_mat"])
        object_com = np.asarray(item["window_obj_com_pos"])
        start_frame = int(item.get("start_t_idx", 0))
        frame_count = min(len(motion), len(rotations), len(object_com))
        for local_frame in range(0, frame_count, args.frame_stride):
            absolute_frame = start_frame + local_frame
            frame_key = (sequence_name, absolute_frame)
            if frame_key in seen_frames:
                counts["duplicate_frames"] += 1
                continue
            seen_frames.add(frame_key)
            counts["frames_seen"] += 1
            if absolute_frame >= len(all_contacts):
                raise IndexError(
                    f"Contact frame {absolute_frame} exceeds {sequence_name} labels "
                    f"with length {len(all_contacts)}"
                )
            contact = np.asarray(all_contacts[absolute_frame]).reshape(-1)
            if len(contact) < 2 or float(contact[0]) <= 0.5 or float(contact[1]) <= 0.5:
                counts["non_bimanual_frames"] += 1
                continue
            joints_world = np.asarray(motion[local_frame, : 24 * 3], dtype=np.float64).reshape(24, 3)
            rotation = np.asarray(rotations[local_frame], dtype=np.float64)
            com = np.asarray(object_com[local_frame], dtype=np.float64)
            palms_object = {
                "left": world_to_object(joints_world[22], rotation, com),
                "right": world_to_object(joints_world[23], rotation, com),
            }
            up_object = _rotation_vector_to_object(np, (0.0, 0.0, 1.0), rotation)

            accepted_this_frame: List[Dict[str, object]] = []
            for source_hand in ("left", "right"):
                target_hand = "right" if source_hand == "left" else "left"
                row = evaluate_contact_pair(
                    sequence_name=sequence_name,
                    absolute_frame=absolute_frame,
                    source_hand=source_hand,
                    source_point=palms_object[source_hand],
                    target_point=palms_object[target_hand],
                    candidates=surface["points"],
                    normals=surface["normals"],
                    centre=surface["centre"],
                    up=up_object,
                    extent=float(surface["extent"]),
                    seed=stable_seed(args.seed, sequence_name, absolute_frame, source_hand),
                    top_ks=args.top_k,
                    hit_radius_ratio=args.hit_radius_ratio,
                    exclusion_ratio=args.exclusion_ratio,
                    normal_weight=args.normal_weight,
                )
                if (
                    float(row["source_surface_distance_norm"]) > args.max_surface_distance_ratio
                    or float(row["target_surface_distance_norm"]) > args.max_surface_distance_ratio
                ):
                    counts["surface_distance_rejections"] += 1
                    accepted_this_frame = []
                    break
                row["object_name"] = object_name
                accepted_this_frame.append(row)
            if len(accepted_this_frame) == 2:
                rows.extend(accepted_this_frame)
                counts["accepted_ordered_samples"] += 2
            if len(rows) >= args.max_samples:
                rows = rows[: args.max_samples]
                counts["accepted_ordered_samples"] = len(rows)
                return rows, counts
    return rows, counts


def add_gate(summary: Dict[str, object], args: argparse.Namespace) -> None:
    methods = summary["methods"]
    if not isinstance(methods, Mapping):
        raise TypeError("summary methods must be a mapping")
    top_k = 5 if 5 in args.top_k else sorted(args.top_k)[-1]
    field = f"sequence_macro_top{top_k}_hit"
    baseline = float(methods["center_antipode"][field])
    chord_scores = {
        method: float(methods[method][field])
        for method in ("section_chord", "section_chord_normal")
    }
    best_method = max(chord_scores, key=chord_scores.get)
    gain = chord_scores[best_method] - baseline
    median_chord_distance = float(summary["geometry"]["median_target_chord_distance_norm"])
    passed = gain >= args.min_top5_gain and median_chord_distance <= args.max_median_chord_distance
    summary["diagnostic_gate"] = {
        "status": "PASS" if passed else "FAIL",
        "meaning": "PASS supports a U3-side section-prior ablation; it does not prove VLM or diffusion gains.",
        "comparison_top_k": top_k,
        "best_section_method": best_method,
        "best_section_hit": chord_scores[best_method],
        "center_antipode_hit": baseline,
        "absolute_hit_gain": gain,
        "required_hit_gain": args.min_top5_gain,
        "median_target_chord_distance_norm": median_chord_distance,
        "maximum_allowed_median_chord_distance_norm": args.max_median_chord_distance,
    }


def write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    args: argparse.Namespace,
    counts: Mapping[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No eligible bimanual contact samples were found")
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        if list(row.keys()) != fieldnames:
            raise ValueError("Result rows do not share a stable CSV schema")
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "experiment": "minimal_contact_conditioned_sectional_prior",
        "mode": "synthetic" if args.synthetic else "omomo",
        "config": {
            key: value
            for key, value in vars(args).items()
            if key not in {"data_root_folder"}
        },
        "data_counts": dict(counts),
        **summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    descriptor_fields = (
        "sequence_name",
        "absolute_frame",
        "source_hand",
        "target_hand",
        "radial_x",
        "radial_y",
        "radial_z",
        "up_x",
        "up_y",
        "up_z",
        "source_x",
        "source_y",
        "source_z",
        "target_x",
        "target_y",
        "target_z",
    )
    with (output_dir / "section_frames.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({field: row[field] for field in descriptor_fields}) + "\n")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.synthetic:
        rows, counts = build_synthetic_rows(args)
    else:
        rows, counts = build_real_rows(args)
    if not rows:
        raise ValueError(
            "No eligible samples remained. Check the split, contact labels, frame stride, "
            "and --max_surface_distance_ratio."
        )
    summary = summarize_results(rows, top_ks=args.top_k)
    add_gate(summary, args)
    output_dir = Path(args.output_dir)
    write_outputs(output_dir, rows, summary, args, counts)
    print(json.dumps({**summary, "output_dir": str(output_dir)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
