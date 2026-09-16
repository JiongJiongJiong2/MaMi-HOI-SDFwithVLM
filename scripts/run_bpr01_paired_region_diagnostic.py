"""Run BPR-01: independent-target paired-contact region diagnostic on CPU.

Synthetic smoke:

    python scripts/run_bpr01_paired_region_diagnostic.py --synthetic \
      --output_dir /tmp/bpr01_synthetic

Real OMOMO validation:

    python scripts/run_bpr01_paired_region_diagnostic.py \
      --data_root_folder /path/to/processed_data \
      --split_manifest /path/to/split_seed1.json \
      --eval_split validation \
      --output_dir /path/to/bpr01_validation
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from manip.geometry.paired_region import (  # noqa: E402
    METHODS,
    bootstrap_difference,
    closest_triangle_points,
    evaluate_projected_contact,
    macro_mean,
    sample_surface_candidates,
    summarize_variants,
)
from manip.geometry.sectional_prior import (  # noqa: E402
    build_contact_section_frame,
    normalize,
    world_to_object,
)


NOISE_DIRECTIONS = ("radial", "lateral", "surface_normal")


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = ":".join(
        (str(base_seed), *(str(part) for part in parts))
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--data_root_folder", type=Path)
    parser.add_argument("--split_manifest", type=Path)
    parser.add_argument(
        "--eval_split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument("--window", type=int, default=120)
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="0 keeps the complete eligible cohort.",
    )
    parser.add_argument("--candidate_count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--top_k", type=int, nargs="+", default=(1, 5, 10))
    parser.add_argument("--hit_radius_ratio", type=float, default=0.08)
    parser.add_argument("--exclusion_ratio", type=float, default=0.08)
    parser.add_argument("--max_surface_distance_ratio", type=float, default=0.12)
    parser.add_argument("--normal_weight", type=float, default=0.25)
    parser.add_argument(
        "--noise_mm",
        type=float,
        nargs="+",
        default=(0.0,),
        help="Pre-registered source displacement magnitudes in millimetres.",
    )
    parser.add_argument(
        "--noise_directions",
        choices=NOISE_DIRECTIONS,
        nargs="+",
        default=("surface_normal",),
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--min_top5_gain", type=float, default=0.05)
    parser.add_argument("--max_median_chord_distance", type=float, default=0.12)
    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.synthetic:
        if args.data_root_folder is None or args.split_manifest is None:
            raise ValueError(
                "Real mode requires --data_root_folder and --split_manifest"
            )
    if args.window < 30:
        raise ValueError("--window must be at least 30")
    if args.frame_stride <= 0:
        raise ValueError("--frame_stride must be positive")
    if args.max_frames < 0:
        raise ValueError("--max_frames must be non-negative")
    if args.candidate_count < 32:
        raise ValueError("--candidate_count must be at least 32")
    if not args.top_k or any(k <= 0 for k in args.top_k):
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
    if not args.noise_mm or any(value < 0.0 for value in args.noise_mm):
        raise ValueError("--noise_mm must contain non-negative values")
    if not args.noise_directions:
        raise ValueError("--noise_directions cannot be empty")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap_samples must be positive")


def _require_real_dependencies():
    try:
        import joblib  # type: ignore
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Real mode requires the project mami_hoi environment with joblib"
        ) from error
    return joblib


def _read_canonical_ply(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read the archive ASCII PLY without loader reordering or cleanup."""
    with path.open("r", encoding="ascii") as handle:
        header = []
        for line in handle:
            header.append(line.strip())
            if line.strip() == "end_header":
                break
        if "format ascii 1.0" not in header or header[-1:] != ["end_header"]:
            raise ValueError(f"Expected archive ASCII PLY: {path}")
        nv = int(next(x for x in header if x.startswith("element vertex ")).split()[-1])
        nf = int(next(x for x in header if x.startswith("element face ")).split()[-1])
        first_property = header.index(f"element vertex {nv}") + 1
        if header[first_property : first_property + 3] != [
            "property float x",
            "property float y",
            "property float z",
        ]:
            raise ValueError("Unsupported PLY vertex schema")
        vertices = np.asarray(
            [
                [float(value) for value in handle.readline().split()[:3]]
                for _ in range(nv)
            ],
            dtype=np.float32,
        ).astype(np.float64)
        faces = []
        for _ in range(nf):
            row = [int(value) for value in handle.readline().split()]
            if len(row) != 4 or row[0] != 3:
                raise ValueError("Only triangle faces are supported")
            faces.append(row[1:])
    faces = np.asarray(faces, dtype=np.int64)
    if (
        vertices.shape != (nv, 3)
        or not np.isfinite(vertices).all()
        or not nv
        or not nf
        or faces.min() < 0
        or faces.max() >= nv
    ):
        raise ValueError(f"Invalid mesh: {path}")
    return vertices, faces


def _object_name(sequence_name: str) -> str:
    parts = sequence_name.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot derive object name from {sequence_name!r}")
    return parts[1]


def _iter_real_windows(
    data: Mapping[object, Mapping[str, object]],
) -> Iterable[Mapping[str, object]]:
    return iter(
        sorted(
            data.values(),
            key=lambda item: (
                str(item["seq_name"]),
                int(item.get("start_t_idx", 0)),
            ),
        )
    )


def collect_real_directed_samples(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    joblib = _require_real_dependencies()
    data_root = args.data_root_folder
    manifest_path = args.split_manifest
    assert data_root is not None and manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_key = f"{args.eval_split}_sequences"
    if split_key not in manifest:
        raise KeyError(f"{manifest_path} is missing {split_key}")
    allowed_sequences = {str(name) for name in manifest[split_key]}
    if not allowed_sequences:
        raise ValueError(f"{split_key} is empty")

    processed_path = (
        data_root
        / f"cano_test_diffusion_manip_window_{args.window}_joints24.p"
    )
    if not processed_path.is_file():
        raise FileNotFoundError(processed_path)
    windows = joblib.load(processed_path)
    if not isinstance(windows, Mapping):
        raise TypeError("Processed window file must contain a mapping")

    contact_cache: Dict[str, np.ndarray] = {}
    seen_frames = set()
    directed: List[Dict[str, object]] = []
    counts = {
        "windows_seen": 0,
        "frames_seen": 0,
        "duplicate_frames": 0,
        "non_bimanual_frames": 0,
        "accepted_frames": 0,
        "accepted_directed_samples": 0,
    }

    for item in _iter_real_windows(windows):
        sequence_name = str(item["seq_name"])
        if sequence_name not in allowed_sequences:
            continue
        counts["windows_seen"] += 1
        object_name = _object_name(sequence_name)
        if sequence_name not in contact_cache:
            contact_path = (
                data_root
                / "contact_labels_w_semantics_npy_files"
                / f"{sequence_name}.npy"
            )
            if not contact_path.is_file():
                raise FileNotFoundError(contact_path)
            contact_cache[sequence_name] = np.load(contact_path)
        contacts = contact_cache[sequence_name]
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
            if absolute_frame >= len(contacts):
                raise IndexError(
                    f"{sequence_name}: frame {absolute_frame} outside contact labels"
                )
            contact = np.asarray(contacts[absolute_frame]).reshape(-1)
            if (
                len(contact) < 2
                or float(contact[0]) <= 0.5
                or float(contact[1]) <= 0.5
            ):
                counts["non_bimanual_frames"] += 1
                continue

            joints_world = np.asarray(
                motion[local_frame, : 24 * 3],
                dtype=np.float64,
            ).reshape(24, 3)
            rotation = np.asarray(rotations[local_frame], dtype=np.float64)
            com = np.asarray(object_com[local_frame], dtype=np.float64)
            palms = {
                "left": world_to_object(joints_world[22], rotation, com),
                "right": world_to_object(joints_world[23], rotation, com),
            }
            up = tuple(
                float(value)
                for value in np.asarray((0.0, 0.0, 1.0)) @ rotation
            )
            directed.append(
                {
                    "sequence_name": sequence_name,
                    "absolute_frame": absolute_frame,
                    "object_name": object_name,
                    "palms": palms,
                    "up": up,
                }
            )
            counts["accepted_frames"] += 1
            counts["accepted_directed_samples"] += 2
            if args.max_frames and counts["accepted_frames"] >= args.max_frames:
                return directed, counts
    return directed, counts


def _build_cube_mesh() -> Tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [
            (-1.0, -1.0, -1.0),
            (1.0, -1.0, -1.0),
            (1.0, 1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, 1.0),
            (-1.0, 1.0, 1.0),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 0, 4),
            (3, 4, 7),
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _collect_synthetic(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, int]]:
    vertices, faces = _build_cube_mesh()
    points, normals = sample_surface_candidates(
        vertices,
        faces,
        args.candidate_count,
        stable_seed(args.seed, "box"),
    )
    directed = [
        {
            "sequence_name": "synthetic_box",
            "absolute_frame": 0,
            "object_name": "box",
            "palms": {
                "left": (1.0, 0.25, 0.25),
                "right": (-1.0, -0.25, 0.25),
            },
            "up": (0.0, 0.0, 1.0),
        }
    ]
    surface = {
        "vertices": vertices,
        "faces": faces,
        "triangles": vertices[faces],
        "points": points,
        "normals": normals,
        "centre": (0.0, 0.0, 0.0),
        "extent": 2.0,
    }
    return directed, surface, {
        "windows_seen": 1,
        "frames_seen": 1,
        "duplicate_frames": 0,
        "non_bimanual_frames": 0,
        "accepted_frames": 1,
        "accepted_directed_samples": 2,
        "surface_distance_rejections": 0,
    }


def _surface_for_object(
    args: argparse.Namespace,
    object_name: str,
) -> Dict[str, object]:
    assert args.data_root_folder is not None
    mesh_path = (
        args.data_root_folder
        / "rest_object_geo"
        / f"{object_name}.ply"
    )
    vertices, faces = _read_canonical_ply(mesh_path)
    points, normals = sample_surface_candidates(
        vertices,
        faces,
        args.candidate_count,
        stable_seed(args.seed, object_name),
    )
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    centre = 0.5 * (bounds[0] + bounds[1])
    extent = float(np.max(bounds[1] - bounds[0]))
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError(f"Invalid mesh extent: {mesh_path}")
    return {
        "vertices": vertices,
        "faces": faces,
        "triangles": vertices[faces],
        "points": points,
        "normals": normals,
        "centre": centre,
        "extent": extent,
    }


def _project_directed_samples(
    args: argparse.Namespace,
    directed: Sequence[Mapping[str, object]],
    surfaces: Mapping[str, Mapping[str, object]],
) -> Tuple[List[Dict[str, object]], int]:
    projected: List[Dict[str, object]] = []
    rejected_frames = 0
    by_object: Dict[str, List[Mapping[str, object]]] = {}
    for item in directed:
        by_object.setdefault(str(item["object_name"]), []).append(item)

    for object_name, items in sorted(by_object.items()):
        surface = surfaces[object_name]
        triangles = np.asarray(surface["triangles"], dtype=np.float64)
        extent = float(surface["extent"])
        queries = np.asarray(
            [
                item["palms"][hand]
                for item in items
                for hand in ("left", "right")
            ],
            dtype=np.float64,
        )
        distances, closest, _, normals = closest_triangle_points(
            queries,
            triangles,
            query_batch=32,
            triangle_chunk=2048,
        )
        projection_error_norm = distances / extent
        candidate_points = np.asarray(surface["points"], dtype=np.float64)
        raw_candidate_distance_norm = np.asarray(
            [
                np.min(np.linalg.norm(candidate_points - query, axis=1)) / extent
                for query in queries
            ],
            dtype=np.float64,
        )
        for item_index, item in enumerate(items):
            left_index = 2 * item_index
            right_index = left_index + 1
            if (
                max(
                    raw_candidate_distance_norm[left_index],
                    raw_candidate_distance_norm[right_index],
                )
                > args.max_surface_distance_ratio
            ):
                rejected_frames += 1
                continue
            projected.append(
                {
                    **item,
                    "source": {
                        "left": closest[left_index],
                        "right": closest[right_index],
                    },
                    "source_normal": {
                        "left": normals[left_index],
                        "right": normals[right_index],
                    },
                    "target": {
                        "left": closest[right_index],
                        "right": closest[left_index],
                    },
                    "target_normal": {
                        "left": normals[right_index],
                        "right": normals[left_index],
                    },
                    "projection_error_norm": {
                        "left": projection_error_norm[left_index],
                        "right": projection_error_norm[right_index],
                    },
                    "cohort_candidate_distance_norm": {
                        "left": raw_candidate_distance_norm[left_index],
                        "right": raw_candidate_distance_norm[right_index],
                    },
                }
            )
    return projected, rejected_frames


def _variant_name(noise_mm: float, direction: str) -> str:
    if noise_mm == 0.0:
        return "clean"
    return f"{direction}_{noise_mm:g}mm"


def _source_direction(
    frame: object,
    source_normal: np.ndarray,
    direction: str,
) -> np.ndarray:
    if direction == "radial":
        return np.asarray(frame.radial, dtype=np.float64)
    if direction == "lateral":
        return np.asarray(frame.vertical_normal, dtype=np.float64)
    if direction == "surface_normal":
        return np.asarray(
            normalize(tuple(float(value) for value in source_normal)),
            dtype=np.float64,
        )
    raise ValueError(f"Unknown noise direction: {direction}")


def evaluate_projected_samples(
    args: argparse.Namespace,
    projected: Sequence[Mapping[str, object]],
    surfaces: Mapping[str, Mapping[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    noise_specs = [(0.0, "clean")]
    noise_specs.extend(
        (float(value), direction)
        for value in args.noise_mm
        if value > 0.0
        for direction in args.noise_directions
    )

    for noise_mm, requested_direction in noise_specs:
        variant = _variant_name(noise_mm, requested_direction)
        source_queries: List[np.ndarray] = []
        sample_meta: List[Dict[str, object]] = []

        for item in projected:
            surface = surfaces[str(item["object_name"])]
            centre = np.asarray(surface["centre"], dtype=np.float64)
            up = np.asarray(item["up"], dtype=np.float64)
            extent = float(surface["extent"])
            for source_hand in ("left", "right"):
                source = np.asarray(item["source"][source_hand], dtype=np.float64)
                source_normal = np.asarray(
                    item["source_normal"][source_hand],
                    dtype=np.float64,
                )
                if noise_mm == 0.0:
                    query = source
                else:
                    frame = build_contact_section_frame(
                        source,
                        centre,
                        up,
                        extent,
                    )
                    direction = _source_direction(
                        frame,
                        source_normal,
                        requested_direction,
                    )
                    query = source + direction * (noise_mm / 1000.0)
                source_queries.append(query)
                sample_meta.append(
                    {
                        "item": item,
                        "source_hand": source_hand,
                    }
                )

        projected_sources: List[
            Tuple[np.ndarray, np.ndarray, float] | None
        ] = [None] * len(sample_meta)
        for object_name in sorted(
            {str(item["object_name"]) for item in projected}
        ):
            indices = [
                index
                for index, meta in enumerate(sample_meta)
                if str(meta["item"]["object_name"]) == object_name
            ]
            surface = surfaces[object_name]
            distances, closest, _, normals = closest_triangle_points(
                np.asarray([source_queries[index] for index in indices]),
                np.asarray(surface["triangles"], dtype=np.float64),
                query_batch=32,
                triangle_chunk=2048,
            )
            extent = float(surface["extent"])
            for local_index, global_index in enumerate(indices):
                projected_sources[global_index] = (
                    closest[local_index],
                    normals[local_index],
                    float(distances[local_index] / extent),
                )
        if any(value is None for value in projected_sources):
            raise RuntimeError("Missing a projected source after batching")

        for index, meta in enumerate(sample_meta):
            item = meta["item"]
            source_hand = str(meta["source_hand"])
            surface = surfaces[str(item["object_name"])]
            projected_source, source_normal, source_error = projected_sources[index]
            row = evaluate_projected_contact(
                sequence_name=str(item["sequence_name"]),
                absolute_frame=int(item["absolute_frame"]),
                source_hand=source_hand,
                source_point=projected_source,
                source_normal=source_normal,
                target_point=np.asarray(item["target"][source_hand], dtype=np.float64),
                target_normal=np.asarray(
                    item["target_normal"][source_hand],
                    dtype=np.float64,
                ),
                candidates=np.asarray(surface["points"], dtype=np.float64),
                normals=np.asarray(surface["normals"], dtype=np.float64),
                centre=np.asarray(surface["centre"], dtype=np.float64),
                up=np.asarray(item["up"], dtype=np.float64),
                extent=float(surface["extent"]),
                seed=stable_seed(
                    args.seed,
                    item["sequence_name"],
                    item["absolute_frame"],
                    source_hand,
                    variant,
                ),
                top_ks=args.top_k,
                hit_radius_ratio=args.hit_radius_ratio,
                exclusion_ratio=args.exclusion_ratio,
                normal_weight=args.normal_weight,
                source_projection_error_norm=source_error,
                target_projection_error_norm=float(
                    item["projection_error_norm"][source_hand]
                ),
                variant=variant,
                noise_mm=noise_mm,
                noise_direction=requested_direction,
            )
            row["object_name"] = str(item["object_name"])
            rows.append(row)
    return rows


def add_gate(
    summary: Dict[str, object],
    rows: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
) -> None:
    clean = summary["variants"]["clean"]
    clean_rows = [row for row in rows if row["variant"] == "clean"]
    primary = "section_chord_normal"
    baseline = "center_antipode"
    top_k = 5 if 5 in args.top_k else sorted(args.top_k)[-1]
    primary_hit = float(clean["methods"][primary][f"top{top_k}_hit"])
    baseline_hit = float(clean["methods"][baseline][f"top{top_k}_hit"])
    gain, ci_low, ci_high = bootstrap_difference(
        clean_rows,
        method=primary,
        baseline=baseline,
        top_k=top_k,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    median_chord = float(
        np.median(
            [float(row["target_chord_distance_norm"]) for row in clean_rows]
        )
    )
    pool_recall = float(clean["pool_recall"])
    eligible_recall = float(clean["eligible_pool_recall"])
    passed = bool(
        gain >= args.min_top5_gain
        and ci_low > 0.0
        and median_chord <= args.max_median_chord_distance
    )
    summary["diagnostic_gate"] = {
        "status": "PASS" if passed else "FAIL",
        "scope": "BPR-01 clean oracle validation development screening",
        "primary_method": primary,
        "comparison_top_k": top_k,
        "primary_hit": primary_hit,
        "baseline_hit": baseline_hit,
        "absolute_gain": gain,
        "paired_sequence_95": [ci_low, ci_high],
        "required_hit_gain": args.min_top5_gain,
        "median_target_chord_distance_norm": median_chord,
        "maximum_allowed_median_chord_distance_norm": (
            args.max_median_chord_distance
        ),
        "pool_recall": pool_recall,
        "eligible_pool_recall": eligible_recall,
        "meaning": (
            "PASS supports proceeding to source-error robustness; it does not "
            "prove a deployable section predictor or model improvement."
        ),
    }
    summary["noise_sensitivity"] = {}
    for variant in sorted(summary["variants"]):
        if variant == "clean":
            continue
        variant_rows = [row for row in rows if row["variant"] == variant]
        variant_gain, variant_low, variant_high = bootstrap_difference(
            variant_rows,
            method=primary,
            baseline=baseline,
            top_k=top_k,
            samples=args.bootstrap_samples,
            seed=stable_seed(args.seed, variant),
        )
        summary["noise_sensitivity"][variant] = {
            "absolute_gain": variant_gain,
            "paired_sequence_95": [variant_low, variant_high],
            "positive_interval": bool(variant_low > 0.0),
        }


def write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    args: argparse.Namespace,
    counts: Mapping[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0].keys())
    for row in rows[1:]:
        if list(row.keys()) != fields:
            raise ValueError("BPR rows do not share a stable CSV schema")
    with (output_dir / "samples.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "experiment": "BPR-01",
        "mode": "synthetic" if args.synthetic else "omomo",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"data_root_folder", "split_manifest"}
        },
        "data_counts": dict(counts),
        **summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "cohort": "G0-style bimanual GT contact cohort",
                "target_definition": (
                    "GT target palm proxy projected independently to the "
                    "original triangle surface"
                ),
                "source_definition": (
                    "GT source palm proxy projected independently, optionally "
                    "displaced by a pre-registered source noise"
                ),
                "candidate_pool": args.candidate_count,
                "selection": (
                    "same greedy spatial separation for all four methods"
                ),
                "noise_mm": list(args.noise_mm),
                "noise_directions": list(args.noise_directions),
                "primary_method": "section_chord_normal",
                "baseline": "center_antipode",
                "gate": (
                    "clean top5 gain >= threshold and paired sequence lower "
                    "bound > 0 and median chord distance <= threshold"
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.synthetic:
        directed, surface, counts = _collect_synthetic(args)
        surfaces = {"box": surface}
    else:
        directed, counts = collect_real_directed_samples(args)
        surfaces = {
            object_name: _surface_for_object(args, object_name)
            for object_name in sorted(
                {str(item["object_name"]) for item in directed}
            )
        }
    if not directed:
        raise ValueError("No bimanual contact frames were collected")
    projected, rejected_frames = _project_directed_samples(
        args,
        directed,
        surfaces,
    )
    counts = {**counts, "surface_distance_rejections": rejected_frames}
    if not projected:
        raise ValueError("No frames remained after source/target projection")
    rows = evaluate_projected_samples(args, projected, surfaces)
    summary = {
        "variants": summarize_variants(rows, top_ks=args.top_k),
        "sample_count": len(rows),
        "frame_count": len(projected),
        "sequence_count": len({str(row["sequence_name"]) for row in rows}),
    }
    add_gate(summary, rows, args)
    write_outputs(args.output_dir, rows, summary, args, counts)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "sample_count": summary["sample_count"],
                "frame_count": summary["frame_count"],
                "sequence_count": summary["sequence_count"],
                "diagnostic_gate": summary["diagnostic_gate"],
                "noise_sensitivity": summary["noise_sensitivity"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
