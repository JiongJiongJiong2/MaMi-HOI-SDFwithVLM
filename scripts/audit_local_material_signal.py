"""CPU diagnostic for local material-side signals near interaction queries.

This is a standalone read-only geometry audit for plasticbox, trashcan, and
smalltable. It compares the legacy 256^3 SDF sign, closest-surface oriented
normals/pseudonormals, and generalized winding number on the current canonical
triangle mesh. Saved model prediction queries are explicitly reported as
missing when no cache is supplied.

No trainer, dataset constructor, mesh repair output, CUDA call, training
change, or U2/U3/U4 implementation is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from manip.model.sdf_utils import sample_object_sdf_at_points
from scripts.audit_bps_surface_queries import (
    memory_snapshot,
    read_canonical_ply,
    sha256,
)
from scripts.material_oracle_probe import (
    conservative_candidate,
    run_fixtures,
    winding,
)


torch.set_num_threads(1)

OBJECTS = ("plasticbox", "trashcan", "smalltable")
BANDS = (
    ("0_5_mm", 0.0, 0.005),
    ("5_10_mm", 0.005, 0.010),
    ("10_20_mm", 0.010, 0.020),
    ("20_50_mm", 0.020, 0.050),
    ("gt_50_mm", 0.050, float("inf")),
)
SIGNALS = ("legacy", "raw_normal", "pseudonormal", "winding")


def stable_seed(seed: int, *parts: object) -> int:
    payload = ":".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def nearest_triangles_with_face(
    queries: np.ndarray,
    triangles: np.ndarray,
    query_batch: int = 16,
    triangle_chunk: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact point-to-triangle distance with the winning global face index."""
    queries = np.asarray(queries, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    distances = np.empty(len(queries))
    closest = np.empty_like(queries)
    face_indices = np.empty(len(queries), dtype=np.int64)

    for start in range(0, len(queries), query_batch):
        q = queries[start : start + query_batch]
        best = np.full(len(q), np.inf)
        result = np.empty_like(q)
        result_ids = np.empty(len(q), dtype=np.int64)

        for offset in range(0, len(triangles), triangle_chunk):
            tri = triangles[offset : offset + triangle_chunk]
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            normal = np.cross(b - a, c - a)
            nn = np.einsum("ki,ki->k", normal, normal)
            safe_nn = np.where(nn > 0, nn, 1.0)
            signed_height = np.einsum(
                "qki,ki->qk", q[:, None] - a, normal
            ) / safe_nn
            projection = q[:, None] - signed_height[..., None] * normal

            inside = np.broadcast_to(nn > 0, signed_height.shape).copy()
            for u, v in ((a, b), (b, c), (c, a)):
                side = np.einsum(
                    "qki,ki->qk", np.cross(v - u, projection - u), normal
                )
                inside &= side >= -1e-12 * nn

            local_best = np.where(inside, signed_height**2 * nn, np.inf)
            local_cp = projection.copy()
            for u, v in ((a, b), (b, c), (c, a)):
                edge = v - u
                ee = np.einsum("ki,ki->k", edge, edge)
                t = np.einsum(
                    "qki,ki->qk", q[:, None] - u, edge
                ) / np.where(ee > 0, ee, 1.0)
                cp = u + np.clip(t, 0.0, 1.0)[..., None] * edge
                delta = q[:, None] - cp
                d2 = np.einsum("qki,qki->qk", delta, delta)
                improve = d2 < local_best
                local_best[improve], local_cp[improve] = d2[improve], cp[improve]

            k = local_best.argmin(axis=1)
            value = local_best[np.arange(len(q)), k]
            improve = value < best
            best[improve] = value[improve]
            result[improve] = local_cp[np.arange(len(q)), k][improve]
            result_ids[improve] = offset + k[improve]

        distances[start : start + len(q)] = np.sqrt(best)
        closest[start : start + len(q)] = result
        face_indices[start : start + len(q)] = result_ids

    return distances, closest, face_indices


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    unit = np.zeros_like(normals)
    np.divide(
        normals,
        lengths[:, None],
        out=unit,
        where=lengths[:, None] > 0,
    )
    return unit, lengths


def barycentric_weights(point: np.ndarray, triangle: np.ndarray) -> np.ndarray | None:
    a, b, c = np.asarray(triangle, dtype=np.float64)
    full_normal = np.cross(b - a, c - a)
    full_area = 0.5 * np.linalg.norm(full_normal)
    if full_area <= 0:
        return None
    w0 = 0.5 * np.linalg.norm(np.cross(b - point, c - point))
    w1 = 0.5 * np.linalg.norm(np.cross(a - point, c - point))
    w2 = 0.5 * np.linalg.norm(np.cross(a - point, b - point))
    return np.array([w0, w1, w2], dtype=np.float64) / full_area


def position_kind(
    weights: np.ndarray,
    eps: float = 1e-7,
) -> Tuple[str, int | Tuple[int, int] | None]:
    vertex = int(np.argmax(weights))
    if weights[vertex] >= 1.0 - eps:
        return "vertex", vertex
    small = [i for i in range(3) if weights[i] <= eps]
    if len(small) == 1:
        edge = tuple(sorted(i for i in range(3) if i != small[0]))
        return "edge", edge
    return "face", None


def build_pseudonormal_tables(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    lengths: np.ndarray,
) -> Dict[str, Any]:
    vertex_faces: Dict[int, List[int]] = defaultdict(list)
    edge_faces: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    areas = 0.5 * lengths
    for face_id, (a, b, c) in enumerate(np.asarray(faces, dtype=np.int64)):
        for vertex in (int(a), int(b), int(c)):
            vertex_faces[vertex].append(face_id)
        for edge in (
            (int(a), int(b)),
            (int(b), int(c)),
            (int(c), int(a)),
        ):
            edge_faces[tuple(sorted(edge))].append(face_id)
    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int64),
        "normals": normals,
        "areas": areas,
        "vertex_faces": vertex_faces,
        "edge_faces": edge_faces,
    }


def pseudonormal_at_point(
    point: np.ndarray,
    face_id: int,
    triangle: np.ndarray,
    tables: Dict[str, Any],
    eps: float = 1e-10,
) -> Tuple[np.ndarray, bool]:
    weights = barycentric_weights(point, triangle)
    if weights is None:
        return np.zeros(3, dtype=np.float64), False
    kind, detail = position_kind(weights)
    if kind == "face":
        candidate = tables["normals"][face_id]
    elif kind == "vertex":
        face_ids = tables["vertex_faces"].get(int(detail))
        if not face_ids:
            return np.zeros(3, dtype=np.float64), False
        ids = np.asarray(face_ids, dtype=np.int64)
        candidate = np.sum(
            tables["areas"][ids, None] * tables["normals"][ids],
            axis=0,
        )
    else:
        face_ids = tables["edge_faces"].get(tuple(detail))
        if not face_ids:
            return np.zeros(3, dtype=np.float64), False
        ids = np.asarray(face_ids, dtype=np.int64)
        candidate = np.sum(
            tables["areas"][ids, None] * tables["normals"][ids],
            axis=0,
        )
    norm = np.linalg.norm(candidate)
    if norm <= eps:
        return np.zeros(3, dtype=np.float64), False
    return candidate / norm, True


def band_for_distance(distance: float) -> str:
    for name, lo, hi in BANDS:
        if distance >= lo and distance < hi:
            return name
    return "invalid"


def quantiles(values: Iterable[float]) -> Dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def load_legacy_sdf(data_root: Path, object_name: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sdf_path = data_root / "rest_object_sdf_256_npy_files" / f"{object_name}.ply.npy"
    metadata_path = data_root / "rest_object_sdf_256_npy_files" / f"{object_name}.ply.json"
    if not sdf_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing legacy 256^3 SDF pair for {object_name}: {sdf_path}, {metadata_path}"
        )
    grid = np.load(sdf_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if grid.shape != (256, 256, 256) or not np.isfinite(grid).all():
        raise ValueError(f"Invalid legacy SDF grid: {sdf_path}")
    tensor = torch.from_numpy(np.ascontiguousarray(grid)).float()[None, None]
    centroid = torch.tensor(metadata["centroid"], dtype=torch.float32)[None]
    extents = torch.tensor(metadata["extents"], dtype=torch.float32)[None]
    if (extents <= 0).any():
        raise ValueError(f"Non-positive SDF extents in {metadata_path}")
    return tensor, centroid, extents


def sample_legacy_batch(
    points: np.ndarray,
    grid: torch.Tensor,
    centroid: torch.Tensor,
    extents: torch.Tensor,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    values: List[np.ndarray] = []
    valid: List[np.ndarray] = []
    points = np.asarray(points, dtype=np.float32)
    for start in range(0, len(points), batch_size):
        block = torch.tensor(points[start : start + batch_size])[None]
        with torch.no_grad():
            sampled, mask = sample_object_sdf_at_points(
                grid,
                block,
                centroid,
                extents,
                return_valid_mask=True,
            )
        values.append(sampled.detach().cpu().numpy().reshape(-1))
        valid.append(mask.detach().cpu().numpy().reshape(-1))
    return np.concatenate(values), np.concatenate(valid)


def legacy_gradient(
    points: np.ndarray,
    grid: torch.Tensor,
    centroid: torch.Tensor,
    extents: torch.Tensor,
    step: float,
) -> Tuple[np.ndarray, np.ndarray]:
    directions = np.eye(3, dtype=np.float64)
    plus: List[np.ndarray] = []
    minus: List[np.ndarray] = []
    plus_valid: List[np.ndarray] = []
    minus_valid: List[np.ndarray] = []
    for axis in range(3):
        offset = np.zeros((len(points), 3), dtype=np.float64)
        offset[:, axis] = step
        pv, pvm = sample_legacy_batch(points + offset, grid, centroid, extents)
        mv, mvm = sample_legacy_batch(points - offset, grid, centroid, extents)
        plus.append(pv)
        minus.append(mv)
        plus_valid.append(pvm)
        minus_valid.append(mvm)
    plus_all = np.stack(plus, axis=1)
    minus_all = np.stack(minus, axis=1)
    gradient = (plus_all - minus_all) / (2.0 * step)
    valid = np.stack(plus_valid + minus_valid, axis=1).all(axis=1)
    return gradient, valid


def evaluate_geometry(
    points: np.ndarray,
    raw_triangles: np.ndarray,
    raw_normals: np.ndarray,
    raw_lengths: np.ndarray,
    candidate_triangles: np.ndarray,
    pseudonormal_tables: Dict[str, Any],
    legacy_grid: torch.Tensor,
    legacy_centroid: torch.Tensor,
    legacy_extents: torch.Tensor,
    query_batch: int = 16,
    triangle_chunk: int = 1024,
    compute_gradient: bool = False,
) -> Dict[str, Any]:
    raw_distance, raw_closest, raw_face = nearest_triangles_with_face(
        points,
        raw_triangles,
        query_batch,
        triangle_chunk,
    )
    candidate_distance, candidate_closest, candidate_face = nearest_triangles_with_face(
        points,
        candidate_triangles,
        query_batch,
        triangle_chunk,
    )
    candidate_triangles = np.asarray(candidate_triangles, dtype=np.float64)
    candidate_normals = pseudonormal_tables["normals"]
    candidate_areas = pseudonormal_tables["areas"]
    candidate_lengths = np.linalg.norm(candidate_normals, axis=1)
    surface_eps = 1e-9 * float(np.ptp(pseudonormal_tables["vertices"], axis=0).max())

    pseudonormals = np.zeros_like(points)
    pseudonormal_available = np.zeros(len(points), dtype=bool)
    for i, (point, face_id) in enumerate(zip(candidate_closest, candidate_face)):
        pn, ok = pseudonormal_at_point(
            point,
            int(face_id),
            candidate_triangles[int(face_id)],
            pseudonormal_tables,
        )
        pseudonormals[i] = pn
        pseudonormal_available[i] = ok

    raw_normal_unit = np.zeros_like(points)
    raw_normal_available = np.zeros(len(points), dtype=bool)
    for i, face_id in enumerate(raw_face):
        face_id = int(face_id)
        if raw_lengths[face_id] > 0 and raw_distance[i] > surface_eps:
            raw_normal_unit[i] = raw_normals[face_id]
            raw_normal_available[i] = True

    legacy_values, legacy_valid = sample_legacy_batch(
        points,
        legacy_grid,
        legacy_centroid,
        legacy_extents,
    )

    winding_values = winding(points, candidate_triangles, query_batch=8, triangle_chunk=triangle_chunk)

    result: Dict[str, Any] = {
        "raw_distance": raw_distance,
        "raw_closest": raw_closest,
        "raw_face": raw_face,
        "candidate_distance": candidate_distance,
        "candidate_closest": candidate_closest,
        "candidate_face": candidate_face,
        "raw_normal": raw_normal_unit,
        "raw_normal_available": raw_normal_available,
        "pseudonormal": pseudonormals,
        "pseudonormal_available": pseudonormal_available,
        "legacy_values": legacy_values,
        "legacy_valid": legacy_valid,
        "winding": winding_values,
    }
    if compute_gradient:
        max_extent = float(legacy_extents.max().item())
        step = max_extent / 255.0
        gradient, gradient_valid = legacy_gradient(
            points,
            legacy_grid,
            legacy_centroid,
            legacy_extents,
            step,
        )
        result["legacy_gradient"] = gradient
        result["legacy_gradient_valid"] = gradient_valid
    return result


def signal_codes_for_rows(
    rows: Sequence[Dict[str, Any]],
    geometry: Dict[str, Any],
    boundary_tol: float,
) -> None:
    raw_distance = geometry["raw_distance"]
    raw_closest = geometry["raw_closest"]
    candidate_distance = geometry["candidate_distance"]
    candidate_closest = geometry["candidate_closest"]
    raw_normal = geometry["raw_normal"]
    raw_available = geometry["raw_normal_available"]
    pseudonormal = geometry["pseudonormal"]
    pseudonormal_available = geometry["pseudonormal_available"]
    legacy_values = geometry["legacy_values"]
    legacy_valid = geometry["legacy_valid"]
    winding_values = geometry["winding"]

    for i, row in enumerate(rows):
        q = np.array([row["x"], row["y"], row["z"]], dtype=np.float64)

        raw_side = float(np.dot(q - raw_closest[i], raw_normal[i]))
        raw_code = None
        raw_unknown = True
        if raw_available[i] and raw_distance[i] > boundary_tol and abs(raw_side) > boundary_tol:
            raw_code = 1 if raw_side < 0 else -1
            raw_unknown = False

        candidate_side = float(np.dot(q - candidate_closest[i], pseudonormal[i]))
        pseudonormal_code = None
        pseudonormal_unknown = True
        if (
            pseudonormal_available[i]
            and candidate_distance[i] > boundary_tol
            and abs(candidate_side) > boundary_tol
        ):
            pseudonormal_code = 1 if candidate_side < 0 else -1
            pseudonormal_unknown = False

        legacy_code = None
        legacy_unknown = True
        if legacy_valid[i] and np.isfinite(legacy_values[i]) and abs(legacy_values[i]) > boundary_tol:
            legacy_code = 1 if legacy_values[i] < 0 else -1
            legacy_unknown = False

        winding_code = None
        winding_unknown = not bool(np.isfinite(winding_values[i]))
        if winding_unknown is False:
            winding_code = 1 if winding_values[i] > 0.5 else -1

        row.update(
            {
                "raw_udf_m": float(raw_distance[i]),
                "band": band_for_distance(float(raw_distance[i])),
                "raw_closest_x": float(raw_closest[i, 0]),
                "raw_closest_y": float(raw_closest[i, 1]),
                "raw_closest_z": float(raw_closest[i, 2]),
                "raw_face_id": int(geometry["raw_face"][i]),
                "raw_face_normal_x": float(raw_normal[i, 0]),
                "raw_face_normal_y": float(raw_normal[i, 1]),
                "raw_face_normal_z": float(raw_normal[i, 2]),
                "raw_normal_side_m": float(raw_side),
                "raw_normal_signal_code": raw_code,
                "raw_normal_unknown": raw_unknown,
                "candidate_udf_m": float(candidate_distance[i]),
                "candidate_closest_x": float(candidate_closest[i, 0]),
                "candidate_closest_y": float(candidate_closest[i, 1]),
                "candidate_closest_z": float(candidate_closest[i, 2]),
                "candidate_face_id": int(geometry["candidate_face"][i]),
                "pseudonormal_x": float(pseudonormal[i, 0]),
                "pseudonormal_y": float(pseudonormal[i, 1]),
                "pseudonormal_z": float(pseudonormal[i, 2]),
                "pseudonormal_available": bool(pseudonormal_available[i]),
                "pseudonormal_side_m": float(candidate_side),
                "pseudonormal_signal_code": pseudonormal_code,
                "pseudonormal_unknown": pseudonormal_unknown,
                "legacy_sdf_m": float(legacy_values[i]),
                "legacy_valid": bool(legacy_valid[i]),
                "legacy_signal_code": legacy_code,
                "legacy_unknown": legacy_unknown,
                "winding": float(winding_values[i]),
                "winding_material_025": bool(winding_values[i] > 0.25),
                "winding_material_050": bool(winding_values[i] > 0.5),
                "winding_material_075": bool(winding_values[i] > 0.75),
                "winding_signal_code": winding_code,
                "winding_unknown": winding_unknown,
            }
        )


def load_gt_hand_rows(csv_path: Path, objects: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("object") not in objects:
                continue
            if row.get("stratum") not in ("gt_palm_contact", "gt_palm_noncontact"):
                continue
            try:
                point = np.array(
                    [float(row["x"]), float(row["y"]), float(row["z"])],
                    dtype=np.float64,
                )
            except (KeyError, ValueError) as error:
                raise ValueError(f"Invalid GT query row: {row}") from error
            rows.append(
                {
                    "object": row["object"],
                    "source_class": "gt_hand",
                    "source_stratum": row["stratum"],
                    "contact_annotation": int(row.get("contact_annotation") or -1),
                    "sequence": row.get("sequence") or "",
                    "frame": int(row.get("frame") or -1),
                    "joint": int(row.get("joint") or -1),
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "label": None,
                    "region": None,
                    "source": "GT joints 22/23 canonical palm proxies from frozen BPS diagnostic; not model predictions",
                }
            )
    return rows


def load_manual_rows(
    manifest_path: Path,
    objects: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for object_name in objects:
        data = manifest["objects"][object_name]
        for probe in data["probes"]:
            point = np.asarray(probe["point"], dtype=np.float64)
            rows.append(
                {
                    "object": object_name,
                    "source_class": "reviewed_manual",
                    "source_stratum": f"manual_{probe['label']}",
                    "contact_annotation": None,
                    "sequence": None,
                    "frame": None,
                    "joint": None,
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2]),
                    "label": probe["label"],
                    "region": probe["region"],
                    "source": probe["source"],
                }
            )
    return rows, manifest


def add_perturbations(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    axis_names = ("x", "y", "z")
    for base in rows:
        base_copy = dict(base)
        base_copy["query_id"] = f"{base['object']}_{len(output)}"
        base_copy["parent_query_id"] = base_copy["query_id"]
        base_copy["is_base"] = True
        base_copy["perturbation_mm"] = None
        base_copy["perturbation_axis"] = None
        base_copy["perturbation_sign"] = None
        output.append(base_copy)

        magnitudes = (0.0005, 0.0010) if base["source_class"] == "gt_hand" else (0.0005,)
        point = np.array([base["x"], base["y"], base["z"]], dtype=np.float64)
        for axis in range(3):
            for sign in (-1, 1):
                for magnitude in magnitudes:
                    child = dict(base)
                    child["query_id"] = f"{base['object']}_{len(output)}"
                    child["parent_query_id"] = base_copy["query_id"]
                    child["is_base"] = False
                    child["perturbation_mm"] = magnitude * 1000.0
                    child["perturbation_axis"] = axis_names[axis]
                    child["perturbation_sign"] = sign
                    offset = np.zeros(3, dtype=np.float64)
                    offset[axis] = sign * magnitude
                    moved = point + offset
                    child["x"], child["y"], child["z"] = float(moved[0]), float(moved[1]), float(moved[2])
                    child["label"] = None
                    child["region"] = None
                    child["source"] = (
                        f"Fixed local perturbation of {base['source_class']} query "
                        f"{base_copy['query_id']}; no occupancy label is inherited"
                    )
                    output.append(child)
    return output


def build_rows(
    objects: Sequence[str],
    queries_csv: Path,
    manifest_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    gt_rows = load_gt_hand_rows(queries_csv, objects)
    manual_rows, manifest = load_manual_rows(manifest_path, objects)
    rows = add_perturbations([*gt_rows, *manual_rows])
    return rows, {
        "gt_palm_queries": len(gt_rows),
        "reviewed_manual_queries": len(manual_rows),
        "total_rows_with_perturbations": len(rows),
    }


def group_by_parent(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["parent_query_id"]].append(row)
    return dict(groups)


def signal_value(row: Dict[str, Any], signal: str) -> int | None:
    key = f"{signal}_signal_code"
    return row.get(key)


def signal_unknown(row: Dict[str, Any], signal: str) -> bool:
    key = f"{signal}_unknown"
    return bool(row.get(key, True))


def signal_material(row: Dict[str, Any], signal: str) -> bool:
    return signal_value(row, signal) == 1


def signal_free(row: Dict[str, Any], signal: str) -> bool:
    return signal_value(row, signal) == -1


def band_summary(
    rows: Sequence[Dict[str, Any]],
    signals: Sequence[str],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for band, lo, hi in BANDS:
        mask = [lo <= row["raw_udf_m"] < hi for row in rows]
        selected = [row for row, keep in zip(rows, mask) if keep]
        if not selected:
            continue
        by_source = {}
        for row in selected:
            key = (row["source_class"], row["source_stratum"])
            by_source[key] = by_source.get(key, 0) + 1
        signal_stats = {}
        for signal in signals:
            unknown = sum(signal_unknown(row, signal) for row in selected)
            material = sum(signal_material(row, signal) for row in selected)
            free = sum(signal_free(row, signal) for row in selected)
            signal_stats[signal] = {
                "count": len(selected),
                "valid_count": len(selected) - unknown,
                "valid_fraction": (len(selected) - unknown) / len(selected),
                "unknown_count": unknown,
                "unknown_fraction": unknown / len(selected),
                "material_fraction": material / len(selected),
                "free_fraction": free / len(selected),
            }
        result.append(
            {
                "band": band,
                "count": len(selected),
                "source_counts": {
                    f"{cls}|{stratum}": count for (cls, stratum), count in sorted(by_source.items())
                },
                "signals": signal_stats,
            }
        )
    return result


def base_band_summary(
    rows: Sequence[Dict[str, Any]],
    signals: Sequence[str],
) -> List[Dict[str, Any]]:
    return band_summary([row for row in rows if row["is_base"]], signals)


def label_metrics(
    rows: Sequence[Dict[str, Any]],
    signals: Sequence[str],
) -> Dict[str, Any]:
    known = [
        row
        for row in rows
        if row["source_class"] == "reviewed_manual"
        and row["is_base"]
        and row["label"] in ("material", "free")
    ]
    result: Dict[str, Any] = {}
    for label in ("material", "free"):
        subset = [row for row in known if row["label"] == label]
        expected = 1 if label == "material" else -1
        result[label] = {"count": len(subset)}
        for signal in signals:
            predicted = [signal_value(row, signal) for row in subset]
            unknown = [signal_unknown(row, signal) for row in subset]
            correct = [
                value == expected and not unk
                for value, unk in zip(predicted, unknown)
            ]
            decisions = [not unk for unk in unknown]
            abstained = len(subset) - sum(decisions)
            result[label][signal] = {
                "correct": int(sum(correct)),
                "known_total": len(subset),
                "accuracy_all_known": float(sum(correct) / len(subset)) if subset else None,
                "abstained": abstained,
                "decisions": int(sum(decisions)),
                "decision_accuracy": (
                    float(sum(correct) / sum(decisions)) if sum(decisions) else None
                ),
                "predicted_material": int(sum(value == 1 for value in predicted)),
                "predicted_free": int(sum(value == -1 for value in predicted)),
            }
    return result


def free_region_metrics(
    rows: Sequence[Dict[str, Any]],
    signals: Sequence[str],
) -> Dict[str, Any]:
    base = [row for row in rows if row["is_base"] and row["source_class"] == "reviewed_manual"]
    result: Dict[str, Any] = {}
    for object_name in OBJECTS:
        object_rows = [row for row in base if row["object"] == object_name]
        for region in sorted({row["region"] for row in object_rows if row["region"]}):
            subset = [row for row in object_rows if row["region"] == region and row["label"] == "free"]
            if not subset:
                continue
            result[f"{object_name}|{region}"] = {
                "free_count": len(subset),
                "signals": {},
            }
            for signal in signals:
                material = sum(signal_material(row, signal) for row in subset)
                unknown = sum(signal_unknown(row, signal) for row in subset)
                result[f"{object_name}|{region}"]["signals"][signal] = {
                    "material_count": material,
                    "material_fraction": material / len(subset),
                    "unknown_count": unknown,
                    "unknown_fraction": unknown / len(subset),
                }
    return result


def stability_metrics(
    rows: Sequence[Dict[str, Any]],
    signals: Sequence[str],
) -> Dict[str, Any]:
    groups = group_by_parent(rows)
    records: List[Dict[str, Any]] = []
    for parent, members in groups.items():
        base = next(row for row in members if row["is_base"])
        children = [row for row in members if not row["is_base"]]
        if not children:
            continue
        record = {
            "object": base["object"],
            "source_class": base["source_class"],
            "source_stratum": base["source_stratum"],
            "band": base["band"],
            "parent": parent,
            "children": len(children),
            "raw_face_changes": 0,
            "candidate_face_changes": 0,
            "normal_angle_deg": [],
            "signals": {signal: {"flips": 0, "comparable": 0} for signal in signals},
        }
        base_unknown = {signal: signal_unknown(base, signal) for signal in signals}
        for child in children:
            if child["raw_face_id"] != base["raw_face_id"]:
                record["raw_face_changes"] += 1
            if child["candidate_face_id"] != base["candidate_face_id"]:
                record["candidate_face_changes"] += 1
            if (
                child["pseudonormal_available"]
                and base["pseudonormal_available"]
            ):
                a = np.array(
                    [
                        base["pseudonormal_x"],
                        base["pseudonormal_y"],
                        base["pseudonormal_z"],
                    ]
                )
                b = np.array(
                    [
                        child["pseudonormal_x"],
                        child["pseudonormal_y"],
                        child["pseudonormal_z"],
                    ]
                )
                cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
                record["normal_angle_deg"].append(float(np.degrees(np.arccos(cosine))))
            for signal in signals:
                if not base_unknown[signal] and not signal_unknown(child, signal):
                    record["signals"][signal]["comparable"] += 1
                    if signal_value(child, signal) != signal_value(base, signal):
                        record["signals"][signal]["flips"] += 1
        records.append(record)

    def aggregate(key: str, subset: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not subset:
            return {"parents": 0, "children": 0}
        total_children = sum(record["children"] for record in subset)
        raw_face_changes = sum(record["raw_face_changes"] for record in subset)
        candidate_face_changes = sum(record["candidate_face_changes"] for record in subset)
        angle_values = [x for record in subset for x in record["normal_angle_deg"]]
        signal_stats = {}
        for signal in signals:
            flips = sum(record["signals"][signal]["flips"] for record in subset)
            comparable = sum(record["signals"][signal]["comparable"] for record in subset)
            signal_stats[signal] = {
                "flips": flips,
                "comparable": comparable,
                "flip_fraction": flips / comparable if comparable else None,
            }
        return {
            "parents": len(subset),
            "children": total_children,
            "raw_face_change_fraction": raw_face_changes / total_children,
            "candidate_face_change_fraction": candidate_face_changes / total_children,
            "pseudonormal_angle_deg": quantiles(angle_values),
            "signals": signal_stats,
        }

    summary: Dict[str, Any] = {}
    for source_class in ("gt_hand", "reviewed_manual"):
        for band, _, _ in BANDS:
            subset = [
                record
                for record in records
                if record["source_class"] == source_class and record["band"] == band
            ]
            if subset:
                summary[f"{source_class}|{band}"] = aggregate(
                    f"{source_class}|{band}",
                    subset,
                )
    return summary


def line_analysis(
    manifest: Dict[str, Any],
    object_name: str,
    raw_triangles: np.ndarray,
    raw_normals: np.ndarray,
    raw_lengths: np.ndarray,
    candidate_triangles: np.ndarray,
    pseudonormal_tables: Dict[str, Any],
    legacy_grid: torch.Tensor,
    legacy_centroid: torch.Tensor,
    legacy_extents: torch.Tensor,
    output_csv_path: Path,
    line_samples: int,
    query_batch: int,
    triangle_chunk: int,
) -> List[Dict[str, Any]]:
    lines = manifest["objects"][object_name].get("lines", [])
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for line in lines:
        start = np.asarray(line["start"], dtype=np.float64)
        end = np.asarray(line["end"], dtype=np.float64)
        points = np.linspace(start, end, line_samples)
        geometry = evaluate_geometry(
            points,
            raw_triangles,
            raw_normals,
            raw_lengths,
            candidate_triangles,
            pseudonormal_tables,
            legacy_grid,
            legacy_centroid,
            legacy_extents,
            query_batch,
            triangle_chunk,
            compute_gradient=False,
        )
        descriptors = [
            {
                "object": object_name,
                "line_id": line.get("id", "line"),
                "region": line.get("region", ""),
                "x": float(point[0]),
                "y": float(point[1]),
                "z": float(point[2]),
            }
            for point in points
        ]
        signal_codes_for_rows(descriptors, geometry, boundary_tol=1e-7)
        for i, descriptor in enumerate(descriptors):
            rows.append(
                {
                    **descriptor,
                    "point_index": i,
                    "raw_udf_m": descriptor["raw_udf_m"],
                    "legacy_sdf_m": descriptor["legacy_sdf_m"],
                    "legacy_valid": descriptor["legacy_valid"],
                    "legacy_signal_code": descriptor["legacy_signal_code"],
                    "raw_normal_signal_code": descriptor["raw_normal_signal_code"],
                    "pseudonormal_signal_code": descriptor["pseudonormal_signal_code"],
                    "winding": descriptor["winding"],
                    "winding_signal_code": descriptor["winding_signal_code"],
                }
            )
        signal_summary = {}
        for signal in SIGNALS:
            codes = [descriptor[f"{signal}_signal_code"] for descriptor in descriptors]
            unknown = [descriptor[f"{signal}_unknown"] for descriptor in descriptors]
            material = sum(code == 1 for code in codes)
            free = sum(code == -1 for code in codes)
            signal_summary[signal] = {
                "material_count": material,
                "free_count": free,
                "unknown_count": sum(unknown),
                "material_fraction": material / len(codes),
                "free_fraction": free / len(codes),
                "has_material_run": any(
                    codes[i] == 1 and codes[i + 1] == 1
                    for i in range(len(codes) - 1)
                ),
            }
        legacy_values = np.array([descriptor["legacy_sdf_m"] for descriptor in descriptors])
        winding_values = np.array([descriptor["winding"] for descriptor in descriptors])
        segment = np.linalg.norm(end - start) / max(line_samples - 1, 1)
        legacy_jump = (
            float(np.max(np.abs(np.diff(legacy_values)) / segment))
            if len(legacy_values) > 1 and np.isfinite(legacy_values).all()
            else None
        )
        winding_jump = (
            float(np.max(np.abs(np.diff(winding_values)) / segment))
            if len(winding_values) > 1 and np.isfinite(winding_values).all()
            else None
        )
        summaries.append(
            {
                "object": object_name,
                "line_id": line.get("id", "line"),
                "region": line.get("region", ""),
                "evidence": line.get("evidence", ""),
                "start": line["start"],
                "end": line["end"],
                "samples": len(points),
                "segment_m": float(segment),
                "signals": signal_summary,
                "legacy_signed_jump_max_per_m": legacy_jump,
                "winding_jump_max_per_m": winding_jump,
                "raw_udf_m": quantiles([descriptor["raw_udf_m"] for descriptor in descriptors]),
                "legacy_sdf_m": quantiles(legacy_values),
                "material_interval_truth": (
                    "reviewed_tabletop_material"
                    if line.get("region") == "tabletop_material"
                    else "unknown_or_not_certified"
                ),
            }
        )
    fields = sorted({key for row in rows for key in row})
    with output_csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return summaries


def fast_winding_status() -> Dict[str, Any]:
    packages = {
        "libigl": "libigl",
        "igl": "igl",
        "fast_winding_number": "fast_winding_number",
    }
    available = [
        name for name, module in packages.items() if importlib.util.find_spec(module)
    ]
    return {
        "status": "NOT_RUN_OR_UNAVAILABLE",
        "available_specialized_packages": available,
        "used": "direct generalized winding implementation already present in material_oracle_probe",
        "reason": (
            "No installed specialized fast-winding backend was found; "
            "no new complex dependency or approximation was introduced."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root_folder", type=Path, required=True)
    parser.add_argument(
        "--queries_csv",
        type=Path,
        default=REPOSITORY_ROOT
        / "outputs"
        / "bps_surface_queries_local_20260907"
        / "queries.csv",
    )
    parser.add_argument(
        "--probe_manifest",
        type=Path,
        default=REPOSITORY_ROOT / "tests" / "fixtures" / "material_oracle_probes_v1.json",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Must not exist; must be outside the processed-data root.",
    )
    parser.add_argument("--objects", nargs="+", default=list(OBJECTS), choices=OBJECTS)
    parser.add_argument("--query_batch_size", type=int, default=16)
    parser.add_argument("--triangle_chunk_size", type=int, default=1024)
    parser.add_argument("--line_samples", type=int, default=33)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.query_batch_size <= 0 or args.triangle_chunk_size <= 0:
        raise ValueError("Query/triangle chunk sizes must be positive")
    if args.line_samples < 3:
        raise ValueError("--line_samples must be at least 3")
    if len(set(args.objects)) != len(args.objects):
        raise ValueError("Object names must be unique")
    data_root = args.data_root_folder.resolve()
    output = args.output_dir.resolve()
    if output == data_root or data_root in output.parents:
        raise ValueError("Output must be outside the processed-data root")


def process_object(
    args: argparse.Namespace,
    object_name: str,
    rows: Sequence[Dict[str, Any]],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    data_root = args.data_root_folder.resolve()
    mesh_path = data_root / "rest_object_geo" / f"{object_name}.ply"
    vertices, faces = read_canonical_ply(mesh_path)
    raw_triangles = vertices[faces]
    raw_normals, raw_lengths = face_normals(vertices, faces)
    (
        candidate_vertices,
        candidate_faces,
        candidate_info,
        _,
    ) = conservative_candidate(vertices, faces)
    candidate_triangles = candidate_vertices[candidate_faces]
    candidate_normals, candidate_lengths = face_normals(
        candidate_vertices,
        candidate_faces,
    )
    pseudonormal_tables = build_pseudonormal_tables(
        candidate_vertices,
        candidate_faces,
        candidate_normals,
        candidate_lengths,
    )
    legacy_grid, legacy_centroid, legacy_extents = load_legacy_sdf(
        data_root,
        object_name,
    )
    points = np.array(
        [[row["x"], row["y"], row["z"]] for row in rows],
        dtype=np.float64,
    )
    boundary_tol = float(
        max(
            1e-6,
            1e-5 * np.ptp(vertices, axis=0).max(),
        )
    )
    geometry = evaluate_geometry(
        points,
        raw_triangles,
        raw_normals,
        raw_lengths,
        candidate_triangles,
        pseudonormal_tables,
        legacy_grid,
        legacy_centroid,
        legacy_extents,
        args.query_batch_size,
        args.triangle_chunk_size,
        compute_gradient=True,
    )
    signal_codes_for_rows(rows, geometry, boundary_tol)
    gradient = geometry["legacy_gradient"]
    gradient_valid = geometry["legacy_gradient_valid"]
    for position, row in enumerate(rows):
        if not row["is_base"]:
            continue
        row["legacy_gradient_x"] = float(gradient[position, 0])
        row["legacy_gradient_y"] = float(gradient[position, 1])
        row["legacy_gradient_z"] = float(gradient[position, 2])
        row["legacy_gradient_valid"] = bool(gradient_valid[position])

    line_csv = args.output_dir / f"{object_name}_line_points.csv"
    line_summaries = line_analysis(
        manifest,
        object_name,
        raw_triangles,
        raw_normals,
        raw_lengths,
        candidate_triangles,
        pseudonormal_tables,
        legacy_grid,
        legacy_centroid,
        legacy_extents,
        line_csv,
        args.line_samples,
        args.query_batch_size,
        args.triangle_chunk_size,
    )
    return {
        "object": object_name,
        "mesh_sha256": sha256(mesh_path),
        "raw_topology": {
            "vertices": int(len(vertices)),
            "faces": int(len(faces)),
        },
        "candidate_topology": candidate_info,
        "rows": rows,
        "band_summary": band_summary(rows, SIGNALS),
        "base_band_summary": base_band_summary(rows, SIGNALS),
        "label_metrics": label_metrics(rows, SIGNALS),
        "free_region_metrics": free_region_metrics(rows, SIGNALS),
        "stability_metrics": stability_metrics(rows, SIGNALS),
        "lines": line_summaries,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    input_hashes: Dict[str, str] = {}
    input_paths: List[Path] = [
        args.queries_csv,
        args.probe_manifest,
        Path(__file__),
        Path(__file__).with_name("audit_bps_surface_queries.py"),
        Path(__file__).with_name("material_oracle_probe.py"),
        REPOSITORY_ROOT / "manip" / "model" / "sdf_utils.py",
    ]
    for object_name in args.objects:
        input_paths.extend(
            [
                args.data_root_folder / "rest_object_geo" / f"{object_name}.ply",
                args.data_root_folder
                / "rest_object_sdf_256_npy_files"
                / f"{object_name}.ply.npy",
                args.data_root_folder
                / "rest_object_sdf_256_npy_files"
                / f"{object_name}.ply.json",
            ]
        )
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        input_hashes[str(path.resolve())] = sha256(path)

    rows, query_manifest = build_rows(args.objects, args.queries_csv, args.probe_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.probe_manifest.read_text(encoding="utf-8"))

    object_results = []
    for object_name in args.objects:
        object_rows = [row for row in rows if row["object"] == object_name]
        if not object_rows:
            raise ValueError(f"No diagnostic rows for {object_name}")
        object_results.append(
            process_object(args, object_name, object_rows, manifest)
        )

    all_rows = [row for result in object_results for row in result["rows"]]
    csv_path = args.output_dir / "queries.csv"
    fields = sorted({key for row in all_rows for key in row})
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    after_hashes = {str(path.resolve()): sha256(path) for path in input_paths}
    summary = {
        "scope": "CPU_LOCAL_MATERIAL_SIGNAL_DIAGNOSTIC_U1_PENETRATION_NOT_GATE_PASS",
        "objects": [
            {
                key: value
                for key, value in result.items()
                if key != "rows"
            }
            for result in object_results
        ],
        "query_manifest": query_manifest,
        "prediction_queries": {
            "status": "MISSING",
            "count": 0,
            "evidence": (
                "No saved model prediction query cache, generated .npz result, "
                "or prediction-query CSV was found in the inspected workspace."
            ),
            "gt_not_substituted": True,
        },
        "fast_winding": fast_winding_status(),
        "analytic_fixture": run_fixtures(),
        "input_hashes": input_hashes,
        "input_hashes_unchanged": input_hashes == after_hashes,
        "elapsed_seconds": time.perf_counter() - started,
        "memory": memory_snapshot(),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "CPU",
        },
        "limitations": [
            "Unsigned distance uses the canonical raw triangle union; it is not certified physical surface accuracy.",
            "Candidate coherent orientation is an explicit topological candidate, not material truth.",
            "Legacy SDF sign is compared and reported but does not become ground truth.",
            "Unknown results are retained; difficult samples are not dropped.",
            "GT palm queries are not saved model prediction queries.",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "CPU_LOCAL_MATERIAL_SIGNAL_DIAGNOSTIC_COMPLETE",
                "output": str(args.output_dir),
                "query_rows": len(all_rows),
                "input_hashes_unchanged": input_hashes == after_hashes,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
