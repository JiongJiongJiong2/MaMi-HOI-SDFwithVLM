"""CPU geometry primitives for the BPR-01 paired-contact diagnostic.

This module intentionally does not reuse the old candidate-pool projection
for targets.  Source and target contacts are projected onto the original
triangle surface first; the sampled candidate pool is used only when scoring
the same rankings under a fixed budget.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from manip.geometry.sectional_prior import (
    METHODS,
    build_contact_section_frame,
    normalize,
)


def sample_surface_candidates(
    vertices: np.ndarray,
    faces: np.ndarray,
    count: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample deterministic area-weighted surface points and face normals."""
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3]")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3]")
    if count < 2:
        raise ValueError("count must be at least 2")

    triangles = vertices[faces]
    cross_products = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    double_areas = np.linalg.norm(cross_products, axis=1)
    valid = double_areas > 1e-12
    if not bool(valid.any()):
        raise ValueError("Mesh contains no non-degenerate triangle")
    triangles = triangles[valid]
    cross_products = cross_products[valid]
    probabilities = double_areas[valid] / double_areas[valid].sum()

    rng = np.random.default_rng(seed)
    face_indices = rng.choice(
        len(triangles),
        size=count,
        replace=True,
        p=probabilities,
    )
    selected = triangles[face_indices]
    first = rng.random(count)
    second = rng.random(count)
    sqrt_first = np.sqrt(first)
    barycentric = np.stack(
        (
            1.0 - sqrt_first,
            sqrt_first * (1.0 - second),
            sqrt_first * second,
        ),
        axis=1,
    )
    points = (selected * barycentric[:, :, None]).sum(axis=1)
    normals = cross_products[face_indices]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    return points, normals


def closest_triangle_points(
    queries: np.ndarray,
    triangles: np.ndarray,
    *,
    query_batch: int = 16,
    triangle_chunk: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return exact triangle distances, closest points, face ids and normals."""
    queries = np.asarray(queries, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1] != 3:
        raise ValueError("queries must have shape [Q, 3]")
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangles must have shape [F, 3, 3]")
    if query_batch <= 0 or triangle_chunk <= 0:
        raise ValueError("batch sizes must be positive")

    distances = np.empty(len(queries), dtype=np.float64)
    closest = np.empty_like(queries)
    face_ids = np.empty(len(queries), dtype=np.int64)

    for start in range(0, len(queries), query_batch):
        query = queries[start : start + query_batch]
        best = np.full(len(query), np.inf, dtype=np.float64)
        best_point = np.empty_like(query)
        best_face = np.zeros(len(query), dtype=np.int64)

        for offset in range(0, len(triangles), triangle_chunk):
            tri = triangles[offset : offset + triangle_chunk]
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            normal = np.cross(b - a, c - a)
            normal_sq = np.einsum("ki,ki->k", normal, normal)
            safe_normal_sq = np.where(normal_sq > 0.0, normal_sq, 1.0)

            signed_height = np.einsum(
                "qki,ki->qk",
                query[:, None] - a,
                normal,
            ) / safe_normal_sq
            projection = query[:, None] - signed_height[..., None] * normal
            inside = np.broadcast_to(normal_sq > 0.0, signed_height.shape).copy()
            for edge_start, edge_end in ((a, b), (b, c), (c, a)):
                side = np.einsum(
                    "qki,ki->qk",
                    np.cross(edge_end - edge_start, projection - edge_start),
                    normal,
                )
                inside &= side >= -1e-12 * normal_sq

            local_best = np.where(
                inside,
                signed_height**2 * normal_sq,
                np.inf,
            )
            local_point = projection.copy()
            for edge_start, edge_end in ((a, b), (b, c), (c, a)):
                edge = edge_end - edge_start
                edge_sq = np.einsum("ki,ki->k", edge, edge)
                t = np.einsum(
                    "qki,ki->qk",
                    query[:, None] - edge_start,
                    edge,
                ) / np.where(edge_sq > 0.0, edge_sq, 1.0)
                edge_point = (
                    edge_start
                    + np.clip(t, 0.0, 1.0)[..., None] * edge
                )
                delta = query[:, None] - edge_point
                edge_distance_sq = np.einsum("qki,qki->qk", delta, delta)
                improve = edge_distance_sq < local_best
                local_best[improve] = edge_distance_sq[improve]
                local_point[improve] = edge_point[improve]

            local_face = local_best.argmin(axis=1)
            local_distance_sq = local_best[
                np.arange(len(query)),
                local_face,
            ]
            improve = local_distance_sq < best
            best[improve] = local_distance_sq[improve]
            best_point[improve] = local_point[
                np.arange(len(query)),
                local_face,
            ][improve]
            best_face[improve] = offset + local_face[improve]

        distances[start : start + len(query)] = np.sqrt(best)
        closest[start : start + len(query)] = best_point
        face_ids[start : start + len(query)] = best_face

    chosen = triangles[face_ids]
    normals = np.cross(chosen[:, 1] - chosen[:, 0], chosen[:, 2] - chosen[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("Closest triangle has zero area")
    normals /= lengths[:, None]
    return distances, closest, face_ids, normals


def diverse_order(
    order: Sequence[int],
    points: np.ndarray,
    radius: float,
    count: int,
) -> List[int]:
    """Take ranked candidates while requiring spatial separation."""
    if count <= 0:
        raise ValueError("count must be positive")
    if radius < 0.0:
        raise ValueError("radius must be non-negative")
    points = np.asarray(points, dtype=np.float64)
    chosen: List[int] = []
    for index in order:
        index = int(index)
        if not chosen:
            chosen.append(index)
        elif np.all(
            np.linalg.norm(points[np.asarray(chosen)] - points[index], axis=1)
            >= radius
        ):
            chosen.append(index)
        if len(chosen) >= count:
            break
    return chosen


def projected_source_rankings(
    points: np.ndarray,
    normals: np.ndarray,
    source_point: np.ndarray,
    source_normal: np.ndarray,
    centre: np.ndarray,
    up: np.ndarray,
    extent: float,
    *,
    seed: int,
    exclusion_ratio: float = 0.08,
    normal_weight: float = 0.25,
) -> Tuple[Dict[str, List[int]], object, np.ndarray]:
    """Rank a fixed candidate pool from an independently projected source."""
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    source_point = np.asarray(source_point, dtype=np.float64)
    source_normal = np.asarray(
        normalize(tuple(float(v) for v in source_normal)),
        dtype=np.float64,
    )
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if normals.shape != points.shape:
        raise ValueError("normals must match points")
    if source_point.shape != (3,):
        raise ValueError("source_point must have shape [3]")
    if not 0.0 <= exclusion_ratio < 1.0:
        raise ValueError("exclusion_ratio must lie in [0, 1)")
    if normal_weight < 0.0:
        raise ValueError("normal_weight must be non-negative")

    frame = build_contact_section_frame(
        source_point,
        centre,
        up,
        extent,
    )
    delta = points - source_point
    distances = np.linalg.norm(delta, axis=1)
    eligible = np.flatnonzero(distances >= exclusion_ratio * extent)
    if not len(eligible):
        raise ValueError("Source-contact exclusion removed every candidate")

    rng = np.random.default_rng(seed)
    random_cost = rng.random(len(points))
    central_antipode = 2.0 * np.asarray(centre, dtype=np.float64) - source_point
    antipode_cost = np.linalg.norm(points - central_antipode, axis=1) / extent
    radial = np.asarray(frame.radial, dtype=np.float64)
    inward = np.asarray(frame.inward, dtype=np.float64)
    line_distance = np.abs(delta @ radial) / extent
    travel = delta @ inward / extent
    base_cost = line_distance + 0.50 * np.maximum(-travel, 0.0) - 0.05 * travel
    normal_cosine = normals @ source_normal
    normal_cost = base_cost + normal_weight * 0.5 * (normal_cosine + 1.0)

    costs = {
        "random": random_cost,
        "center_antipode": antipode_cost,
        "section_chord": base_cost,
        "section_chord_normal": normal_cost,
    }
    rankings = {
        method: [
            int(index)
            for index in eligible[
                np.lexsort((eligible, costs[method][eligible]))
            ]
        ]
        for method in METHODS
    }
    return rankings, frame, eligible


def evaluate_projected_contact(
    *,
    sequence_name: str,
    absolute_frame: int,
    source_hand: str,
    source_point: np.ndarray,
    source_normal: np.ndarray,
    target_point: np.ndarray,
    target_normal: np.ndarray,
    candidates: np.ndarray,
    normals: np.ndarray,
    centre: np.ndarray,
    up: np.ndarray,
    extent: float,
    seed: int,
    top_ks: Sequence[int] = (1, 5, 10),
    hit_radius_ratio: float = 0.08,
    exclusion_ratio: float = 0.08,
    normal_weight: float = 0.25,
    source_projection_error_norm: float | None = None,
    target_projection_error_norm: float | None = None,
    variant: str = "clean",
    noise_mm: float = 0.0,
    noise_direction: str = "clean",
) -> Dict[str, object]:
    """Evaluate one projected source-to-target pair under a fixed pool budget."""
    if source_hand not in {"left", "right"}:
        raise ValueError("source_hand must be 'left' or 'right'")
    if hit_radius_ratio <= 0.0:
        raise ValueError("hit_radius_ratio must be positive")
    if not top_ks or any(int(k) <= 0 for k in top_ks):
        raise ValueError("top_ks must contain positive integers")

    candidates = np.asarray(candidates, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    source_point = np.asarray(source_point, dtype=np.float64)
    target_point = np.asarray(target_point, dtype=np.float64)
    rankings, frame, eligible = projected_source_rankings(
        candidates,
        normals,
        source_point,
        source_normal,
        centre,
        up,
        extent,
        seed=seed,
        exclusion_ratio=exclusion_ratio,
        normal_weight=normal_weight,
    )
    radius = hit_radius_ratio * extent
    target_distances = np.linalg.norm(candidates - target_point, axis=1)
    full_pool_hit = bool(np.any(target_distances <= radius))
    eligible_pool_hit = bool(np.any(target_distances[eligible] <= radius))
    source_normal_array = np.asarray(
        normalize(tuple(float(v) for v in source_normal)),
        dtype=np.float64,
    )
    target_normal_array = np.asarray(
        normalize(tuple(float(v) for v in target_normal)),
        dtype=np.float64,
    )
    target_chord_distance = frame.chord_distance(target_point) / extent

    result: Dict[str, object] = {
        "sequence_name": sequence_name,
        "absolute_frame": int(absolute_frame),
        "source_hand": source_hand,
        "target_hand": "right" if source_hand == "left" else "left",
        "variant": variant,
        "noise_mm": float(noise_mm),
        "noise_direction": noise_direction,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "source_projection_error_norm": source_projection_error_norm,
        "target_projection_error_norm": target_projection_error_norm,
        "target_chord_distance_norm": target_chord_distance,
        "target_normal_cosine": float(source_normal_array @ target_normal_array),
        "pool_recall": float(full_pool_hit),
        "eligible_pool_recall": float(eligible_pool_hit),
        "oracle_top1_hit": float(eligible_pool_hit),
        "oracle_top5_hit": float(eligible_pool_hit),
        "oracle_top10_hit": float(eligible_pool_hit),
    }

    max_top_k = max(int(k) for k in top_ks)
    for method, order in rankings.items():
        chosen = diverse_order(
            order,
            candidates,
            exclusion_ratio * extent,
            max_top_k,
        )
        result[f"{method}_diverse_count"] = len(chosen)
        for top_k in sorted(set(int(k) for k in top_ks)):
            selected = chosen[:top_k]
            if not selected:
                raise ValueError("Ranking produced no selectable candidate")
            best_error = float(
                np.min(np.linalg.norm(candidates[selected] - target_point, axis=1))
            )
            result[f"{method}_top{top_k}_error_norm"] = best_error / extent
            result[f"{method}_top{top_k}_hit"] = float(best_error <= radius)
    return result


def macro_mean(
    rows: Sequence[Mapping[str, object]],
    field: str,
    *,
    sequence_field: str = "sequence_name",
) -> float:
    """Average a field within sequence, then across sequences."""
    if not rows:
        return float("nan")
    sequences = sorted({str(row[sequence_field]) for row in rows})
    per_sequence = []
    for sequence in sequences:
        values = [
            float(row[field])
            for row in rows
            if str(row[sequence_field]) == sequence
        ]
        if values:
            per_sequence.append(float(np.mean(values)))
    return float(np.mean(per_sequence))


def bootstrap_difference(
    rows: Sequence[Mapping[str, object]],
    *,
    method: str,
    baseline: str,
    top_k: int,
    samples: int = 10000,
    seed: int = 1,
) -> Tuple[float, float, float]:
    """Paired sequence bootstrap for method minus baseline hit rate."""
    sequences = sorted({str(row["sequence_name"]) for row in rows})
    differences = np.asarray(
        [
            macro_mean(
                [row for row in rows if str(row["sequence_name"]) == sequence],
                f"{method}_top{top_k}_hit",
            )
            - macro_mean(
                [row for row in rows if str(row["sequence_name"]) == sequence],
                f"{baseline}_top{top_k}_hit",
            )
            for sequence in sequences
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    interval = np.quantile(
        differences[indices].mean(axis=1),
        (0.025, 0.975),
    )
    return float(differences.mean()), float(interval[0]), float(interval[1])


def summarize_variants(
    rows: Sequence[Mapping[str, object]],
    *,
    top_ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, object]:
    """Aggregate BPR rows by variant, sequence, object and direction."""
    if not rows:
        raise ValueError("Cannot summarize empty BPR rows")
    variants = sorted({str(row["variant"]) for row in rows})
    objects = sorted({str(row["object_name"]) for row in rows})
    summary: Dict[str, object] = {}
    for variant in variants:
        variant_rows = [row for row in rows if str(row["variant"]) == variant]
        methods = {
            method: {
                f"top{top_k}_hit": macro_mean(
                    variant_rows,
                    f"{method}_top{top_k}_hit",
                )
                for top_k in sorted(set(int(k) for k in top_ks))
            }
            for method in METHODS
        }
        summary[variant] = {
            "sample_count": len(variant_rows),
            "sequence_count": len(
                {str(row["sequence_name"]) for row in variant_rows}
            ),
            "methods": methods,
            "pool_recall": macro_mean(variant_rows, "pool_recall"),
            "eligible_pool_recall": macro_mean(
                variant_rows,
                "eligible_pool_recall",
            ),
            "oracle_top5_hit": macro_mean(variant_rows, "oracle_top5_hit"),
            "target_chord_distance_norm": macro_mean(
                variant_rows,
                "target_chord_distance_norm",
            ),
            "by_object_top5": {
                object_name: macro_mean(
                    [
                        row
                        for row in variant_rows
                        if str(row["object_name"]) == object_name
                    ],
                    "section_chord_normal_top5_hit",
                )
                for object_name in objects
            },
            "by_direction_top5": {
                source_hand: macro_mean(
                    [
                        row
                        for row in variant_rows
                        if str(row["source_hand"]) == source_hand
                    ],
                    "section_chord_normal_top5_hit",
                )
                for source_hand in ("left", "right")
            },
        }
    return summary
