"""Minimal contact-conditioned sectional geometry for bimanual diagnostics.

The module intentionally uses only the Python standard library.  The real-data
CLI converts NumPy/trimesh values to tuples at its boundary, while the geometry
and synthetic smoke tests stay runnable in lightweight environments.

The prior is defined by two planes:

* a gravity-vertical plane through the object centre and source contact; and
* a gravity-horizontal plane at the source-contact height.

Their intersection is a line through the source contact.  Surface candidates
near the inward half of this line are possible paired-contact seeds.  They are
not guaranteed grasps; force closure, reachability, collision, task semantics,
and temporal consistency remain outside this diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


Vec3 = Tuple[float, float, float]
METHODS = ("random", "center_antipode", "section_chord", "section_chord_normal")


def _vec3(value: Sequence[float]) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"Expected a 3-vector, got length {len(value)}")
    result = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"Vector contains NaN or Inf: {result}")
    return result


def add(left: Vec3, right: Vec3) -> Vec3:
    return tuple(left[i] + right[i] for i in range(3))  # type: ignore[return-value]


def subtract(left: Vec3, right: Vec3) -> Vec3:
    return tuple(left[i] - right[i] for i in range(3))  # type: ignore[return-value]


def scale(vector: Vec3, factor: float) -> Vec3:
    return tuple(component * factor for component in vector)  # type: ignore[return-value]


def dot(left: Vec3, right: Vec3) -> float:
    return sum(left[i] * right[i] for i in range(3))


def cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def norm(vector: Vec3) -> float:
    return math.sqrt(dot(vector, vector))


def normalize(vector: Vec3, *, eps: float = 1e-10) -> Vec3:
    magnitude = norm(vector)
    if magnitude < eps:
        raise ValueError(f"Cannot normalize near-zero vector: {vector}")
    return scale(vector, 1.0 / magnitude)


def distance(left: Vec3, right: Vec3) -> float:
    return norm(subtract(left, right))


def point_line_distance(point: Vec3, origin: Vec3, direction: Vec3) -> float:
    unit_direction = normalize(direction)
    offset = subtract(point, origin)
    perpendicular = subtract(offset, scale(unit_direction, dot(offset, unit_direction)))
    return norm(perpendicular)


def point_plane_distance(point: Vec3, origin: Vec3, normal: Vec3) -> float:
    return abs(dot(subtract(point, origin), normalize(normal)))


def world_to_object(
    point_world: Sequence[float],
    object_rotation: Sequence[Sequence[float]],
    object_com: Sequence[float],
) -> Vec3:
    """Apply the repository's row-vector `(point - com) @ rotation` convention."""
    point = _vec3(point_world)
    centre = _vec3(object_com)
    if len(object_rotation) != 3 or any(len(row) != 3 for row in object_rotation):
        raise ValueError("object_rotation must be a 3x3 matrix")
    relative = subtract(point, centre)
    result = tuple(
        sum(relative[row] * float(object_rotation[row][column]) for row in range(3))
        for column in range(3)
    )
    return _vec3(result)


@dataclass(frozen=True)
class ContactSectionFrame:
    """Canonical frame for the two contact-conditioned section planes."""

    centre: Vec3
    contact: Vec3
    up: Vec3
    radial: Vec3
    vertical_normal: Vec3
    horizontal_origin: Vec3
    extent: float

    @property
    def inward(self) -> Vec3:
        return scale(self.radial, -1.0)

    def vertical_plane_distance(self, point: Sequence[float]) -> float:
        return point_plane_distance(_vec3(point), self.centre, self.vertical_normal)

    def horizontal_plane_distance(self, point: Sequence[float]) -> float:
        return point_plane_distance(_vec3(point), self.horizontal_origin, self.up)

    def chord_distance(self, point: Sequence[float]) -> float:
        return point_line_distance(_vec3(point), self.contact, self.radial)

    def inward_travel(self, point: Sequence[float]) -> float:
        return dot(subtract(_vec3(point), self.contact), self.inward)


def _least_parallel_axis(up: Vec3) -> Vec3:
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return min(axes, key=lambda axis: abs(dot(axis, up)))


def build_contact_section_frame(
    contact: Sequence[float],
    centre: Sequence[float],
    up: Sequence[float],
    extent: float,
    fallback_axis: Sequence[float] = (1.0, 0.0, 0.0),
) -> ContactSectionFrame:
    """Build a stable gravity/contact-aligned frame in object coordinates."""
    if not math.isfinite(extent) or extent <= 0.0:
        raise ValueError(f"extent must be positive and finite, got {extent}")
    contact_vec = _vec3(contact)
    centre_vec = _vec3(centre)
    up_vec = normalize(_vec3(up))
    offset = subtract(contact_vec, centre_vec)
    height = dot(offset, up_vec)
    horizontal = subtract(offset, scale(up_vec, height))

    if norm(horizontal) < 1e-8:
        fallback = _vec3(fallback_axis)
        horizontal = subtract(fallback, scale(up_vec, dot(fallback, up_vec)))
        if norm(horizontal) < 1e-8:
            fallback = _least_parallel_axis(up_vec)
            horizontal = subtract(fallback, scale(up_vec, dot(fallback, up_vec)))

    radial = normalize(horizontal)
    vertical_normal = normalize(cross(radial, up_vec))
    horizontal_origin = add(centre_vec, scale(up_vec, height))
    return ContactSectionFrame(
        centre=centre_vec,
        contact=contact_vec,
        up=up_vec,
        radial=radial,
        vertical_normal=vertical_normal,
        horizontal_origin=horizontal_origin,
        extent=float(extent),
    )


def nearest_candidate(point: Sequence[float], candidates: Sequence[Sequence[float]]) -> Tuple[int, float]:
    if not candidates:
        raise ValueError("At least one surface candidate is required")
    query = _vec3(point)
    best_index = 0
    best_distance = math.inf
    for index, candidate in enumerate(candidates):
        candidate_distance = distance(query, _vec3(candidate))
        if candidate_distance < best_distance:
            best_index = index
            best_distance = candidate_distance
    return best_index, best_distance


def _validate_surface(
    candidates: Sequence[Sequence[float]],
    normals: Sequence[Sequence[float]],
) -> Tuple[List[Vec3], List[Vec3]]:
    if len(candidates) != len(normals):
        raise ValueError("candidates and normals must have equal length")
    if len(candidates) < 2:
        raise ValueError("At least two surface candidates are required")
    points = [_vec3(candidate) for candidate in candidates]
    normal_vectors = [normalize(_vec3(normal)) for normal in normals]
    return points, normal_vectors


def rank_surface_candidates(
    candidates: Sequence[Sequence[float]],
    normals: Sequence[Sequence[float]],
    source_index: int,
    frame: ContactSectionFrame,
    *,
    seed: int,
    exclusion_ratio: float = 0.08,
    normal_weight: float = 0.25,
) -> Dict[str, List[int]]:
    """Rank the same eligible surface candidates with four fixed baselines."""
    points, normal_vectors = _validate_surface(candidates, normals)
    if not 0 <= source_index < len(points):
        raise IndexError(f"source_index {source_index} is outside {len(points)} candidates")
    if not 0.0 <= exclusion_ratio < 1.0:
        raise ValueError("exclusion_ratio must lie in [0, 1)")
    if normal_weight < 0.0:
        raise ValueError("normal_weight must be non-negative")

    source = points[source_index]
    source_normal = normal_vectors[source_index]
    exclusion_distance = exclusion_ratio * frame.extent
    eligible = [
        index
        for index, point in enumerate(points)
        if index != source_index and distance(point, source) >= exclusion_distance
    ]
    if not eligible:
        raise ValueError("Source-contact exclusion removed every surface candidate")

    rng = random.Random(seed)
    random_cost = {index: rng.random() for index in eligible}
    central_antipode = subtract(scale(frame.centre, 2.0), source)
    antipode_cost = {
        index: distance(points[index], central_antipode) / frame.extent
        for index in eligible
    }

    chord_cost: Dict[int, float] = {}
    chord_normal_cost: Dict[int, float] = {}
    for index in eligible:
        point = points[index]
        line_distance = frame.chord_distance(point) / frame.extent
        travel = frame.inward_travel(point) / frame.extent
        wrong_direction_penalty = max(0.0, -travel)
        # Surface candidates close to the inward chord are preferred.  A small
        # travel reward selects the far boundary instead of the source patch.
        base_cost = line_distance + 0.50 * wrong_direction_penalty - 0.05 * travel
        opposed_normal_penalty = 0.5 * (dot(source_normal, normal_vectors[index]) + 1.0)
        chord_cost[index] = base_cost
        chord_normal_cost[index] = base_cost + normal_weight * opposed_normal_penalty

    cost_maps: Mapping[str, Mapping[int, float]] = {
        "random": random_cost,
        "center_antipode": antipode_cost,
        "section_chord": chord_cost,
        "section_chord_normal": chord_normal_cost,
    }
    return {
        method: sorted(eligible, key=lambda index: (costs[index], index))
        for method, costs in cost_maps.items()
    }


def evaluate_contact_pair(
    *,
    sequence_name: str,
    absolute_frame: int,
    source_hand: str,
    source_point: Sequence[float],
    target_point: Sequence[float],
    candidates: Sequence[Sequence[float]],
    normals: Sequence[Sequence[float]],
    centre: Sequence[float],
    up: Sequence[float],
    extent: float,
    seed: int,
    top_ks: Sequence[int] = (1, 5, 10),
    hit_radius_ratio: float = 0.08,
    exclusion_ratio: float = 0.08,
    normal_weight: float = 0.25,
) -> Dict[str, object]:
    """Evaluate one ordered source-hand to target-hand contact sample."""
    if source_hand not in {"left", "right"}:
        raise ValueError("source_hand must be 'left' or 'right'")
    if not top_ks or any(int(k) <= 0 for k in top_ks):
        raise ValueError("top_ks must contain positive integers")
    if hit_radius_ratio <= 0.0:
        raise ValueError("hit_radius_ratio must be positive")

    points, normal_vectors = _validate_surface(candidates, normals)
    source_index, source_surface_distance = nearest_candidate(source_point, points)
    target_index, target_surface_distance = nearest_candidate(target_point, points)
    source_surface = points[source_index]
    target_surface = points[target_index]
    frame = build_contact_section_frame(source_surface, centre, up, extent)
    rankings = rank_surface_candidates(
        points,
        normal_vectors,
        source_index,
        frame,
        seed=seed,
        exclusion_ratio=exclusion_ratio,
        normal_weight=normal_weight,
    )

    target_normal_cosine = dot(normal_vectors[source_index], normal_vectors[target_index])
    result: Dict[str, object] = {
        "sequence_name": sequence_name,
        "absolute_frame": int(absolute_frame),
        "source_hand": source_hand,
        "target_hand": "right" if source_hand == "left" else "left",
        "candidate_count": len(points),
        "eligible_candidate_count": len(rankings[METHODS[0]]),
        "source_surface_distance_norm": source_surface_distance / extent,
        "target_surface_distance_norm": target_surface_distance / extent,
        "target_vertical_plane_distance_norm": frame.vertical_plane_distance(target_surface) / extent,
        "target_horizontal_plane_distance_norm": frame.horizontal_plane_distance(target_surface) / extent,
        "target_chord_distance_norm": frame.chord_distance(target_surface) / extent,
        "target_inward_travel_norm": frame.inward_travel(target_surface) / extent,
        "target_normal_cosine": target_normal_cosine,
        "radial_x": frame.radial[0],
        "radial_y": frame.radial[1],
        "radial_z": frame.radial[2],
        "up_x": frame.up[0],
        "up_y": frame.up[1],
        "up_z": frame.up[2],
        "source_x": source_surface[0],
        "source_y": source_surface[1],
        "source_z": source_surface[2],
        "target_x": target_surface[0],
        "target_y": target_surface[1],
        "target_z": target_surface[2],
    }

    hit_radius = hit_radius_ratio * extent
    for method, order in rankings.items():
        top1_point = points[order[0]]
        result[f"{method}_top1_x"] = top1_point[0]
        result[f"{method}_top1_y"] = top1_point[1]
        result[f"{method}_top1_z"] = top1_point[2]
        try:
            target_rank = order.index(target_index) + 1
        except ValueError:
            target_rank = len(order) + 1
        result[f"{method}_rank"] = target_rank
        result[f"{method}_reciprocal_rank"] = 1.0 / target_rank
        # The target may be masked with the source neighbourhood.  In that case
        # it is ranked just after all eligible candidates, but the normalized
        # percentile remains a probability-like value in [0, 1].
        result[f"{method}_rank_percentile"] = min(target_rank, len(order)) / max(
            1, len(order)
        )
        for top_k in sorted(set(int(k) for k in top_ks)):
            selected = order[: min(top_k, len(order))]
            best_error = min(distance(points[index], target_surface) for index in selected)
            result[f"{method}_top{top_k}_error_norm"] = best_error / extent
            result[f"{method}_top{top_k}_hit"] = float(best_error <= hit_radius)
    return result


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def summarize_results(
    rows: Sequence[Mapping[str, object]],
    *,
    top_ks: Sequence[int] = (1, 5, 10),
) -> Dict[str, object]:
    """Macro-average method metrics over sequences, then report sample medians."""
    if not rows:
        raise ValueError("Cannot summarize an empty result set")
    sequences = sorted({str(row["sequence_name"]) for row in rows})
    by_sequence = {
        sequence: [row for row in rows if str(row["sequence_name"]) == sequence]
        for sequence in sequences
    }
    method_summary: Dict[str, object] = {}
    for method in METHODS:
        metrics: Dict[str, float] = {}
        for metric_suffix in ("reciprocal_rank", "rank_percentile"):
            field = f"{method}_{metric_suffix}"
            per_sequence = [
                _mean(float(row[field]) for row in sequence_rows)
                for sequence_rows in by_sequence.values()
            ]
            metrics[f"sequence_macro_{metric_suffix}"] = _mean(per_sequence)
        for top_k in sorted(set(int(k) for k in top_ks)):
            for metric_suffix in (f"top{top_k}_hit", f"top{top_k}_error_norm"):
                field = f"{method}_{metric_suffix}"
                per_sequence = [
                    _mean(float(row[field]) for row in sequence_rows)
                    for sequence_rows in by_sequence.values()
                ]
                metrics[f"sequence_macro_{metric_suffix}"] = _mean(per_sequence)
        method_summary[method] = metrics

    geometry_fields = (
        "source_surface_distance_norm",
        "target_surface_distance_norm",
        "target_vertical_plane_distance_norm",
        "target_horizontal_plane_distance_norm",
        "target_chord_distance_norm",
        "target_inward_travel_norm",
        "target_normal_cosine",
    )
    geometry = {
        f"median_{field}": statistics.median(float(row[field]) for row in rows)
        for field in geometry_fields
    }
    return {
        "sample_count": len(rows),
        "sequence_count": len(sequences),
        "methods": method_summary,
        "geometry": geometry,
    }


def make_box_surface_candidates(
    half_extents: Sequence[float] = (1.0, 0.7, 0.5),
    steps: int = 7,
) -> Tuple[List[Vec3], List[Vec3]]:
    """Create a deterministic box surface used by dependency-free smoke tests."""
    hx, hy, hz = _vec3(half_extents)
    if min(hx, hy, hz) <= 0.0:
        raise ValueError("half_extents must be positive")
    if steps < 3:
        raise ValueError("steps must be at least 3")

    axes = [
        [(-extent + 2.0 * extent * index / (steps - 1)) for index in range(steps)]
        for extent in (hx, hy, hz)
    ]
    point_to_normal: Dict[Tuple[float, float, float], Vec3] = {}
    for sign in (-1.0, 1.0):
        for y in axes[1]:
            for z in axes[2]:
                point_to_normal.setdefault((sign * hx, y, z), (sign, 0.0, 0.0))
        for x in axes[0]:
            for z in axes[2]:
                point_to_normal.setdefault((x, sign * hy, z), (0.0, sign, 0.0))
        for x in axes[0]:
            for y in axes[1]:
                point_to_normal.setdefault((x, y, sign * hz), (0.0, 0.0, sign))
    points = sorted(point_to_normal)
    return points, [point_to_normal[point] for point in points]
