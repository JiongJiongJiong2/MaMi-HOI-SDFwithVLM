"""CPU-only BPS/vertex/triangle unsigned-query diagnostic; never writes inputs.

Only NumPy is required for geometry. Torch loads the existing BPS basis on CPU;
joblib is needed only for mmap access to existing held-out motion windows.
No dataset constructor, SDF, mesh repair, training, or CUDA calls are used.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings

# Process-local limits, set before importing numeric libraries.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(seed, *parts):
    raw = ":".join(map(str, (seed,) + parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def read_canonical_ply(path):
    """Read the archive's ASCII triangle PLY without loader cleanup/reordering."""
    with Path(path).open("r", encoding="ascii") as handle:
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
        if header[first_property:first_property + 3] != [
            "property float x", "property float y", "property float z"
        ]:
            raise ValueError("Unsupported PLY vertex schema")
        vertices = np.array([[float(v) for v in handle.readline().split()[:3]]
                             for _ in range(nv)], dtype=np.float64)
        faces = []
        for _ in range(nf):
            row = [int(v) for v in handle.readline().split()]
            if len(row) != 4 or row[0] != 3:
                raise ValueError("Only triangle faces are supported")
            faces.append(row[1:])
    faces = np.asarray(faces, dtype=np.int64)
    if (vertices.shape != (nv, 3) or not np.isfinite(vertices).all()
            or not nv or not nf or faces.min() < 0 or faces.max() >= nv):
        raise ValueError(f"Invalid mesh: {path}")
    return vertices, faces


def nearest_points(queries, points, query_batch=16, point_chunk=1024):
    """Exact Euclidean nearest point in the supplied finite point set."""
    queries = np.asarray(queries, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    distances = np.empty(len(queries))
    closest = np.empty_like(queries)
    indices = np.empty(len(queries), dtype=np.int64)
    for start in range(0, len(queries), query_batch):
        q = queries[start:start + query_batch]
        best = np.full(len(q), np.inf)
        cp, ids = np.empty_like(q), np.zeros(len(q), dtype=np.int64)
        for offset in range(0, len(points), point_chunk):
            p = points[offset:offset + point_chunk]
            delta = q[:, None] - p[None]
            d2 = np.einsum("qki,qki->qk", delta, delta)
            k = d2.argmin(axis=1)
            value = d2[np.arange(len(q)), k]
            improve = value < best
            best[improve] = value[improve]
            cp[improve] = p[k[improve]]
            ids[improve] = offset + k[improve]
        distances[start:start + len(q)] = np.sqrt(best)
        closest[start:start + len(q)] = cp
        indices[start:start + len(q)] = ids
    return distances, closest, indices


def nearest_triangles(queries, triangles, query_batch=16, triangle_chunk=1024):
    """Exact point-to-triangle distance (float64), exhaustive bounded blocks.

    Minimize over each triangle's plane projection (when inside) and three
    closed edges. Zero-area triangles reduce to their edges/vertices.
    Opposite duplicate faces have identical geometric distance.
    """
    queries = np.asarray(queries, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    distances, closest = np.empty(len(queries)), np.empty_like(queries)
    for start in range(0, len(queries), query_batch):
        q = queries[start:start + query_batch]
        best, result = np.full(len(q), np.inf), np.empty_like(q)
        for offset in range(0, len(triangles), triangle_chunk):
            tri = triangles[offset:offset + triangle_chunk]
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            normal = np.cross(b - a, c - a)
            nn = np.einsum("ki,ki->k", normal, normal)
            safe_nn = np.where(nn > 0, nn, 1.)
            signed_height = np.einsum("qki,ki->qk", q[:, None] - a, normal) / safe_nn
            projection = q[:, None] - signed_height[..., None] * normal
            inside = np.broadcast_to(nn > 0, signed_height.shape).copy()
            for u, v in ((a, b), (b, c), (c, a)):
                side = np.einsum("qki,ki->qk", np.cross(v - u, projection - u), normal)
                inside &= side >= -1e-12 * nn
            local_best = np.where(inside, signed_height ** 2 * nn, np.inf)
            local_cp = projection.copy()
            for u, v in ((a, b), (b, c), (c, a)):
                edge = v - u
                ee = np.einsum("ki,ki->k", edge, edge)
                t = np.einsum("qki,ki->qk", q[:, None] - u, edge) / np.where(ee > 0, ee, 1.)
                cp = u + np.clip(t, 0., 1.)[..., None] * edge
                delta = q[:, None] - cp
                d2 = np.einsum("qki,qki->qk", delta, delta)
                improve = d2 < local_best
                local_best[improve], local_cp[improve] = d2[improve], cp[improve]
            k = local_best.argmin(axis=1)
            value = local_best[np.arange(len(q)), k]
            improve = value < best
            best[improve] = value[improve]
            result[improve] = local_cp[np.arange(len(q)), k][improve]
        distances[start:start + len(q)] = np.sqrt(best)
        closest[start:start + len(q)] = result
    return distances, closest


def world_to_canonical(points, rotation, com):
    rotation = np.asarray(rotation, dtype=np.float64)
    if (not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=0)
            or abs(np.linalg.det(rotation) - 1.) > 1e-4):
        raise ValueError("Motion contains an invalid object rotation")
    return (np.asarray(points) - com) @ rotation


def load_validation_palms(args, input_hashes):
    if args.geometry_only:
        return {name: [] for name in args.objects}, {"status": "NOT_RUN_GEOMETRY_ONLY"}
    import joblib

    manifest_path = Path(args.split_manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation, test = manifest["validation_sequences"], manifest["test_sequences"]
    if (len(set(validation)) != len(validation) or len(set(test)) != len(test)
            or not validation or set(validation) & set(test) or manifest["window"] != 120):
        raise ValueError("Invalid/disjointness-violating frozen split manifest")
    root = Path(args.data_root_folder)
    processed = root / "cano_test_diffusion_manip_window_120_joints24.p"
    if processed.stat().st_size > 512 * 1024 ** 2:
        raise ValueError("Refusing >512 MiB motion input under the 2 GiB diagnostic budget")
    input_hashes[str(manifest_path.resolve())] = sha256(manifest_path)
    input_hashes[str(processed.resolve())] = sha256(processed)
    # Compressed joblib ignores mmap; treat that warning as a hard stop.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        windows = joblib.load(processed, mmap_mode="r")
    result, selected_sequences, counts = {}, {}, {}
    for name in args.objects:
        names = [s for s in validation if s.split("_")[1] == name]
        names.sort(key=lambda s: stable_seed(args.seed, name, s))
        names = names[:args.sequences_per_object]
        if not names:
            raise ValueError(f"No validation sequences for {name}")
        selected_sequences[name] = names
        result[name] = []
        for sequence in names:
            contact_path = root / "contact_labels_w_semantics_npy_files" / (sequence + ".npy")
            contact = np.load(contact_path, mmap_mode="r", allow_pickle=False)
            input_hashes[str(contact_path.resolve())] = sha256(contact_path)
            candidates = {(hand, active): [] for hand in (0, 1) for active in (False, True)}
            seen = set()
            items = sorted((w for w in windows.values() if w["seq_name"] == sequence),
                           key=lambda w: int(w["start_t_idx"]))
            if not items:
                raise ValueError(f"Manifest sequence absent from motion file: {sequence}")
            for item in items:
                motion, rotations, coms = [item[k] for k in ("motion", "obj_rot_mat", "window_obj_com_pos")]
                if any(not isinstance(x, np.memmap) for x in (motion, rotations, coms)):
                    raise ValueError("Motion fields are not memory-mapped; refusing eager fallback")
                if not (len(motion) == len(rotations) == len(coms)):
                    raise ValueError("Motion/object window lengths differ")
                start = int(item["start_t_idx"])
                for local in range(0, len(motion), args.frame_stride):
                    absolute = start + local
                    if absolute in seen:
                        continue
                    seen.add(absolute)
                    if absolute >= len(contact):
                        raise ValueError(f"Contact frame out of range: {sequence}/{absolute}")
                    world = np.asarray(motion[local, :72], dtype=np.float64).reshape(24, 3)[[22, 23]]
                    points = world_to_canonical(world, rotations[local], coms[local])
                    if not np.isfinite(points).all():
                        raise ValueError("Nonfinite palm points")
                    for hand in (0, 1):
                        label = float(contact[absolute, hand])
                        if label not in (0., 1.):
                            raise ValueError("Expected binary per-hand contact annotation")
                        active = bool(label)
                        candidates[hand, active].append(dict(
                            point=points[hand].tolist(), sequence=sequence, frame=absolute,
                            window_start=start, local_frame=local, joint=22 + hand,
                            contact_annotation=int(active),
                            stratum="gt_palm_contact" if active else "gt_palm_noncontact"))
            for key, values in candidates.items():
                values.sort(key=lambda r: stable_seed(args.seed, sequence, r["frame"], r["joint"]))
                counts[f"{sequence}/{key}"] = {"available": len(values),
                                              "selected": min(len(values), args.hand_samples_per_stratum)}
                result[name].extend(values[:args.hand_samples_per_stratum])
        if not result[name]:
            raise ValueError(f"No palm queries selected for {name}")
    del windows
    gc.collect()
    return result, dict(status="GT_PALMS_VALIDATION_ONLY", sequences=selected_sequences,
                        strata_counts=counts, joblib_version=joblib.__version__,
                        proxy_note="Joints 22/23 are palm proxies, not full-hand contact points or predictions")


def make_queries(vertices, faces, hands, name, seed, samples):
    rng = np.random.default_rng(stable_seed(seed, name, "queries"))
    lo, hi = vertices.min(0), vertices.max(0)
    extent, center = float((hi - lo).max()), (lo + hi) / 2
    voxel = extent / 63.
    # Sampling view only: opposite duplicate faces must not double-count area.
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    tri = vertices[faces[np.sort(first)]]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area2 = np.linalg.norm(cross, axis=1)
    ids = rng.choice(len(tri), size=samples, p=area2 / area2.sum())
    weights = rng.random((samples, 2))
    u, v = np.sqrt(weights[:, 0]), weights[:, 1]
    surface = ((1-u[:, None]) * tri[ids, 0] + (u*(1-v))[:, None] * tri[ids, 1]
               + (u*v)[:, None] * tri[ids, 2])
    normal = cross[ids] / area2[ids, None]
    rows = []

    def add(point, stratum, **extra):
        rows.append(dict(point=np.asarray(point).tolist(), stratum=stratum, **extra))

    offsets = rng.choice([.25, .5, 1., 2.], samples) * voxel
    for k, point in enumerate(surface):
        add(point, "surface_coverage")
        for side in (-1, 1):
            add(point + side * offsets[k] * normal[k], "near_surface_probe",
                offset_m=float(side * offsets[k]))
    for point in center + rng.uniform(-.5, .5, (samples, 3)) * extent:
        add(point, "uniform_cube_probe")
    if name in ("plasticbox", "trashcan"):
        # This is a central-volume probe, not a certified cavity/inside label.
        for x in (.3, .5, .7):
            for y in (.3, .5, .7):
                for z in (.3, .6, .9):
                    add(lo + np.array([x, y, z]) * (hi - lo), "central_volume_probe")
    for row in hands:
        rows.append(dict(row))
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        for side in (-1, 1):
            changed = dict(row)
            changed.update(point=(np.asarray(row["point"]) + side * voxel * direction).tolist(),
                           stratum="gt_palm_perturbed", offset_m=side * voxel,
                           parent_contact_annotation=row["contact_annotation"])
            changed.pop("contact_annotation")  # Perturbed locations have no GT contact label.
            rows.append(changed)
    return rows, extent


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {k: None for k in ("mean", "p50", "p90", "p95", "p99", "max")}
    return dict(zip(("mean", "p50", "p90", "p95", "p99", "max"),
                    [float(values.mean()), *np.quantile(values, [.5, .9, .95, .99, 1]).tolist()]))


def compare_queries(queries, distances, closest, reference_distance, reference_closest, extent):
    error = distances - reference_distance
    if float(error.min()) < -1e-5 * extent:
        raise ValueError("Point-set distance unexpectedly below triangle distance; check coordinates/reference")
    valid = (reference_distance > 1e-6 * extent) & (distances > 1e-6 * extent)
    angles = np.full(len(queries), np.nan)
    pull = closest[valid] - queries[valid]
    reference_pull = reference_closest[valid] - queries[valid]
    cosine = np.einsum("ij,ij->i", pull, reference_pull) / (distances[valid] * reference_distance[valid])
    angles[valid] = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    return error, angles


def benchmark(method, queries, repeats):
    method(queries)  # CPU warmup, excluded.
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        method(queries)
        elapsed.append(time.perf_counter() - start)
    return dict(query_count=len(queries), repeats=repeats,
                median_us_per_query=float(np.median(elapsed) * 1e6 / len(queries)),
                min_us_per_query=float(min(elapsed) * 1e6 / len(queries)),
                implementation="single-thread NumPy float64 exhaustive search; not optimized production latency")


def memory_snapshot():
    result = {}
    for key in ("memory.max", "memory.current", "memory.peak"):
        path = Path("/sys/fs/cgroup") / key
        if path.is_file():
            result[key] = path.read_text().strip()
    try:
        import resource
        result["process_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (key, ctypes.c_size_t) for key in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage")]
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                result["process_peak_working_set_bytes"] = counters.PeakWorkingSetSize
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root_folder", required=True)
    parser.add_argument("--output_dir", required=True, help="Must not exist; must be outside the data root")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--split_manifest", help="Existing frozen validation/test manifest; only validation is read")
    mode.add_argument("--geometry_only", action="store_true")
    parser.add_argument("--objects", nargs="+", default=["plasticbox", "trashcan", "largebox", "smalltable"])
    parser.add_argument("--basis_path", default=str(Path(__file__).resolve().parents[1] / "bps.pt"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--surface_samples", type=int, default=64)
    parser.add_argument("--sequences_per_object", type=int, default=4)
    parser.add_argument("--hand_samples_per_stratum", type=int, default=3)
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--query_batch_size", type=int, default=16)
    parser.add_argument("--triangle_chunk_size", type=int, default=1024)
    parser.add_argument("--timing_queries", type=int, default=32)
    parser.add_argument("--timing_repeats", type=int, default=3)
    args = parser.parse_args()
    for key in ("surface_samples", "sequences_per_object", "hand_samples_per_stratum", "frame_stride",
                "query_batch_size", "triangle_chunk_size", "timing_queries", "timing_repeats"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    if (max(args.query_batch_size, args.triangle_chunk_size) > 2048
            or args.query_batch_size * args.triangle_chunk_size > 32768
            or args.surface_samples > 512 or args.sequences_per_object > 8
            or args.hand_samples_per_stratum > 8 or args.timing_queries > 128):
        parser.error("Request exceeds the bounded small-diagnostic budget")
    if len(set(args.objects)) != len(args.objects) or any(
            not name.replace("_", "").isalnum() for name in args.objects):
        parser.error("Object names must be unique simple names")
    root, output = Path(args.data_root_folder).resolve(), Path(args.output_dir).resolve()
    if output == root or root in output.parents:
        parser.error("Output must be outside the input data root")
    if output.exists():
        parser.error("Output already exists; use a new directory, never overwrite")
    input_hashes = {str(Path(args.basis_path).resolve()): sha256(args.basis_path)}
    before = memory_snapshot()
    import torch
    torch.set_num_threads(1)
    payload = torch.load(args.basis_path, map_location="cpu", weights_only=True)
    basis = payload["obj"].detach().cpu().numpy().reshape(-1, 3).copy()
    if basis.shape != (1024, 3) or not np.isfinite(basis).all():
        raise ValueError("Expected existing finite 1024-point BPS basis")
    del payload
    hands, hand_info = load_validation_palms(args, input_hashes)
    after_hand_loading = memory_snapshot()
    output.mkdir(parents=True, exist_ok=False)
    all_rows, objects = [], []
    started = time.perf_counter()
    for name in args.objects:
        print(f"[CPU] {name}: reading existing mesh and BPS", flush=True)
        mesh_path, delta_path = root / "rest_object_geo" / (name + ".ply"), root / "rest_object_geo" / (name + ".npy")
        for path in (mesh_path, delta_path):
            input_hashes[str(path)] = sha256(path)
        vertices, faces = read_canonical_ply(mesh_path)
        deltas = np.load(delta_path, mmap_mode="r", allow_pickle=False)
        if deltas.shape != (1, 1024, 3) or not np.isfinite(deltas).all():
            raise ValueError(f"Invalid cached BPS deltas: {delta_path}")
        # Match dataset float32 addition before converting to diagnostic float64.
        anchors = (basis.astype(np.float32) + deltas[0].astype(np.float32)).astype(np.float64)
        descriptions, extent = make_queries(vertices, faces, hands[name], name, args.seed, args.surface_samples)
        q = np.asarray([row["point"] for row in descriptions])
        triangles = vertices[faces]
        qb, tc = args.query_batch_size, args.triangle_chunk_size
        match_distance, _, anchor_ids = nearest_points(anchors, vertices, qb, tc)
        if match_distance.max() > 1e-5 * extent:
            raise ValueError(f"{name}: restored BPS does not match canonical vertices")
        methods = {
            "bps_1024": lambda x: nearest_points(x, anchors, qb, tc)[:2],
            "mesh_vertices": lambda x: nearest_points(x, vertices, qb, tc)[:2],
            "mesh_triangles": lambda x: nearest_triangles(x, triangles, qb, tc),
        }
        computed = {}
        for method, fn in methods.items():
            print(f"[CPU] {name}: {method}, queries={len(q)}", flush=True)
            computed[method] = fn(q)
        reference_d, reference_cp = computed["mesh_triangles"]
        errors, angles, summaries = {}, {}, {}
        for method in ("bps_1024", "mesh_vertices"):
            errors[method], angles[method] = compare_queries(q, *computed[method], reference_d, reference_cp, extent)
        for stratum in sorted({row["stratum"] for row in descriptions}):
            mask = np.array([row["stratum"] == stratum for row in descriptions])
            summary = dict(count=int(mask.sum()), reference_distance_mm=quantiles(reference_d[mask] * 1000))
            for method in errors:
                near = mask & (reference_d <= .01)
                predicted_d = computed[method][0]
                summary[method] = dict(
                    absolute_error_mm=quantiles(abs(errors[method][mask]) * 1000),
                    absolute_error_over_h64=quantiles(abs(errors[method][mask]) / (extent / 63)),
                    pull_angle_deg=quantiles(angles[method][mask]),
                    direction_defined_count=int(np.isfinite(angles[method][mask]).sum()),
                    reference_within_10mm_count=int(near.sum()),
                    missed_reference_10mm_fraction=(float((predicted_d[near] > .01).mean()) if near.any() else None))
            summaries[stratum] = summary
        coverage_mask = np.array([r["stratum"] == "surface_coverage" for r in descriptions])
        coverage = {method: {str(mm): float((computed[method][0][coverage_mask] <= mm / 1000).mean())
                             for mm in (2, 5, 10)} for method in errors}
        timed_ids = np.random.default_rng(stable_seed(args.seed, name, "timing")).choice(
            len(q), size=min(args.timing_queries, len(q)), replace=False)
        timings = {method: benchmark(fn, q[timed_ids], args.timing_repeats) for method, fn in methods.items()}
        for k, description in enumerate(descriptions):
            row = dict(object=name, query_id=k, **{key: value for key, value in description.items() if key != "point"},
                       x=q[k, 0], y=q[k, 1], z=q[k, 2], h64_m=extent / 63,
                       triangle_distance_m=reference_d[k])
            for method in errors:
                angle = angles[method][k]
                row[method + "_distance_m"] = computed[method][0][k]
                row[method + "_error_m"] = errors[method][k]
                row[method + "_pull_angle_deg"] = float(angle) if np.isfinite(angle) else None
            for axis, value in zip("xyz", reference_cp[k]):
                row["triangle_closest_" + axis] = value
            all_rows.append(row)
        objects.append(dict(object=name,vertex_count=len(vertices),face_count=len(faces),
                            anchor_count=len(anchors), unique_anchor_vertices=len(set(anchor_ids.tolist())),
                            anchor_vertex_match_max_m=float(match_distance.max()), h64_m=extent / 63,
                            surface_coverage_fraction=coverage, strata=summaries, timing=timings))
        print(f"[CPU] {name}: complete; unique BPS vertices={objects[-1]['unique_anchor_vertices']}", flush=True)
    csv_path = output / "queries.csv"
    fields = sorted({key for row in all_rows for key in row})
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    summary = dict(status="DIAGNOSTIC_COMPLETE_NOT_GATE_PASS", args=vars(args),
                   hand_queries=hand_info, objects=objects, query_count=len(all_rows),
                   elapsed_seconds=time.perf_counter()-started, memory_before=before,
                   memory_after_hand_loading=after_hand_loading,
                   memory_after=memory_snapshot(), environment=dict(python=sys.version, numpy=np.__version__,
                   torch=torch.__version__, platform=platform.platform(), device="CPU"),
                   input_sha256=input_hashes, script_sha256=sha256(__file__),
                   limitations=["Unsigned geometry only; no SDF/cache was read or regenerated",
                   "No inside/outside, penetration, CUDA, autograd, or U1 efficacy validation",
                   "Mesh triangle union is the distance reference, not a certified physical solid",
                   "GT 22/23 proxies and annotation strata do not certify full-hand contact",
                   "Surface directions are undefined at zero distance; medial-axis ties can have multiple valid directions",
                   "Central-volume probes are not certified cavity labels; offset sides are not inside/outside labels",
                   "Sparse deterministic samples are a diagnostic, not a population-level effect estimate",
                   "Timing uses exhaustive NumPy search, not an optimized BPS-versus-BVH comparison"])
    with (output / "summary.json").open("x", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(dict(status=summary["status"], query_count=len(all_rows), output=str(output))), flush=True)


if __name__ == "__main__":
    main()
