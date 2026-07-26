"""Measure canonical 64^3 SDF query error against the original 256^3 grids.

This script is read-only with respect to SDF data.  It queries both resolutions
at the same deterministic canonical points and writes per-object plus aggregate
metrics in world units.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from manip.model.sdf_utils import sample_object_sdf_at_points


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--samples_per_object", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--objects",
        nargs="*",
        default=None,
        help="Optional object-name subset; default audits every 64^3 cache.",
    )
    return parser.parse_args()


def load_source(data_root, object_name):
    source_folder = data_root / "rest_object_sdf_256_npy_files"
    sdf_path = source_folder / f"{object_name}.ply.npy"
    metadata_path = source_folder / f"{object_name}.ply.json"
    if not sdf_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing 256^3 source pair for {object_name}: "
            f"{sdf_path}, {metadata_path}"
        )
    sdf = np.load(sdf_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sdf.shape != (256, 256, 256):
        raise ValueError(f"{sdf_path} is not 256^3: {sdf.shape}")
    if not np.isfinite(sdf).all():
        raise ValueError(f"{sdf_path} contains NaN or Inf")
    return (
        torch.from_numpy(np.ascontiguousarray(sdf)).float()[None, None],
        torch.tensor(metadata["centroid"], dtype=torch.float32)[None],
        torch.tensor(metadata["extents"], dtype=torch.float32)[None],
    )


def load_cache(data_root, object_name):
    cache_path = data_root / "object_sdf_64" / f"{object_name}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing 64^3 cache: {cache_path}")
    payload = torch.load(cache_path, map_location="cpu")
    required = {"sdf_grid", "centroid", "extents"}
    missing = required.difference(payload)
    if missing:
        raise KeyError(f"{cache_path} is missing {sorted(missing)}")
    sdf = payload["sdf_grid"].float()
    if sdf.shape != (1, 64, 64, 64):
        raise ValueError(f"{cache_path} grid must be [1,64,64,64], got {sdf.shape}")
    if not torch.isfinite(sdf).all():
        raise ValueError(f"{cache_path} contains NaN or Inf")
    return (
        sdf[None],
        payload["centroid"].float().reshape(1, 3),
        payload["extents"].float().reshape(1, 3),
        payload,
    )


def audit_object(data_root, object_name, samples, batch_size, generator):
    sdf256, centroid256, extents256 = load_source(data_root, object_name)
    sdf64, centroid64, extents64, payload = load_cache(data_root, object_name)
    if not torch.allclose(centroid64, centroid256, atol=1e-6, rtol=0):
        raise ValueError(f"{object_name}: 64^3 and 256^3 centroids differ")
    if not torch.allclose(extents64, extents256, atol=1e-6, rtol=0):
        raise ValueError(f"{object_name}: 64^3 and 256^3 extents differ")
    if (extents64 <= 0).any():
        raise ValueError(f"{object_name}: non-positive extents")

    max_extent = extents64.max().item()
    errors = []
    agreements = 0
    total = 0
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        normalized = torch.rand(
            1, count, 3, generator=generator, dtype=torch.float32
        ) * 2.0 - 1.0
        points = centroid64[:, None, :] + normalized * (max_extent / 2.0)
        values64 = sample_object_sdf_at_points(
            sdf64, points, centroid64, extents64
        ).squeeze(0).squeeze(-1)
        values256 = sample_object_sdf_at_points(
            sdf256, points, centroid256, extents256
        ).squeeze(0).squeeze(-1)
        error = (values64 - values256).abs()
        errors.append(error)
        agreements += int(
            (torch.sign(values64) == torch.sign(values256)).sum().item()
        )
        total += count

    all_errors = torch.cat(errors)
    voxel_scale = max_extent / 63.0
    return {
        "object_name": object_name,
        "samples": total,
        "mae_m": float(all_errors.mean().item()),
        "p95_error_m": float(torch.quantile(all_errors, 0.95).item()),
        "sign_agreement": agreements / total,
        "voxel_scale_64_m": voxel_scale,
        "max_extent_m": max_extent,
        "cache_resolution_metadata": payload.get("resolution"),
        "source_resolution_metadata": payload.get("source_resolution"),
    }


def main():
    args = parse_args()
    if args.samples_per_object <= 0 or args.batch_size <= 0:
        raise ValueError("sample counts must be positive")
    data_root = Path(args.data_root_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    cache_folder = data_root / "object_sdf_64"
    object_names = (
        sorted(args.objects)
        if args.objects
        else sorted(path.stem for path in cache_folder.glob("*.pt"))
    )
    if not object_names:
        raise FileNotFoundError(f"No 64^3 caches found in {cache_folder}")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    rows = [
        audit_object(
            data_root,
            object_name,
            args.samples_per_object,
            args.batch_size,
            generator,
        )
        for object_name in object_names
    ]
    csv_path = output_folder / "sdf_64_vs_256_error.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_samples = sum(row["samples"] for row in rows)
    summary = {
        "seed": args.seed,
        "samples_per_object": args.samples_per_object,
        "object_count": len(rows),
        "aggregate_weighted_mae_m": sum(
            row["mae_m"] * row["samples"] for row in rows
        ) / total_samples,
        "aggregate_weighted_sign_agreement": sum(
            row["sign_agreement"] * row["samples"] for row in rows
        ) / total_samples,
        "aggregate_p95_of_object_p95_m": float(
            np.quantile([row["p95_error_m"] for row in rows], 0.95)
        ),
        "max_object_p95_error_m": max(row["p95_error_m"] for row in rows),
        "min_sign_agreement": min(row["sign_agreement"] for row in rows),
        "per_object_csv": str(csv_path),
    }
    summary_path = output_folder / "sdf_64_vs_256_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
