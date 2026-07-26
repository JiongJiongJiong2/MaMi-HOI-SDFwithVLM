"""Prepare one compact canonical SDF volume per training object.

The source SDF files are already part of the CHOIS/MaMi-HOI processed data.
This script only downsamples them and preserves their evaluator metadata. It
never uses motion or hand trajectories, so the generated files are safe for
dynamic training-time queries.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_root_folder",
        required=True,
        help="Processed-data root containing rest_object_sdf_256_npy_files.",
    )
    parser.add_argument(
        "--output_folder",
        default="",
        help="Defaults to <data_root_folder>/object_sdf_64.",
    )
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_source_pairs(source_folder):
    pairs = []
    for sdf_path in sorted(source_folder.glob("*.ply.npy")):
        object_name = sdf_path.name[: -len(".ply.npy")]
        metadata_path = source_folder / f"{object_name}.ply.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing SDF metadata for {sdf_path}: {metadata_path}")
        pairs.append((object_name, sdf_path, metadata_path))
    if not pairs:
        raise FileNotFoundError(f"No '*.ply.npy' SDFs found in {source_folder}")
    return pairs


def downsample_sdf(sdf, resolution):
    if sdf.ndim != 3:
        raise ValueError(f"Expected a 3D SDF grid, got shape {sdf.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(sdf)).float()[None, None]
    if tensor.shape[-3:] == (resolution, resolution, resolution):
        return tensor.squeeze(0).contiguous()
    return F.interpolate(
        tensor,
        size=(resolution, resolution, resolution),
        mode="trilinear",
        align_corners=True,
    ).squeeze(0).contiguous()


def main():
    args = parse_args()
    if args.resolution < 8:
        raise ValueError("resolution must be at least 8")

    data_root = Path(args.data_root_folder)
    source_folder = data_root / "rest_object_sdf_256_npy_files"
    output_folder = Path(args.output_folder) if args.output_folder else data_root / f"object_sdf_{args.resolution}"
    output_folder.mkdir(parents=True, exist_ok=True)

    written = 0
    for object_name, sdf_path, metadata_path in resolve_source_pairs(source_folder):
        output_path = output_folder / f"{object_name}.pt"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path}")
            continue

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "centroid" not in metadata or "extents" not in metadata:
            raise KeyError(f"{metadata_path} must contain centroid and extents")

        source_sdf = np.load(sdf_path)
        if source_sdf.shape != (256, 256, 256):
            raise ValueError(
                f"{sdf_path} must be a 256^3 source SDF, got {source_sdf.shape}"
            )
        if not np.isfinite(source_sdf).all():
            raise ValueError(f"{sdf_path} contains NaN or Inf")
        centroid = torch.tensor(metadata["centroid"], dtype=torch.float32)
        extents = torch.tensor(metadata["extents"], dtype=torch.float32)
        if centroid.shape != (3,) or extents.shape != (3,):
            raise ValueError(
                f"{metadata_path} centroid/extents must both have length 3"
            )
        if not torch.isfinite(centroid).all() or not torch.isfinite(extents).all():
            raise ValueError(f"{metadata_path} centroid/extents contains NaN or Inf")
        if (extents <= 0).any():
            raise ValueError(f"{metadata_path} extents must be strictly positive")

        sdf_grid = downsample_sdf(source_sdf, args.resolution)
        payload = {
            "sdf_grid": sdf_grid,  # [1, R, R, R], normalized SDF samples
            "centroid": centroid,
            "extents": extents,
            "source_sdf": str(sdf_path.name),
            "source_resolution": 256,
            "resolution": args.resolution,
            "axis_order": "D(z),H(y),W(x)",
            "centroid_definition": "canonical mesh bounding-box centre",
            "extents_definition": "canonical mesh full side lengths",
            "sdf_sign_convention": "negative-inside, positive-outside",
            "sdf_value_units": "normalized by max(extents)/2",
            "align_corners": True,
        }
        torch.save(payload, output_path)
        written += 1
        print(
            f"[write] {object_name}: {tuple(sdf_grid.shape)}, "
            f"extent={payload['extents'].tolist()} -> {output_path}"
        )

    print(f"Finished. Wrote {written} object SDF files to {output_folder}")


if __name__ == "__main__":
    main()
