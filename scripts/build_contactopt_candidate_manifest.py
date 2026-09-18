#!/usr/bin/env python3
"""Select a deterministic saved-vertex MaMi cohort for ContactOpt audits."""

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "pred_right_hand_verts",
    "pred_object_verts",
    "object_faces",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, nargs="+", required=True)
    parser.add_argument("--object", dest="objects", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-size-mb", type=float, default=10.0)
    return parser.parse_args()


def infer_object_name(path, archive):
    if "object_name" in archive.files:
        value = archive["object_name"]
        if value.shape == ():
            return str(value.item())
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer object name from {path}")
    return parts[1]


def main():
    args = parse_args()
    minimum_size = int(args.min_size_mb * 1024 * 1024)
    candidates = []

    for root in args.root:
        for path in sorted(root.rglob("*.npz")):
            if path.stat().st_size < minimum_size:
                continue
            with np.load(path, allow_pickle=True) as archive:
                if not REQUIRED_KEYS.issubset(archive.files):
                    continue
                object_name = infer_object_name(path, archive)
                sequence_name = (
                    str(archive["seq_name"].item())
                    if "seq_name" in archive.files
                    else path.stem
                )
            candidates.append(
                {
                    "path": str(path.resolve()),
                    "sequence": sequence_name,
                    "object_name": object_name,
                    "size_bytes": path.stat().st_size,
                }
            )

    by_object = {}
    for candidate in candidates:
        by_object.setdefault(candidate["object_name"], []).append(
            candidate
        )

    selected = []
    missing = []
    for object_name in args.objects:
        matches = sorted(
            by_object.get(object_name, []),
            key=lambda item: item["path"],
        )
        if not matches:
            missing.append(object_name)
            continue
        selected.append(matches[0])

    result = {
        "objects_requested": args.objects,
        "saved_vertex_candidate_count": len(candidates),
        "available_objects": sorted(by_object),
        "missing_objects": missing,
        "selected": selected,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
