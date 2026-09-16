#!/usr/bin/env python3
"""Create a deterministic validation subset for candidate-selection smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    source_path = Path(args.input)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validation = list(source["validation_sequences"])
    if args.offset < 0:
        raise ValueError("offset must be non-negative")
    end = args.offset + args.count
    if len(validation) < end:
        raise ValueError(
            f"Requested sequences [{args.offset}, {end}) but only "
            f"{len(validation)} exist"
        )
    selected = sorted(validation)[args.offset : end]
    output = {
        key: value
        for key, value in source.items()
        if key not in {"validation_sequences", "test_sequences"}
    }
    output["validation_sequences"] = selected
    output["test_sequences"] = []
    output["contact_selection_subset"] = {
        "source_manifest": str(source_path),
        "source_sha256": sha256(source_path),
        "count": args.count,
        "offset": args.offset,
        "selection": (
            "ascending sorted validation sequences "
            f"[{args.offset}, {end})"
        ),
        "seed": args.seed,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "count": len(selected),
                "sequences": selected,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
