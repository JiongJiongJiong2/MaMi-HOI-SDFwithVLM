"""Convert predicted query JSONL into the existing material-audit CSV schema."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


OBJECTS = ("plasticbox", "trashcan", "smalltable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", default=list(OBJECTS))
    parser.add_argument(
        "--max_queries_per_object_hand",
        type=int,
        default=0,
        help="0 keeps all queries; otherwise sample deterministically.",
    )
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_csv.exists():
        raise FileExistsError(args.output_csv)
    objects = set(args.objects)
    rows = []
    with args.input_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("object") not in objects:
                continue
            row = {
                "object": record["object"],
                "stratum": "predicted_model",
                "x": float(record["canonical_x"]),
                "y": float(record["canonical_y"]),
                "z": float(record["canonical_z"]),
                "sequence": record["sequence"],
                "frame": int(record["frame"]),
                "joint": int(record["joint"]),
                "hand": record["hand"],
            }
            if not all(
                math.isfinite(row[key])
                for key in ("x", "y", "z")
            ):
                raise ValueError(f"Non-finite query: {record}")
            rows.append(row)
    if not rows:
        raise ValueError("No predicted queries matched the requested objects")
    if args.max_queries_per_object_hand < 0:
        raise ValueError("--max_queries_per_object_hand must be non-negative")
    if args.max_queries_per_object_hand:
        groups = defaultdict(list)
        for row in rows:
            groups[(row["object"], row["hand"])].append(row)
        sampled = []
        for key, group in sorted(groups.items()):
            group.sort(
                key=lambda row: hashlib.sha256(
                    ":".join(
                        [
                            str(args.seed),
                            row["object"],
                            row["hand"],
                            row["sequence"],
                            str(row["frame"]),
                            str(row["joint"]),
                        ]
                    ).encode("utf-8")
                ).digest()
            )
            sampled.extend(group[: args.max_queries_per_object_hand])
        rows = sampled
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "object",
                "stratum",
                "x",
                "y",
                "z",
                "sequence",
                "frame",
                "joint",
                "hand",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    counts = defaultdict(int)
    for row in rows:
        counts[(row["object"], row["hand"])] += 1
    print(json.dumps({
        "status": "PASS",
        "rows": len(rows),
        "counts": {
            f"{object_name}|{hand}": count
            for (object_name, hand), count in sorted(counts.items())
        },
    }))


if __name__ == "__main__":
    main()
