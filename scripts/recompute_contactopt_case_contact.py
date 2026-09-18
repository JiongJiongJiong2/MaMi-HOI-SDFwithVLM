#!/usr/bin/env python3
"""Recompute ContactOpt case contact metrics from an existing result pkl."""

import argparse
import json
from pathlib import Path

from run_mami_contactopt_case import (
    aggregate_rows,
    build_gate,
    evaluate_runs,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", type=Path, required=True)
    parser.add_argument("--optimized-pkl", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    result = json.loads(args.case_json.read_text(encoding="utf-8"))
    optimized_pkl = args.optimized_pkl or Path(result["optimized_pkl"])
    rows = evaluate_runs(optimized_pkl)
    aggregate = aggregate_rows(rows, result["alignment"])
    eligible_frames = result["eligible_frames"]
    min_eligible_frames = int(result["min_eligible_frames"])
    gate, required_improved_frames = build_gate(
        aggregate,
        len(eligible_frames),
        min_eligible_frames,
    )

    result.update(aggregate)
    result.update(
        {
            "rows": rows,
            "required_improved_frames": required_improved_frames,
            "gate": gate,
            "contact_metric_policy": (
                "nonfinite capsule contact values are treated as zero; "
                "finite fractions are retained per frame"
            ),
        }
    )

    output_json = args.output_json or args.case_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "sequence": result["sequence"],
                "object_name": result["object_name"],
                "contact_improved_frames": result[
                    "contact_improved_frames"
                ],
                "eligible_frame_count": len(eligible_frames),
                "required_improved_frames": required_improved_frames,
                "mean_hand_contact_relative_change": result[
                    "mean_hand_contact_relative_change"
                ],
                "gate": result["gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
