#!/usr/bin/env python3
"""Audit geometry-only hand-object contact eligibility features."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("case_json", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def distance_features(hand_vertices, object_vertices):
    distances = cKDTree(object_vertices).query(hand_vertices)[0]
    return {
        "min_distance_m": float(distances.min()),
        "mean_distance_m": float(distances.mean()),
        "p05_distance_m": float(np.percentile(distances, 5)),
        "p10_distance_m": float(np.percentile(distances, 10)),
        "p25_distance_m": float(np.percentile(distances, 25)),
        "fraction_within_1cm": float(np.mean(distances <= 0.01)),
        "fraction_within_2cm": float(np.mean(distances <= 0.02)),
        "fraction_within_3cm": float(np.mean(distances <= 0.03)),
    }


def main():
    args = parse_args()
    cases = []
    rows = []

    for case_path in args.case_json:
        case = json.loads(case_path.read_text(encoding="utf-8"))
        candidate = np.load(case["candidate_npz"], allow_pickle=True)
        hand_vertices = np.asarray(
            candidate["pred_right_hand_verts"], dtype=np.float64
        )
        object_vertices = np.asarray(
            candidate["pred_object_verts"], dtype=np.float64
        )

        case_rows = []
        for frame, outcome in zip(case["frames"], case["rows"]):
            features = distance_features(
                hand_vertices[frame], object_vertices[frame]
            )
            contact_change = outcome["hand_contact_relative_change"]
            distance_change = outcome["distance_relative_change"]
            row = {
                "case_json": str(case_path),
                "sequence": case["sequence"],
                "object_name": case["object_name"],
                "frame": int(frame),
                **features,
                "contact_relative_change": contact_change,
                "distance_relative_change": distance_change,
                "contact_supported": (
                    contact_change is not None and contact_change > 0
                ),
                "distance_supported": distance_change < 0,
                "jointly_supported": (
                    contact_change is not None
                    and contact_change > 0
                    and distance_change < 0
                ),
            }
            rows.append(row)
            case_rows.append(row)

        cases.append(
            {
                "case_json": str(case_path),
                "sequence": case["sequence"],
                "object_name": case["object_name"],
                "frames": len(case_rows),
                "contact_supported_frames": int(
                    sum(row["contact_supported"] for row in case_rows)
                ),
                "distance_supported_frames": int(
                    sum(row["distance_supported"] for row in case_rows)
                ),
                "jointly_supported_frames": int(
                    sum(row["jointly_supported"] for row in case_rows)
                ),
                "feature_means": {
                    key: float(
                        np.mean([row[key] for row in case_rows])
                    )
                    for key in (
                        "min_distance_m",
                        "mean_distance_m",
                        "p05_distance_m",
                        "p10_distance_m",
                        "p25_distance_m",
                        "fraction_within_1cm",
                        "fraction_within_2cm",
                        "fraction_within_3cm",
                    )
                },
            }
        )

    result = {
        "case_count": len(cases),
        "frame_count": len(rows),
        "cases": cases,
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
