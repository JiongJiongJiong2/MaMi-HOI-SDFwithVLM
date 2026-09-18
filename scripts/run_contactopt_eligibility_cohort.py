#!/usr/bin/env python3
"""Run the frozen Phase 9 gate over a deterministic ContactOpt cohort."""

import argparse
import json
import math
import os
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--case-script", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--sequence-db", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--min-frame-separation", type=int, default=5)
    parser.add_argument("--contact-p10-threshold", type=float, default=0.02)
    parser.add_argument("--min-eligible-frames", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=250)
    return parser.parse_args()


def finite(value):
    return value is not None


def summarize_case(path):
    result = json.loads(path.read_text(encoding="utf-8"))
    gate = result.get("gate", {})
    rows = result.get("rows", [])
    invoked = bool(rows)
    summary = {
        "path": str(path),
        "sequence": result.get("sequence"),
        "object_name": result.get("object_name"),
        "selected_frame_count": len(result.get("selected_frames", [])),
        "eligible_frame_count": len(result.get("eligible_frames", [])),
        "gate_passed": bool(gate.get("passed", False)),
        "gate": gate,
        "invoked": invoked,
    }
    if invoked:
        summary.update(
            {
                "contact_improved_frames": result[
                    "contact_improved_frames"
                ],
                "distance_improved_frames": result[
                    "distance_improved_frames"
                ],
                "mean_hand_contact_relative_change": result[
                    "mean_hand_contact_relative_change"
                ],
                "mean_distance_relative_change": result[
                    "mean_distance_relative_change"
                ],
                "max_wrist_drift_m": result["max_wrist_drift_m"],
                "max_object_drift_m": result["max_object_drift_m"],
            }
        )
    return summary


def main():
    args = parse_args()
    manifest = json.loads(
        args.manifest_json.read_text(encoding="utf-8")
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    case_dir = args.case_dir or args.output_dir

    if args.summary_only:
        case_results = [
            summarize_case(case_dir / f"{candidate['sequence']}.json")
            for candidate in manifest["selected"]
        ]
        write_summary(args, case_results)
        return

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)

    case_results = []
    for index, candidate in enumerate(manifest["selected"]):
        sequence = candidate["sequence"]
        output_json = args.output_dir / f"{sequence}.json"
        output_tag = f"phase10_{sequence}"
        command = [
            str(args.python),
            str(args.case_script),
            "--candidate-npz",
            candidate["path"],
            "--sequence-db",
            str(args.sequence_db),
            "--num-frames",
            str(args.num_frames),
            "--min-frame-separation",
            str(args.min_frame_separation),
            "--contact-p10-threshold",
            str(args.contact_p10_threshold),
            "--min-eligible-frames",
            str(args.min_eligible_frames),
            "--n-iter",
            str(args.n_iter),
            "--output-tag",
            output_tag,
            "--output-json",
            str(output_json),
        ]
        print(
            f"[{index + 1}/{len(manifest['selected'])}] "
            f"{sequence}",
            flush=True,
        )
        subprocess.run(
            command,
            cwd=args.workdir,
            env=env,
            check=True,
        )
        case_results.append(summarize_case(output_json))

    write_summary(args, case_results)


def write_summary(args, case_results):

    invoked = [result for result in case_results if result["invoked"]]
    eligible = [
        result
        for result in case_results
        if result["eligible_frame_count"]
        >= args.min_eligible_frames
    ]
    aggregate = {
        "manifest_json": str(args.manifest_json),
        "selected_case_count": len(case_results),
        "invoked_case_count": len(invoked),
        "eligible_case_count": len(eligible),
        "passed_case_count": sum(
            result["gate_passed"] for result in case_results
        ),
        "eligible_contact_success_count": sum(
            result.get("contact_improved_frames", 0)
            >= max(
                3,
                math.ceil(0.7 * result["eligible_frame_count"]),
            )
            for result in eligible
        ),
        "max_wrist_drift_m": max(
            (
                result["max_wrist_drift_m"]
                for result in invoked
                if finite(result["max_wrist_drift_m"])
            ),
            default=None,
        ),
        "max_object_drift_m": max(
            (
                result["max_object_drift_m"]
                for result in invoked
                if finite(result["max_object_drift_m"])
            ),
            default=None,
        ),
        "cases": case_results,
    }
    aggregate["gate"] = {
        "six_cases_selected": aggregate["selected_case_count"] >= 6,
        "four_cases_eligible": aggregate["eligible_case_count"] >= 4,
        "frozen_coordinates": (
            aggregate["max_wrist_drift_m"] is not None
            and aggregate["max_wrist_drift_m"] <= 0.00001
            and aggregate["max_object_drift_m"] is not None
            and aggregate["max_object_drift_m"] <= 0.000001
        ),
        "three_eligible_contact_successes": (
            aggregate["eligible_contact_success_count"] >= 3
        ),
        "no_large_distance_regression": all(
            result.get("mean_distance_relative_change", 0.0) <= 0.10
            for result in invoked
        ),
    }
    aggregate["gate"]["passed"] = all(aggregate["gate"].values())

    aggregate_path = args.output_dir / "cohort_summary.json"
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
