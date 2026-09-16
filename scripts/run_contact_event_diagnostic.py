"""Generate the CEWM-01 diagnostic from frozen U0-FT event arrays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from manip.contact.event_schema import (
    CONTACT_EVENT_SCHEMA_VERSION,
    HANDS,
    ContactEventSchema,
    aggregate_sequence_hand_records,
    analyze_hand_sequence,
)


OUTPUT_FILES = (
    "per_sequence.csv",
    "events.jsonl",
    "summary.json",
    "provenance.json",
    "report.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen_events_npz", type=Path, required=True)
    parser.add_argument("--v3_metrics_jsonl", type=Path, required=True)
    parser.add_argument("--frozen_queries_jsonl", type=Path)
    parser.add_argument("--location_json", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--arm_id", default="U0_FT_TC")
    parser.add_argument("--schema_json", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_schema(path: Path | None) -> ContactEventSchema:
    if path is None:
        return ContactEventSchema()
    return ContactEventSchema.from_dict(
        json.loads(path.read_text(encoding="utf-8"))
    )


def load_frozen_arrays(path: Path) -> dict:
    arrays = np.load(path, allow_pickle=False)
    required = {
        "sequence_names",
        "object_names",
        "gt_contact",
        "pred_contact",
        "distance_m",
    }
    if set(arrays.files) != required:
        raise ValueError(
            f"Frozen arrays keys differ: expected={sorted(required)}, "
            f"observed={sorted(arrays.files)}"
        )
    sequence_names = arrays["sequence_names"].astype(str)
    object_names = arrays["object_names"].astype(str)
    gt_contact = arrays["gt_contact"].astype(bool)
    pred_contact = arrays["pred_contact"].astype(bool)
    distance_m = arrays["distance_m"].astype(np.float64)
    if gt_contact.ndim != 3 or gt_contact.shape[2] != len(HANDS):
        raise ValueError(f"gt_contact must be [S, T, 2], got {gt_contact.shape}")
    if pred_contact.shape != gt_contact.shape:
        raise ValueError("pred_contact must match gt_contact")
    if distance_m.shape != gt_contact.shape:
        raise ValueError("distance_m must match gt_contact")
    if sequence_names.shape != (gt_contact.shape[0],):
        raise ValueError("sequence_names does not match the first dimension")
    if object_names.shape != sequence_names.shape:
        raise ValueError("object_names does not match sequence_names")
    if len(set(sequence_names.tolist())) != len(sequence_names):
        raise ValueError("sequence_names contains duplicates")
    return {
        "sequence_names": sequence_names,
        "object_names": object_names,
        "gt_contact": gt_contact,
        "pred_contact": pred_contact,
        "distance_m": distance_m,
    }


def load_v3_metrics(path: Path) -> dict[str, dict]:
    result = {}
    for record in load_jsonl(path):
        raw_sequence = str(record["sequence"]).split("_sidx_", 1)[0]
        object_suffix = f"_{record['object']}"
        sequence = (
            raw_sequence[: -len(object_suffix)]
            if raw_sequence.endswith(object_suffix)
            else raw_sequence
        )
        if sequence in result:
            raise ValueError(f"Duplicate v3 sequence: {sequence}")
        result[sequence] = record
    if not result:
        raise ValueError("No v3 metrics found")
    return result


def validate_v3_metrics(records: dict[str, dict], sequences: list[str]) -> None:
    observed = set(records)
    expected = set(sequences)
    if observed != expected:
        raise ValueError(
            f"v3 sequence mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    for sequence in sequences:
        record = records[sequence]
        if record.get("version") != "hand_contact_common_v3":
            raise ValueError(f"{sequence}: unexpected v3 version")
        if abs(float(record["contact_threshold_m"]) - 0.05) > 1e-12:
            raise ValueError(f"{sequence}: v3 threshold is not 0.05 m")
        if int(record["valid_frame_count"]) <= 0:
            raise ValueError(f"{sequence}: invalid frame count")


def v3_count_mismatches(
    records: dict[str, dict],
    analysis: dict[str, dict[str, dict]],
) -> tuple[list[dict], set[str]]:
    mismatches = []
    exact_sequences = set()
    for sequence in sorted(analysis):
        exact = True
        for hand in HANDS:
            expected = records[sequence]
            observed = analysis[sequence][hand]
            for key in ("tp", "fp", "tn", "fn"):
                if int(observed[key]) != int(expected[f"{hand}_{key}"]):
                    exact = False
                    mismatches.append(
                        {
                            "sequence": sequence,
                            "hand": hand,
                            "metric": key,
                            "v3": int(expected[f"{hand}_{key}"]),
                            "local": int(observed[key]),
                        }
                    )
            for local_key, metric_key in (
                ("gt_contact_frames", "gt_contact_frame_count"),
                ("pred_contact_frames", "pred_contact_frame_count"),
            ):
                if int(observed[local_key]) != int(
                    expected[f"{hand}_{metric_key}"]
                ):
                    exact = False
                    mismatches.append(
                        {
                            "sequence": sequence,
                            "hand": hand,
                            "metric": local_key,
                            "v3": int(expected[f"{hand}_{metric_key}"]),
                            "local": int(observed[local_key]),
                        }
                    )
        if exact:
            exact_sequences.add(sequence)
    return mismatches, exact_sequences


def f1_from_counts(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else None


def v3_headline(records: dict[str, dict]) -> dict:
    result = {}
    for hand in HANDS:
        totals = {
            key: int(sum(int(record[f"{hand}_{key}"]) for record in records.values()))
            for key in ("tp", "fp", "tn", "fn")
        }
        result[hand] = {
            **totals,
            "gt_contact_frames": totals["tp"] + totals["fn"],
            "pred_contact_frames": totals["tp"] + totals["fp"],
            "gt_contact_pred_noncontact": totals["fn"],
            "gt_noncontact_pred_contact": totals["fp"],
            "f1": f1_from_counts(totals["tp"], totals["fp"], totals["fn"]),
        }
    return result


def macro_f1_from_v3(records: dict[str, dict], hand: str) -> float | None:
    values = [
        float(record[f"{hand}_f1"])
        for record in records.values()
        if record.get(f"{hand}_f1") is not None
    ]
    return float(np.mean(values)) if values else None


def event_rows(
    arrays: dict,
    analysis: dict[str, dict[str, dict]],
    exact_sequences: set[str],
) -> list[dict]:
    rows = []
    for sequence_index, sequence in enumerate(arrays["sequence_names"].tolist()):
        object_name = str(arrays["object_names"][sequence_index])
        for hand in HANDS:
            record = analysis[sequence][hand]
            scope = "v3_exact" if sequence in exact_sequences else "bounded_local"
            for gt_index, interval in enumerate(record["gt_intervals"]):
                match = record["interval_matches"][gt_index]
                rows.append(
                    {
                        "event_type": "gt_contact_interval",
                        "sequence": sequence,
                        "object": object_name,
                        "hand": hand,
                        "interval_index": gt_index,
                        "start_frame": int(interval[0]),
                        "end_frame": int(interval[1]),
                        "match_status": match["match_status"],
                        "matched_interval_index": match["pred_interval_index"],
                        "onset_error_frames": match["onset_error_frames"],
                        "release_delay_frames": match["release_delay_frames"],
                        "continued_after_release_frames": (
                            match["continued_after_release_frames"]
                        ),
                        "location": None,
                        "location_status": record["contact_location_error"],
                        "scope": scope,
                    }
                )
            matched_prediction_indices = {
                int(match["pred_interval_index"])
                for match in record["interval_matches"]
                if match["pred_interval_index"] is not None
            }
            for pred_index, interval in enumerate(record["pred_intervals"]):
                rows.append(
                    {
                        "event_type": "pred_contact_interval",
                        "sequence": sequence,
                        "object": object_name,
                        "hand": hand,
                        "interval_index": pred_index,
                        "start_frame": int(interval[0]),
                        "end_frame": int(interval[1]),
                        "match_status": (
                            "matched"
                            if pred_index in matched_prediction_indices
                            else "unmatched_prediction"
                        ),
                        "matched_interval_index": None,
                        "onset_error_frames": None,
                        "release_delay_frames": None,
                        "continued_after_release_frames": None,
                        "location": None,
                        "location_status": record["contact_location_error"],
                        "scope": scope,
                    }
                )
    return rows


def per_sequence_rows(
    arrays: dict,
    v3_records: dict[str, dict],
    analysis: dict[str, dict[str, dict]],
    exact_sequences: set[str],
) -> list[dict]:
    rows = []
    for sequence_index, sequence in enumerate(arrays["sequence_names"].tolist()):
        object_name = str(arrays["object_names"][sequence_index])
        for hand in HANDS:
            record = analysis[sequence][hand]
            rows.append(
                {
                    "record_type": "sequence_hand_summary",
                    "sequence": sequence,
                    "object": object_name,
                    "hand": hand,
                    "gt_interval_index": "",
                    "gt_start": "",
                    "gt_end": "",
                    "pred_interval_index": "",
                    "pred_start": "",
                    "pred_end": "",
                    "match_status": "",
                    "onset_error_frames": "",
                    "release_delay_frames": "",
                    "continued_after_release_frames": "",
                    "v3_tp": int(v3_records[sequence][f"{hand}_tp"]),
                    "v3_fp": int(v3_records[sequence][f"{hand}_fp"]),
                    "v3_tn": int(v3_records[sequence][f"{hand}_tn"]),
                    "v3_fn": int(v3_records[sequence][f"{hand}_fn"]),
                    "local_tp": record["tp"],
                    "local_fp": record["fp"],
                    "local_tn": record["tn"],
                    "local_fn": record["fn"],
                    "local_counts_match_v3": sequence in exact_sequences,
                    "timing_scope": (
                        "v3_exact"
                        if sequence in exact_sequences
                        else "bounded_local"
                    ),
                }
            )
            for gt_index, interval in enumerate(record["gt_intervals"]):
                match = record["interval_matches"][gt_index]
                rows.append(
                    {
                        "record_type": "gt_interval",
                        "sequence": sequence,
                        "object": object_name,
                        "hand": hand,
                        "gt_interval_index": gt_index,
                        "gt_start": interval[0],
                        "gt_end": interval[1],
                        "pred_interval_index": (
                            ""
                            if match["pred_interval_index"] is None
                            else match["pred_interval_index"]
                        ),
                        "pred_start": (
                            ""
                            if match["pred_start"] is None
                            else match["pred_start"]
                        ),
                        "pred_end": (
                            ""
                            if match["pred_end"] is None
                            else match["pred_end"]
                        ),
                        "match_status": match["match_status"],
                        "onset_error_frames": (
                            ""
                            if match["onset_error_frames"] is None
                            else match["onset_error_frames"]
                        ),
                        "release_delay_frames": (
                            ""
                            if match["release_delay_frames"] is None
                            else match["release_delay_frames"]
                        ),
                        "continued_after_release_frames": (
                            ""
                            if match["continued_after_release_frames"] is None
                            else match["continued_after_release_frames"]
                        ),
                        "v3_tp": "",
                        "v3_fp": "",
                        "v3_tn": "",
                        "v3_fn": "",
                        "local_tp": "",
                        "local_fp": "",
                        "local_tn": "",
                        "local_fn": "",
                        "local_counts_match_v3": sequence in exact_sequences,
                        "timing_scope": (
                            "v3_exact"
                            if sequence in exact_sequences
                            else "bounded_local"
                        ),
                    }
                )
            matched_prediction_indices = {
                int(match["pred_interval_index"])
                for match in record["interval_matches"]
                if match["pred_interval_index"] is not None
            }
            for pred_index, interval in enumerate(record["pred_intervals"]):
                if pred_index in matched_prediction_indices:
                    continue
                rows.append(
                    {
                        "record_type": "unmatched_pred_interval",
                        "sequence": sequence,
                        "object": object_name,
                        "hand": hand,
                        "gt_interval_index": "",
                        "gt_start": "",
                        "gt_end": "",
                        "pred_interval_index": pred_index,
                        "pred_start": interval[0],
                        "pred_end": interval[1],
                        "match_status": "unmatched_prediction",
                        "onset_error_frames": "",
                        "release_delay_frames": "",
                        "continued_after_release_frames": "",
                        "v3_tp": "",
                        "v3_fp": "",
                        "v3_tn": "",
                        "v3_fn": "",
                        "local_tp": "",
                        "local_fp": "",
                        "local_tn": "",
                        "local_fn": "",
                        "local_counts_match_v3": sequence in exact_sequences,
                        "timing_scope": (
                            "v3_exact"
                            if sequence in exact_sequences
                            else "bounded_local"
                        ),
                    }
                )
    return rows


def select_exact_analysis(
    analysis: dict[str, dict[str, dict]],
    sequences: list[str],
    exact_sequences: set[str],
) -> list[dict]:
    return [
        {"sequence": sequence, "hand": hand, **analysis[sequence][hand]}
        for sequence in sequences
        if sequence in exact_sequences
        for hand in HANDS
    ]


def load_location_summary(path: Path | None) -> dict:
    if path is None:
        return {
            "status": "NOT_PROVIDED",
            "definition": None,
        }
    record = json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "BOUNDED_EXISTING_BPR_OVERLAP_DIAGNOSTIC",
        "sample_count": int(record["sample_count"]),
        "sequence_count": int(record["sequence_count"]),
        "max_frame_gap": int(record["max_frame_gap"]),
        "by_hand": record["by_hand"],
        "definition": record["definition"],
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def write_report(path: Path, summary: dict, mismatches: list[dict]) -> None:
    headline = summary["headline_v3_all_sequences"]
    timing = summary["local_timing"]["v3_exact_subset"]
    lines = [
        "# CEWM-01 Contact-Event Diagnostic",
        "",
        f"Arm: `{summary['scope']['arm_id']}`. "
        f"Split: `{summary['scope']['split']}`. "
        "No model training or new sampling was performed.",
        "",
        "## Event presence",
        "",
        "| Hand | TP | FP | FN | F1 | GT-contact / pred-noncontact |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for hand in HANDS:
        row = headline[hand]
        lines.append(
            f"| {hand} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{row['f1']:.6f} | {row['gt_contact_pred_noncontact']} |"
        )
    lines.extend(
        [
            "",
            "## Timing scope",
            "",
            f"Local per-frame counts exactly reproduce v3 on "
            f"{summary['local_timing']['v3_exact_sequence_count']} of "
            f"{summary['scope']['sequence_count']} sequences. Timing results "
            "below are restricted to that intersection.",
            "",
            "| Hand | GT intervals | Matched | Missed | Continued after release | Mean positive release delay |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for hand in HANDS:
        row = timing[hand]["events"]
        macro = timing[hand]["sequence_macro"]
        lines.append(
            f"| {hand} | {row['gt_interval_count']} | "
            f"{row['matched_gt_interval_count']} | "
            f"{row['missed_gt_interval_count']} | "
            f"{row['continued_after_release_interval_count']} | "
            f"{macro['mean_positive_release_delay_frames']} |"
        )
    location = summary["location_diagnostic"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Surface-retrieval mainline: `{summary['decision']['surface_retrieval_mainline']}`.",
            f"- CEWM-02 readiness: `{summary['decision']['cewm_02_readiness']}`.",
            "- Contact location error and slip remain a bounded BPR-overlap "
            f"diagnostic (`{location['status']}`), not a full-validation claim.",
            "",
            "## Provenance limitation",
            "",
            f"{len(mismatches)} local count mismatches remain against the "
            "saved v3 metrics. The event-presence headline uses v3 metrics, "
            "while timing and event matching use only the exact-recompute "
            "subset. CEWM-02 must not start until this mesh/data provenance "
            "gap is repaired or replaced by a server-exact per-frame export.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def git_metadata(repo_root: Path) -> dict:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        status = run("status", "--short")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": None, "branch": None, "dirty": None}


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {args.output_dir}"
        )
    schema = load_schema(args.schema_json)
    arrays = load_frozen_arrays(args.frozen_events_npz)
    v3_records = load_v3_metrics(args.v3_metrics_jsonl)
    sequences = arrays["sequence_names"].tolist()
    validate_v3_metrics(v3_records, sequences)
    if arrays["gt_contact"].shape[1] != int(
        next(iter(v3_records.values()))["valid_frame_count"]
    ):
        raise ValueError("Frozen frame count does not match the v3 metrics")

    analysis = {}
    for sequence_index, sequence in enumerate(sequences):
        analysis[sequence] = {}
        for hand in HANDS:
            hand_index = 0 if hand == "left" else 1
            analysis[sequence][hand] = analyze_hand_sequence(
                arrays["gt_contact"][sequence_index, :, hand_index],
                arrays["pred_contact"][sequence_index, :, hand_index],
                arrays["distance_m"][sequence_index, :, hand_index],
                schema,
            )

    mismatches, exact_sequences = v3_count_mismatches(v3_records, analysis)
    exact_records = select_exact_analysis(
        analysis,
        sequences,
        exact_sequences,
    )
    local_timing_exact = aggregate_sequence_hand_records(exact_records)
    local_timing_all = aggregate_sequence_hand_records(
        [
            {"sequence": sequence, "hand": hand, **analysis[sequence][hand]}
            for sequence in sequences
            for hand in HANDS
        ]
    )
    headline = v3_headline(v3_records)
    location = load_location_summary(args.location_json)

    summary = {
        "schema": schema.to_dict(),
        "scope": {
            "experiment": "CEWM-01",
            "arm_id": args.arm_id,
            "split": "validation",
            "sequence_count": len(sequences),
            "frame_count": int(arrays["gt_contact"].shape[1]),
            "hand_count": len(HANDS),
            "training_performed": False,
            "test_split_used": False,
        },
        "headline_v3_all_sequences": headline,
        "sequence_macro_f1_v3_all_sequences": {
            hand: macro_f1_from_v3(v3_records, hand) for hand in HANDS
        },
        "local_timing": {
            "v3_exact_sequence_count": len(exact_sequences),
            "v3_exact_sequences": sorted(exact_sequences),
            "count_mismatch_count": len(mismatches),
            "v3_exact_subset": local_timing_exact,
            "bounded_all_sequences": local_timing_all,
        },
        "location_diagnostic": location,
        "decision": {
            "surface_retrieval_mainline": "paused",
            "cewm_02_readiness": (
                "BLOCKED_BY_FROZEN_MESH_PROVENANCE"
                if mismatches
                else "READY_FOR_PROTOCOL_FREEZE"
            ),
            "event_failure_observation": (
                "GT-contact/pred-noncontact is the largest exact v3 error "
                "category for both hands."
            ),
        },
    }

    args.output_dir.mkdir(parents=True)
    csv_path = args.output_dir / "per_sequence.csv"
    rows = per_sequence_rows(
        arrays,
        v3_records,
        analysis,
        exact_sequences,
    )
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    events_path = args.output_dir / "events.jsonl"
    rows = event_rows(arrays, analysis, exact_sequences)
    with events_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            )

    write_json(args.output_dir / "summary.json", summary)
    write_report(args.output_dir / "report.md", summary, mismatches)

    source_paths = {
        "frozen_events_npz": args.frozen_events_npz,
        "v3_metrics_jsonl": args.v3_metrics_jsonl,
        "frozen_queries_jsonl": args.frozen_queries_jsonl,
        "location_json": args.location_json,
        "schema_json": args.schema_json,
    }
    sources = {}
    for name, path in source_paths.items():
        if path is None:
            sources[name] = None
            continue
        sources[name] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command_argv": sys.argv,
        "sources": sources,
        "producer_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "git": git_metadata(Path(__file__).resolve().parents[1]),
        "outputs": {
            name: {
                "path": str((args.output_dir / name).resolve()),
                "sha256": sha256_file(args.output_dir / name),
            }
            for name in OUTPUT_FILES
            if name != "provenance.json"
            and (args.output_dir / name).is_file()
        },
        "non_negotiable_limits": [
            "validation split only",
            "no training",
            "no test split",
            "no GT query for inference-time candidate construction",
        ],
    }
    write_json(args.output_dir / "provenance.json", provenance)

    print(
        json.dumps(
            {
                "status": "PASS",
                "output_dir": str(args.output_dir.resolve()),
                "sequence_count": len(sequences),
                "v3_exact_sequence_count": len(exact_sequences),
                "count_mismatch_count": len(mismatches),
                "cewm_02_readiness": summary["decision"]["cewm_02_readiness"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
