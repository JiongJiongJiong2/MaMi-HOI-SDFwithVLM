"""Independently verify a CEWM-01 diagnostic output directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HANDS = ("left", "right")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--verification_output", type=Path)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def independent_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    intervals = []
    start = None
    for frame, value in enumerate(mask.astype(bool).tolist()):
        if value and start is None:
            start = frame
        if start is not None and (not value or frame == len(mask) - 1):
            end = frame if value else frame - 1
            intervals.append((start, end))
            start = None
    return intervals


def independent_match(
    gt_intervals: list[tuple[int, int]],
    pred_intervals: list[tuple[int, int]],
    tolerance: int,
) -> list[int | None]:
    unused = set(range(len(pred_intervals)))
    result = []
    for gt_start, _ in gt_intervals:
        eligible = [
            index
            for index in unused
            if abs(pred_intervals[index][0] - gt_start) <= tolerance
        ]
        if not eligible:
            result.append(None)
            continue
        selected = min(
            eligible,
            key=lambda index: (
                abs(pred_intervals[index][0] - gt_start),
                pred_intervals[index][0],
                index,
            ),
        )
        unused.remove(selected)
        result.append(selected)
    return result


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (output_dir / "provenance.json").read_text(encoding="utf-8")
    )
    schema = summary["schema"]
    threshold = float(schema["contact_threshold_m"])
    tolerance = int(schema["onset_match_tolerance_frames"])
    min_interval = int(schema["min_interval_frames"])
    checks = 0
    errors = []

    for name, expected in provenance["outputs"].items():
        path = output_dir / name
        if not path.is_file():
            errors.append(f"missing output: {name}")
            continue
        checks += 1
        if sha256_file(path) != expected["sha256"]:
            errors.append(f"output hash mismatch: {name}")

    for name, source in provenance["sources"].items():
        if source is None:
            continue
        path = Path(source["path"])
        checks += 1
        if not path.is_file():
            errors.append(f"missing source: {name}")
        elif sha256_file(path) != source["sha256"]:
            errors.append(f"source hash mismatch: {name}")

    frozen_path = Path(provenance["sources"]["frozen_events_npz"]["path"])
    arrays = np.load(frozen_path, allow_pickle=False)
    sequences = arrays["sequence_names"].astype(str).tolist()
    objects = arrays["object_names"].astype(str)
    gt_contact = arrays["gt_contact"].astype(bool)
    pred_contact = arrays["pred_contact"].astype(bool)
    distance_m = arrays["distance_m"].astype(np.float64)
    if not np.array_equal(pred_contact, distance_m < threshold):
        errors.append("frozen prediction mask disagrees with distance threshold")
    checks += 1

    v3_path = Path(provenance["sources"]["v3_metrics_jsonl"]["path"])
    v3 = {}
    for line in v3_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        raw_sequence = str(record["sequence"]).split("_sidx_", 1)[0]
        object_suffix = f"_{record['object']}"
        sequence = (
            raw_sequence[: -len(object_suffix)]
            if raw_sequence.endswith(object_suffix)
            else raw_sequence
        )
        v3[sequence] = record

    per_hand = {
        hand: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for hand in HANDS
    }
    expected_gt_events = 0
    expected_pred_events = 0
    expected_unmatched_pred_events = 0
    exact_sequences = set()
    expected_interval_rows = 0
    for sequence_index, sequence in enumerate(sequences):
        exact = True
        for hand_index, hand in enumerate(HANDS):
            gt = gt_contact[sequence_index, :, hand_index]
            pred = pred_contact[sequence_index, :, hand_index]
            local_counts = {}
            for key, mask in (
                ("tp", gt & pred),
                ("fp", ~gt & pred),
                ("tn", ~gt & ~pred),
                ("fn", gt & ~pred),
            ):
                local_counts[key] = int(mask.sum())
                per_hand[hand][key] += local_counts[key]
            for key in ("tp", "fp", "tn", "fn"):
                if local_counts[key] != int(v3[sequence][f"{hand}_{key}"]):
                    exact = False
            gt_intervals = [
                interval
                for interval in independent_runs(gt)
                if interval[1] - interval[0] + 1 >= min_interval
            ]
            pred_intervals = [
                interval
                for interval in independent_runs(pred)
                if interval[1] - interval[0] + 1 >= min_interval
            ]
            matches = independent_match(
                gt_intervals,
                pred_intervals,
                tolerance,
            )
            matched_pred = {match for match in matches if match is not None}
            expected_gt_events += len(gt_intervals)
            expected_pred_events += len(pred_intervals)
            expected_unmatched_pred_events += (
                len(pred_intervals) - len(matched_pred)
            )
            expected_interval_rows += len(gt_intervals) + (
                len(pred_intervals) - len(matched_pred)
            )
        if exact:
            exact_sequences.add(sequence)

    for hand in HANDS:
        observed = summary["headline_v3_all_sequences"][hand]
        for key in ("tp", "fp", "tn", "fn"):
            checks += 1
            expected = int(sum(int(v3[seq][f"{hand}_{key}"]) for seq in sequences))
            if int(observed[key]) != expected:
                errors.append(f"headline {hand}.{key} mismatch")
        if int(observed["gt_contact_pred_noncontact"]) != int(observed["fn"]):
            errors.append(f"headline {hand} FN alias mismatch")
        if int(observed["gt_noncontact_pred_contact"]) != int(observed["fp"]):
            errors.append(f"headline {hand} FP alias mismatch")

    if int(summary["local_timing"]["v3_exact_sequence_count"]) != len(
        exact_sequences
    ):
        errors.append("exact sequence count mismatch")
    checks += 1
    if set(summary["local_timing"]["v3_exact_sequences"]) != exact_sequences:
        errors.append("exact sequence set mismatch")
    checks += 1

    for hand_index, hand in enumerate(HANDS):
        event_totals = {
            "gt_interval_count": 0,
            "pred_interval_count": 0,
            "matched_gt_interval_count": 0,
            "missed_gt_interval_count": 0,
            "unmatched_pred_interval_count": 0,
            "continued_after_release_interval_count": 0,
        }
        release_sequence_means = []
        for sequence in sorted(exact_sequences):
            sequence_index = sequences.index(sequence)
            gt = gt_contact[sequence_index, :, hand_index]
            pred = pred_contact[sequence_index, :, hand_index]
            gt_intervals = [
                interval
                for interval in independent_runs(gt)
                if interval[1] - interval[0] + 1 >= min_interval
            ]
            pred_intervals = [
                interval
                for interval in independent_runs(pred)
                if interval[1] - interval[0] + 1 >= min_interval
            ]
            matches = independent_match(
                gt_intervals,
                pred_intervals,
                tolerance,
            )
            matched_pred = {match for match in matches if match is not None}
            release_delays = [
                pred_intervals[pred_index][1] - gt_intervals[gt_index][1]
                for gt_index, pred_index in enumerate(matches)
                if pred_index is not None
            ]
            positive_delays = [
                value for value in release_delays if value > 0
            ]
            if positive_delays:
                release_sequence_means.append(
                    float(np.mean(positive_delays))
                )
            event_totals["gt_interval_count"] += len(gt_intervals)
            event_totals["pred_interval_count"] += len(pred_intervals)
            event_totals["matched_gt_interval_count"] += len(matched_pred)
            event_totals["missed_gt_interval_count"] += (
                len(gt_intervals) - len(matched_pred)
            )
            event_totals["unmatched_pred_interval_count"] += (
                len(pred_intervals) - len(matched_pred)
            )
            event_totals["continued_after_release_interval_count"] += int(
                sum(value > 0 for value in release_delays)
            )

        observed = summary["local_timing"]["v3_exact_subset"][hand]
        for key, expected in event_totals.items():
            checks += 1
            if int(observed["events"][key]) != expected:
                errors.append(
                    f"timing event mismatch {hand}.{key}: "
                    f"expected={expected}, observed={observed['events'][key]}"
                )
        observed_release = observed["sequence_macro"][
            "mean_positive_release_delay_frames"
        ]
        expected_release = (
            float(np.mean(release_sequence_means))
            if release_sequence_means
            else None
        )
        checks += 1
        if expected_release is None and observed_release is not None:
            errors.append(
                f"timing release mismatch {hand}: expected=None, "
                f"observed={observed_release}"
            )
        elif expected_release is not None and (
            observed_release is None
            or abs(float(observed_release) - expected_release) > 1e-12
        ):
            errors.append(
                f"timing release mismatch {hand}: "
                f"expected={expected_release}, observed={observed_release}"
            )

    rows = read_csv(output_dir / "per_sequence.csv")
    interval_rows = [
        row for row in rows if row["record_type"] != "sequence_hand_summary"
    ]
    if len(interval_rows) != expected_interval_rows:
        errors.append(
            f"per_sequence interval rows: expected={expected_interval_rows}, "
            f"observed={len(interval_rows)}"
        )
    checks += 1
    summary_rows = [
        row for row in rows if row["record_type"] == "sequence_hand_summary"
    ]
    if len(summary_rows) != len(sequences) * len(HANDS):
        errors.append("per_sequence summary row count mismatch")
    checks += 1

    events = load_jsonl(output_dir / "events.jsonl")
    gt_events = [
        event for event in events if event["event_type"] == "gt_contact_interval"
    ]
    pred_events = [
        event
        for event in events
        if event["event_type"] == "pred_contact_interval"
    ]
    unmatched_pred_events = [
        event
        for event in pred_events
        if event["match_status"] == "unmatched_prediction"
    ]
    if len(gt_events) != expected_gt_events:
        errors.append("GT event row count mismatch")
    if len(pred_events) != expected_pred_events:
        errors.append("prediction event row count mismatch")
    if len(unmatched_pred_events) != expected_unmatched_pred_events:
        errors.append("unmatched prediction event count mismatch")
    checks += 3

    verification = {
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
        "verifier_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "recomputed": {
            "sequence_count": len(sequences),
            "v3_exact_sequence_count": len(exact_sequences),
            "gt_interval_events": expected_gt_events,
            "pred_interval_events": expected_pred_events,
            "unmatched_pred_interval_events": expected_unmatched_pred_events,
            "per_sequence_interval_rows": expected_interval_rows,
        },
        "verified_files": {
            name: sha256_file(output_dir / name)
            for name in (
                "per_sequence.csv",
                "events.jsonl",
                "summary.json",
                "report.md",
            )
        },
    }
    output_path = (
        args.verification_output
        if args.verification_output
        else output_dir / "verification.json"
    )
    output_path.write_text(
        json.dumps(verification, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
