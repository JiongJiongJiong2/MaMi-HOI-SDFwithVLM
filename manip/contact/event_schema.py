"""Versioned contact-event definitions for the CEWM diagnostic."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


CONTACT_EVENT_SCHEMA_VERSION = "cewm_event_v1"
HANDS = ("left", "right")
HAND_COLUMNS = {"left": 0, "right": 1}


@dataclass(frozen=True)
class ContactEventSchema:
    """Frozen definitions shared by the diagnostic and verifier."""

    version: str = CONTACT_EVENT_SCHEMA_VERSION
    contact_threshold_m: float = 0.05
    min_interval_frames: int = 1
    onset_match_tolerance_frames: int = 5
    sequence_bootstrap_unit: str = "sequence"
    window_deduplication_key: str = "sequence_name,absolute_frame"
    contact_distance_definition: str = (
        "exact unsigned canonical predicted palm-to-object-triangle distance"
    )
    hand_mesh_clearance_status: str = (
        "NOT_DEFINED_IN_CEWM_01_FROZEN_ARRAYS"
    )
    location_status: str = (
        "BOUNDED_EXISTING_BPR_OVERLAP_DIAGNOSTIC"
    )

    def __post_init__(self) -> None:
        if self.version != CONTACT_EVENT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported event schema: {self.version}")
        if not math.isfinite(self.contact_threshold_m) or self.contact_threshold_m <= 0:
            raise ValueError("contact_threshold_m must be finite and positive")
        if self.min_interval_frames < 1:
            raise ValueError("min_interval_frames must be at least one")
        if self.onset_match_tolerance_frames < 0:
            raise ValueError("onset_match_tolerance_frames must be non-negative")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "ContactEventSchema":
        expected = set(asdict(cls()).keys())
        observed = set(values.keys())
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                f"Schema keys differ: missing={missing}, extra={extra}"
            )
        return cls(
            version=str(values["version"]),
            contact_threshold_m=float(values["contact_threshold_m"]),
            min_interval_frames=int(values["min_interval_frames"]),
            onset_match_tolerance_frames=int(
                values["onset_match_tolerance_frames"]
            ),
            sequence_bootstrap_unit=str(values["sequence_bootstrap_unit"]),
            window_deduplication_key=str(
                values["window_deduplication_key"]
            ),
            contact_distance_definition=str(
                values["contact_distance_definition"]
            ),
            hand_mesh_clearance_status=str(
                values["hand_mesh_clearance_status"]
            ),
            location_status=str(values["location_status"]),
        )


def _as_mask(mask: np.ndarray, name: str) -> np.ndarray:
    mask = np.asarray(mask)
    if mask.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(mask.astype(np.float64))):
        raise ValueError(f"{name} contains non-finite values")
    return mask.astype(bool)


def interval_runs(
    mask: np.ndarray,
    min_interval_frames: int = 1,
) -> list[tuple[int, int]]:
    """Return inclusive true runs, dropping runs shorter than the frozen minimum."""
    if min_interval_frames < 1:
        raise ValueError("min_interval_frames must be at least one")
    values = _as_mask(mask, "mask")
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(values.tolist()):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(values) - 1):
            end = index if value else index - 1
            if end - start + 1 >= min_interval_frames:
                intervals.append((start, end))
            start = None
    return intervals


def match_intervals(
    gt_intervals: Sequence[tuple[int, int]],
    pred_intervals: Sequence[tuple[int, int]],
    onset_match_tolerance_frames: int,
) -> list[int | None]:
    """One-to-one match each GT interval to the closest eligible pred onset."""
    if onset_match_tolerance_frames < 0:
        raise ValueError("onset tolerance must be non-negative")
    available = set(range(len(pred_intervals)))
    matches: list[int | None] = []
    for gt_start, _ in gt_intervals:
        candidates = [
            index
            for index in available
            if abs(pred_intervals[index][0] - gt_start)
            <= onset_match_tolerance_frames
        ]
        if not candidates:
            matches.append(None)
            continue
        selected = min(
            candidates,
            key=lambda index: (
                abs(pred_intervals[index][0] - gt_start),
                pred_intervals[index][0],
                index,
            ),
        )
        available.remove(selected)
        matches.append(selected)
    return matches


def _safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(np.mean(values)) if values else None


def _confusion(gt: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def _f1(tp: int, fp: int, fn: int) -> float | None:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else None


def analyze_hand_sequence(
    gt_contact: np.ndarray,
    pred_contact: np.ndarray,
    distance_m: np.ndarray,
    schema: ContactEventSchema | None = None,
) -> dict:
    """Compute counts and interval timing for one sequence and one hand."""
    schema = schema or ContactEventSchema()
    gt = _as_mask(gt_contact, "gt_contact")
    pred = _as_mask(pred_contact, "pred_contact")
    distance_m = np.asarray(distance_m, dtype=np.float64)
    if distance_m.shape != gt.shape:
        raise ValueError("distance_m must match the contact masks")
    if not np.all(np.isfinite(distance_m)):
        raise ValueError("distance_m contains non-finite values")
    if np.any(distance_m < 0.0):
        raise ValueError("distance_m must be non-negative")
    if not np.array_equal(pred, distance_m < schema.contact_threshold_m):
        raise ValueError(
            "pred_contact does not match the frozen distance threshold"
        )

    counts = _confusion(gt, pred)
    gt_intervals = interval_runs(gt, schema.min_interval_frames)
    pred_intervals = interval_runs(pred, schema.min_interval_frames)
    matches = match_intervals(
        gt_intervals,
        pred_intervals,
        schema.onset_match_tolerance_frames,
    )
    matched_prediction_indices = {
        match for match in matches if match is not None
    }
    matched_rows = []
    onset_errors = []
    release_delays = []
    continued_after_release = []
    for gt_index, (gt_start, gt_end) in enumerate(gt_intervals):
        pred_index = matches[gt_index]
        if pred_index is None:
            matched_rows.append(
                {
                    "gt_interval_index": gt_index,
                    "gt_start": gt_start,
                    "gt_end": gt_end,
                    "pred_interval_index": None,
                    "pred_start": None,
                    "pred_end": None,
                    "match_status": "missed",
                    "onset_error_frames": None,
                    "release_delay_frames": None,
                    "continued_after_release_frames": None,
                }
            )
            continue
        pred_start, pred_end = pred_intervals[pred_index]
        onset_error = pred_start - gt_start
        release_delay = pred_end - gt_end
        continued = max(0, pred_end - gt_end)
        onset_errors.append(float(onset_error))
        release_delays.append(float(release_delay))
        continued_after_release.append(int(continued))
        matched_rows.append(
            {
                "gt_interval_index": gt_index,
                "gt_start": gt_start,
                "gt_end": gt_end,
                "pred_interval_index": pred_index,
                "pred_start": pred_start,
                "pred_end": pred_end,
                "match_status": "matched",
                "onset_error_frames": int(onset_error),
                "release_delay_frames": int(release_delay),
                "continued_after_release_frames": int(continued),
            }
        )

    gt_contact_distances = distance_m[gt]
    gt_noncontact_distances = distance_m[~gt]
    result = {
        **counts,
        "f1": _f1(counts["tp"], counts["fp"], counts["fn"]),
        "gt_contact_frames": int(gt.sum()),
        "pred_contact_frames": int(pred.sum()),
        "gt_interval_count": len(gt_intervals),
        "pred_interval_count": len(pred_intervals),
        "matched_gt_interval_count": len(matched_prediction_indices),
        "missed_gt_interval_count": (
            len(gt_intervals) - len(matched_prediction_indices)
        ),
        "unmatched_pred_interval_count": (
            len(pred_intervals) - len(matched_prediction_indices)
        ),
        "gt_intervals": gt_intervals,
        "pred_intervals": pred_intervals,
        "interval_matches": matched_rows,
        "onset_errors_frames": [int(value) for value in onset_errors],
        "release_delays_frames": [int(value) for value in release_delays],
        "continued_after_release_frames": continued_after_release,
        "mean_onset_error_frames": _safe_mean(onset_errors),
        "median_abs_onset_error_frames": (
            float(np.median(np.abs(onset_errors)))
            if onset_errors
            else None
        ),
        "mean_release_delay_frames": _safe_mean(release_delays),
        "mean_positive_release_delay_frames": _safe_mean(
            value for value in release_delays if value > 0.0
        ),
        "mean_continued_after_release_frames": _safe_mean(
            float(value) for value in continued_after_release if value > 0
        ),
        "mean_distance_on_gt_contact_mm": (
            float(gt_contact_distances.mean() * 1000.0)
            if len(gt_contact_distances)
            else None
        ),
        "mean_distance_on_gt_noncontact_mm": (
            float(gt_noncontact_distances.mean() * 1000.0)
            if len(gt_noncontact_distances)
            else None
        ),
        "contact_correct_still_floating": (
            schema.hand_mesh_clearance_status
        ),
        "contact_location_error": schema.location_status,
        "contact_slip": schema.location_status,
    }
    return result


def aggregate_sequence_hand_records(
    records: Sequence[Mapping[str, object]],
) -> dict:
    """Aggregate by hand with micro counts and sequence-macro timing."""
    result = {}
    for hand in HANDS:
        hand_records = [
            record for record in records if str(record["hand"]) == hand
        ]
        if not hand_records:
            continue
        totals = {
            key: int(sum(int(record[key]) for record in hand_records))
            for key in ("tp", "fp", "tn", "fn")
        }
        f1_values = [
            float(record["f1"])
            for record in hand_records
            if record["f1"] is not None
        ]
        onset_values = [
            float(record["mean_onset_error_frames"])
            for record in hand_records
            if record["mean_onset_error_frames"] is not None
        ]
        abs_onset_values = [
            float(record["median_abs_onset_error_frames"])
            for record in hand_records
            if record["median_abs_onset_error_frames"] is not None
        ]
        positive_release_values = [
            float(record["mean_positive_release_delay_frames"])
            for record in hand_records
            if record["mean_positive_release_delay_frames"] is not None
        ]
        result[hand] = {
            "micro": {
                **totals,
                "f1": _f1(totals["tp"], totals["fp"], totals["fn"]),
            },
            "sequence_macro": {
                "sequence_count": len(hand_records),
                "f1": _safe_mean(f1_values),
                "mean_onset_error_frames": _safe_mean(onset_values),
                "median_abs_onset_error_frames": _safe_mean(
                    abs_onset_values
                ),
                "mean_positive_release_delay_frames": _safe_mean(
                    positive_release_values
                ),
            },
            "events": {
                "gt_interval_count": int(
                    sum(int(record["gt_interval_count"]) for record in hand_records)
                ),
                "pred_interval_count": int(
                    sum(
                        int(record["pred_interval_count"])
                        for record in hand_records
                    )
                ),
                "matched_gt_interval_count": int(
                    sum(
                        int(record["matched_gt_interval_count"])
                        for record in hand_records
                    )
                ),
                "missed_gt_interval_count": int(
                    sum(
                        int(record["missed_gt_interval_count"])
                        for record in hand_records
                    )
                ),
                "unmatched_pred_interval_count": int(
                    sum(
                        int(record["unmatched_pred_interval_count"])
                        for record in hand_records
                    )
                ),
                "continued_after_release_interval_count": int(
                    sum(
                        int(record["continued_after_release_frames"][index] > 0)
                        for record in hand_records
                        for index in range(
                            len(record["continued_after_release_frames"])
                        )
                    )
                ),
            },
        }
    return result
