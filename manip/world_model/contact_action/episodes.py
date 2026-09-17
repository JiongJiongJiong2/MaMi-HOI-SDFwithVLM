"""Contact episode metrics for short hand-object interaction windows."""

from __future__ import annotations

import numpy as np


def _binary_sequence(values, name):
    array = np.asarray(values, dtype=bool)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}")
    return array


def contact_runs(mask):
    """Return inclusive ``(start, end)`` runs of true values."""
    mask = _binary_sequence(mask, "mask")
    runs = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _runs_overlap(first, second):
    return first[0] <= second[1] and second[0] <= first[1]


def _mean_or_none(values):
    values = [float(value) for value in values if value is not None]
    return float(np.mean(values)) if values else None


def contact_episode_metrics(
    predicted_contact,
    ground_truth_contact,
    stable_min_frames=3,
):
    """Summarize one hand event as frame and episode-level contact metrics."""
    if stable_min_frames < 1:
        raise ValueError("stable_min_frames must be positive")
    prediction = _binary_sequence(predicted_contact, "predicted_contact")
    truth = _binary_sequence(ground_truth_contact, "ground_truth_contact")
    if prediction.shape != truth.shape:
        raise ValueError(
            "predicted_contact and ground_truth_contact must align"
        )

    predicted_runs = contact_runs(prediction)
    ground_truth_runs = contact_runs(truth)
    predicted_stable_runs = [
        run for run in predicted_runs
        if run[1] - run[0] + 1 >= stable_min_frames
    ]
    ground_truth_stable_runs = [
        run for run in ground_truth_runs
        if run[1] - run[0] + 1 >= stable_min_frames
    ]

    true_positive = int((prediction & truth).sum())
    false_positive = int((prediction & ~truth).sum())
    false_negative = int((~prediction & truth).sum())
    f1_denominator = 2 * true_positive + false_positive + false_negative
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative

    predicted_indices = np.flatnonzero(prediction)
    truth_indices = np.flatnonzero(truth)
    first_predicted = (
        int(predicted_indices[0]) if predicted_indices.size else None
    )
    last_predicted = (
        int(predicted_indices[-1]) if predicted_indices.size else None
    )
    first_truth = int(truth_indices[0]) if truth_indices.size else None
    last_truth = int(truth_indices[-1]) if truth_indices.size else None

    onset_delay_frames = (
        first_predicted - first_truth
        if first_predicted is not None and first_truth is not None
        else None
    )
    early_contact_frames = (
        int(prediction[:first_truth].sum())
        if first_truth is not None
        else int(prediction.sum())
    )

    stable_contact_success = int(any(
        _runs_overlap(predicted_run, truth_run)
        for predicted_run in predicted_stable_runs
        for truth_run in ground_truth_runs
    ))
    stable_contact_false_positive = int(bool(predicted_stable_runs) and not any(
        _runs_overlap(predicted_run, truth_run)
        for predicted_run in predicted_stable_runs
        for truth_run in ground_truth_runs
    ))

    dropout_count = 0
    if (
        first_truth is not None
        and last_truth is not None
        and last_truth > first_truth
    ):
        false_runs = contact_runs(~prediction)
        dropout_count = sum(
            1 for run in false_runs
            if run[0] > first_truth and run[1] < last_truth
        )

    late_contact_frames = None
    release_delay_frames = None
    if last_truth is not None and last_truth < len(prediction) - 1:
        after_release = prediction[last_truth + 1 :]
        late_contact_frames = int(after_release.sum())
        after_release_indices = np.flatnonzero(after_release)
        release_delay_frames = (
            int(after_release_indices[-1]) + 1
            if after_release_indices.size
            else 0
        )

    return {
        "predicted_contact_frame_count": int(prediction.sum()),
        "ground_truth_contact_frame_count": int(truth.sum()),
        "predicted_contact_episode_count": len(predicted_runs),
        "ground_truth_contact_episode_count": len(ground_truth_runs),
        "predicted_stable_episode_count": len(predicted_stable_runs),
        "ground_truth_stable_episode_count": len(
            ground_truth_stable_runs
        ),
        "longest_predicted_episode_frames": max(
            (run[1] - run[0] + 1 for run in predicted_runs),
            default=0,
        ),
        "longest_ground_truth_episode_frames": max(
            (run[1] - run[0] + 1 for run in ground_truth_runs),
            default=0,
        ),
        "contact_precision": (
            float(true_positive / precision_denominator)
            if precision_denominator
            else None
        ),
        "contact_recall": (
            float(true_positive / recall_denominator)
            if recall_denominator
            else None
        ),
        "contact_f1": (
            float(2.0 * true_positive / f1_denominator)
            if f1_denominator
            else None
        ),
        "stable_contact_success": stable_contact_success,
        "stable_contact_false_positive": stable_contact_false_positive,
        "dropout_count": int(dropout_count),
        "onset_delay_frames": onset_delay_frames,
        "early_contact_frames": int(early_contact_frames),
        "late_contact_frames": late_contact_frames,
        "release_delay_frames": release_delay_frames,
        "first_predicted_contact_frame": first_predicted,
        "last_predicted_contact_frame": last_predicted,
        "first_ground_truth_contact_frame": first_truth,
        "last_ground_truth_contact_frame": last_truth,
    }


def aggregate_episode_metrics(records):
    """Macro-average episode metrics, ignoring undefined values."""
    records = list(records)
    if not records:
        return {}
    keys = sorted({
        key
        for record in records
        for key in record
    })
    aggregate = {}
    for key in keys:
        values = [
            record.get(key)
            for record in records
            if record.get(key) is not None
        ]
        aggregate[key] = (
            float(np.mean([float(value) for value in values]))
            if values
            else None
        )
        aggregate[f"{key}_defined_count"] = len(values)
    return aggregate

