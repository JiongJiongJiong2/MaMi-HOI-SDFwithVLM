# ARCTIC Motion Feature Stability Protocol

Date: 2026-09-21

Status: frozen before stability audit

## Goal

Determine whether the motion-aware improvement is stable across feature
groups and participant-fold seeds, rather than a single-fold artifact.

This is a stability audit of the existing motion dataset. It does not
redefine the trigger, labels, or official-test boundary.

## Feature Groups

Feature indices refer to the 16-dimensional motion window:

```text
distance:            indices 0-9
hand speed:          indices 10-11
relative speed:      indices 12-13
object translation:  index 14
object angular:      index 15
motion only:         indices 10-15
all:                 indices 0-15
```

The primary seed uses this feature set:

```text
distance
hand speed
relative speed
object motion
motion only
all
all minus hand speed
all minus relative speed
all minus object motion
```

## Fold Stability

Five seeds are tested:

```text
20260921
20260922
20260923
20260924
20260925
```

Participants are assigned to four folds while balancing positive-event
counts and sample counts. Only training participants are used for fold
construction and threshold selection.

For each seed, compare:

```text
distance only
motion only
all features
```

HistGradientBoosting and the fixed hyperparameters remain unchanged.

## Calibration

For the primary all-feature model, report:

```text
raw val Brier and expected calibration error
isotonic-calibrated val Brier and expected calibration error
Platt-calibrated val Brier and expected calibration error
```

Calibration is descriptive and not a pass/fail gate.

## Stability Gate

The motion-aware improvement is stable only if:

1. mean five-seed train OOF AUPRC for all features exceeds distance only;
2. at least four of five seeds have higher all-feature OOF AUPRC than
   distance only;
3. mean five-seed val AUPRC is at least `0.6484`;
4. mean five-seed val F1 is at least `0.6154`.

Failure means the motion-aware gain is not yet robust enough to carry
forward without further feature or data changes.
