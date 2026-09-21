# ARCTIC Online Role-Switch Detector Protocol

Date: 2026-09-21

Status: frozen before execution

## Goal

Build a transition detector that emits candidate moments without using
the future receiving-hand onset, then classify each emitted release as
Tier A or non-Tier A from pre-release history.

This removes the retrospective candidate-selection leakage in the
previous window-model experiment.

## Online Release Trigger

For each hand independently:

```text
stable contact:           at least 15 consecutive raw-contact frames
release frame:            first raw non-contact frame after stable contact
release confirmation:     five consecutive raw non-contact frames
contact threshold:        3 mm
```

The detector uses only frames up to release plus the confirmation window.
It does not read the other hand's future contact or receiving onset.

## Labels

An emitted release is positive only if:

```text
(sequence, outgoing_hand, release_frame)
```

matches a Tier A candidate exactly.

All other confirmed releases are negatives.

## Features

The model uses the same 30-frame, 10-feature window as the previous
candidate-quality experiment. The window ends before release.

## Models

```text
logistic
HistGradientBoostingClassifier
```

Hyperparameters and participant-grouped out-of-fold threshold selection
are unchanged from the candidate-quality protocol.

## Gate

The pilot passes only if:

1. Tier A release recall from the deterministic trigger is at least 0.90
   in train and val;
2. the best model val AUPRC exceeds val prevalence;
3. the best model val F1 exceeds all-positive F1;
4. the best model val F1 is at least 0.50.

Official test remains unavailable.
