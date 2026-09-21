# OakInk Handover Window Model Protocol

Date: 2026-09-21

Status: frozen before execution

## Goal

Test whether a giver-to-receiver handover transition can be predicted
from pre-transition geometry in the participant-disjoint OakInk
manifest.

This is an event-onset prediction pilot, not an image model.

## Inputs

```text
/root/autodl-tmp/oakink_handover_manifest_v1_20260921/
  handover_trajectories.npz
  role_switch_candidates.jsonl.gz
```

Primary contact threshold: `5 mm`.

## Event Selection

For each sequence, select one primary transition from the `5 mm`
role-switch candidates:

1. smallest absolute onset-minus-release offset;
2. earliest release frame as the tie-breaker.

Sequences without a `5 mm` candidate remain all-negative windows.

## Labels

For each valid history frame `t`:

```text
positive: release - 15 <= t < release
ambiguous skip: |t - release| <= 30 outside the positive interval
negative: all remaining frames
```

## Features

```text
history: 30 frames
per frame:
  giver/receiver hand-object distance, capped at 20 mm
  giver/receiver 1-frame distance delta
  giver/receiver 5-frame distance delta
  giver/receiver contact flags at 5 mm
  giver/receiver contact flags at 10 mm
```

No frame at or after the release is used as input.

## Split

The frozen OakInk subject split is mandatory:

```text
train: {0,5,6,9,10}
val:   {2,4,7,8}
test:  {1,3}
```

Hard F1 thresholds are selected on participant-pair-grouped out-of-fold
training scores. Val and test do not participate in fitting or threshold
selection.

## Models

```text
logistic
HistGradientBoostingClassifier
```

Hyperparameters match the established role-switch window baseline.

## Gate

The pilot passes only if the best model:

1. exceeds class prevalence in val AUPRC;
2. exceeds all-positive F1 on val;
3. exceeds class prevalence in test AUPRC;
4. exceeds all-positive F1 on test;
5. reaches test recall at least 0.50 using the frozen threshold.

The held-out test is OakInk subject-disjoint, but it is still a pilot
external dataset result rather than an official OakInk benchmark.
