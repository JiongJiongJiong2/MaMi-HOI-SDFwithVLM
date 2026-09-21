# OakInk Full-Sequence Online Handover Detector Protocol

Date: 2026-09-21

Status: frozen before execution

## Goal

Evaluate autonomous handover event detection over complete OakInk
sequences without using future receiving onset or pre-selected event
windows.

## Frozen Model

```text
StandardScaler + logistic regression
training samples: handover-train windows only
threshold: participant-pair-grouped handover-train OOF F1
```

No val/test threshold tuning is allowed.

## Full-Sequence Scoring

For every frame with at least one prior frame, create the same 30-frame
10-feature history.

Single-hand non-handover sequences use:

```text
active hand -> giver channel
absent hand -> 20 mm no-contact channel
```

## Peak Detection

```text
threshold crossing: score >= frozen threshold
peak: argmax score in each contiguous crossing run
non-maximum suppression: 15 frames
```

The detector emits one event time per retained peak.

## Ground Truth

Handover ground truth is the primary `5 mm` giver release per handover
sequence, selected by:

```text
smallest absolute onset-minus-release offset
earliest release as tie-breaker
```

Non-handover sequences have no ground-truth handover events.

## Event Matching

A prediction matches a ground truth event when both occur in the same
sequence within `15` frames. Matching is greedy nearest-frame and
one-to-one.

## Metrics and Gate

Report:

```text
event precision, recall, F1
true positives, false positives, false negatives
false-positive sequences
```

The detector passes only if val and test each reach:

```text
precision >= 0.50
recall    >= 0.50
F1        >= 0.50
```

Failure means the model is not yet an autonomous event detector.
