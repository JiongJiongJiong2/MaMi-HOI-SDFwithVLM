# OakInk Full-Sequence Online Handover Detector Result

Date: 2026-09-21

Status: frozen v1 gate FAIL on val precision

Protocol:

```text
docs/experiments/oakink-online-handover-detector-protocol-2026-09-21.md
```

Summary:

```text
docs/experiments/oakink_online_handover_detector_summary_v1_20260921.json
```

## Event Metrics

| Split | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.4308 | 0.8485 | 0.5714 | 28 | 37 | 5 |
| val | 0.4286 | 0.7500 | 0.5455 | 27 | 36 | 9 |
| test | 0.5926 | 0.8649 | 0.7033 | 32 | 22 | 5 |

The test split passes all event gates. Val precision is `0.4286`, below
the frozen `0.50` requirement, so the original v1 detector FAILS.

## Error Decomposition

False positives by source:

| Split | Handover False Positives | Non-Handover False Positives |
|---|---:|---:|
| train | 29 | 8 |
| val | 28 | 8 |
| test | 17 | 5 |

Most false positives are multiple peaks inside handover sequences, not
ordinary actions being misclassified.

Per-intent non-handover false positives:

```text
train: hold 6, use 2, liftup 0
val:   hold 2, liftup 2, use 4
test:  hold 1, liftup 3, use 1
```

## Diagnostic Post-Processing

Two post-processing changes were evaluated after reading the v1 result,
without changing the model or threshold:

1. label the candidate as handover-eligible only when a second-hand
   stream exists;
2. keep only the highest-confidence peak per eligible sequence.

Diagnostic metrics:

| Split | Precision | Recall | F1 |
|---|---:|---:|---:|
| val | 0.5000 | 0.5556 | 0.5263 |
| test | 0.7561 | 0.8378 | 0.7949 |

This diagnostic is not treated as a passed frozen experiment. It shows
that the main remaining issue is temporal post-processing rather than
handover intent discrimination.

## Decision

The model's ranking signal is useful, but the raw full-sequence detector
emits too many peaks. The next experiment must freeze a sequence-aware
detector protocol before rerunning:

```text
two-person/two-hand eligibility;
one primary transition per eligible sequence;
event matching tolerance 15 frames;
event precision, recall, and F1.
```

No model retraining or threshold tuning is needed for the next step.

## Implementation Hashes

```text
scripts/evaluate_oakink_online_handover_detector.py
a2652e217f70b82e0b79bb92e78dc9ed54bc47fd9961d1c050506f971e4fd5c9

tests/test_evaluate_oakink_online_handover_detector.py
955f1d32d7bac80168368e5385c886a2870671d9b2691a22d82c751c212ac1a6

summary
b3447eca15e5cfe54b6efd2098b8b37d1bd1165ff01b922c3f8a12107cbc4208
```
