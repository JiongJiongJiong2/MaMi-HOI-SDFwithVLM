# OakInk Sequence-Aware Handover Detector Result

Date: 2026-09-21

Status: complete with sequence-aware event gate PASS

Protocol:

```text
docs/experiments/oakink-sequence-aware-handover-detector-protocol-2026-09-21.md
```

Summary:

```text
docs/experiments/oakink_sequence_aware_detector_summary_v1_20260921.json
```

## Method

The frozen v1 logistic scores and threshold were used without
retraining:

```text
model fitting:          none
threshold tuning:       none
two-person eligibility: source == handover
peaks per sequence:     one argmax peak
event matching:         +/-15 frames, one-to-one
```

## Event Metrics

| Split | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| train | 0.5909 | 0.7879 | 0.6753 | 26 | 18 | 7 |
| val | 0.5000 | 0.5556 | 0.5263 | 20 | 20 | 16 |
| test | 0.7561 | 0.8378 | 0.7949 | 31 | 10 | 6 |

## Gate

```text
train precision >= 0.50: pass
train recall >= 0.50:    pass
train F1 >= 0.50:        pass
val precision >= 0.50:   pass
val recall >= 0.50:      pass
val F1 >= 0.50:          pass
test precision >= 0.50:  pass
test recall >= 0.50:     pass
test F1 >= 0.50:         pass
overall:                 PASS
```

The sequence-aware revision fixes the v1 multi-peak precision failure
without changing the model or threshold.

## Server Reproduction

The deterministic post-processing was executed on the server on the v1
score output:

```text
/root/autodl-tmp/oakink_sequence_aware_detector_v1_20260921
```

Markers:

```text
exit_code: 0
SUCCESS:   present
```

## Boundary

The experiment assumes:

1. a handover event requires an eligible two-person/two-hand stream;
2. each OakInk handover sequence contains one primary transition.

This is appropriate for the OakInk benchmark, but it is not yet a
general long-sequence detector that can emit and rank an unbounded
number of handovers in one stream.

## Decision

The external OakInk route now has:

```text
explicit handover labels;
participant-disjoint train/val/test;
geometric contact verification;
handover-onset ranking;
cross-intent specificity;
sequence-aware event detection.
```

The next project step is to consume these detected transitions for a
downstream contact-aware task rather than continue tuning the event
classifier. Suitable next targets are:

```text
receiver-hand state prediction at the transfer boundary;
object-relative hand residual extraction;
handover transition timing evaluation in a frozen MaMi/contact pipeline.
```

## Implementation Hashes

```text
scripts/evaluate_oakink_sequence_aware_handover_detector.py
f9a8f23c71f1cf147363bbc471a7cc6dcd75b11f2d1cf74ae7fa5e8c7a3ae378

tests/test_evaluate_oakink_sequence_aware_handover_detector.py
e28cf71a9ae552fbde7c4d3b9ad66fa621927299af9afd086e6d710c30ff524e

summary
7e08f5f4b1a10144ef0984c9e55a3330107c274ee0f2ceeafe259a8eedd71173
```
