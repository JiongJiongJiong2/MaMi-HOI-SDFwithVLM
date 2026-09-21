# ARCTIC Role-Switch Window Model Result

Date: 2026-09-21

Status: pilot candidate-quality gate PASS; official test unavailable

Protocol:

```text
docs/experiments/arctic-role-switch-window-model-protocol-2026-09-21.md
```

## Dataset

Every raw `3 mm` candidate is one event-centered sample:

```text
history:       30 frames before outgoing release
features:     10 per frame
train:      1,182 samples, 45 Tier A positives
val:          143 samples,  9 Tier A positives
```

No frame at or after the outgoing release is used as model input.

Class references:

```text
train prevalence:       0.0381
train all-positive F1:  0.0733
val prevalence:         0.0629
val all-positive F1:    0.1184
```

## Metrics

| Model | Split | AUPRC | AUROC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| logistic | train OOF | 0.2023 | 0.7200 | 0.3256 | 0.3415 | 0.3111 |
| logistic | val | 0.2649 | 0.7852 | 0.3750 | 0.4286 | 0.3333 |
| boosting | train OOF | 0.2391 | 0.8476 | 0.3919 | 0.2816 | 0.6444 |
| boosting | val | 0.6257 | 0.9519 | 0.5385 | 0.4118 | 0.7778 |

Hard F1 thresholds were selected only on pooled participant-grouped
out-of-fold training predictions.

The boosting arm is the pilot winner. The result indicates that the
pre-release distance and contact history contains information about
whether a raw transition will become a clean Tier A event.

## Gate

```text
best val AUPRC above prevalence:             pass
best val F1 above all-positive F1:           pass
train OOF AUPRC above train prevalence:      pass
pilot overall:                               PASS
official test available:                     false
final benchmark ready:                       false
```

## Reproduction

The window dataset and models were run locally and on the server.

```text
summary structure differences:       0
boosting score max difference:       1.11e-16
logistic score max difference:       1.34e-4
reported metric differences:         0 after rounding
```

The logistic difference comes from environment-specific solver
arithmetic and does not change the decision.

## Boundary

This is a retrospective candidate-quality classifier, not an online
role-switch detector. Candidate construction uses the future receiving
onset, so deployment would still require an independent trigger.

The result cannot support human handover labels or official ARCTIC test
performance. The public release still lacks test subject `s03`.

## Artifacts

Server:

```text
/root/autodl-tmp/arctic_role_switch_windows_v1_20260921
/root/autodl-tmp/arctic_role_switch_model_v1_20260921
```

Local review copy:

```text
C:\Users\何炯乐\Documents\HOI项目\arctic_role_switch_windows_v1_20260921
C:\Users\何炯乐\Documents\HOI项目\arctic_role_switch_model_v1_20260921
```

## Decision

The accessible ARCTIC data now support:

```text
geometric role-switch candidate generation;
Tier A/B quality manifests;
event-level object/hand motion review;
participant-grouped candidate-quality prediction.
```

The next technical step is an online transition detector that emits
candidate times without using future receiving onset, followed by
event-level precision/recall against Tier A. Handover claims remain
prohibited until receiver-role validation is complete.

## Implementation Hashes

```text
scripts/build_arctic_role_switch_windows.py
14a33711051f98ff8b50c1e92e20c3b254399424c8d0ffca6625ea3b92377355

scripts/train_arctic_role_switch_candidates.py
2739a4b407cad50f60f27e9acad9bdfc63a72a1a1552e2e8b56b39f1612b102c

tests/test_build_arctic_role_switch_windows.py
92bfe7faa768048ccba820c8e021aac8d656702a98be5ac2677164ceee06e267

tests/test_train_arctic_role_switch_candidates.py
d83392ae09fd5daa36a8443bc68cf7e89ffc60962fe6339b6c8bea1821aac4eb
```
