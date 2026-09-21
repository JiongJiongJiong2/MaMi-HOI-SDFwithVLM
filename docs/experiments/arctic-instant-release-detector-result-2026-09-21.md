# ARCTIC Instant-Release Detector Result

Date: 2026-09-21

Status: revised online pilot PASS; official test unavailable

Protocol:

```text
docs/experiments/arctic-instant-release-detector-protocol-2026-09-21.md
```

Failed predecessor:

```text
docs/experiments/arctic-online-role-switch-detector-result-2026-09-21.md
```

## Trigger

The revised trigger emits the first raw non-contact frame after a
15-frame stable contact segment. It has no future confirmation window
and does not read receiving-hand onset.

At `3 mm`:

| Split | Tier A Events | Triggered | Recall |
|---|---:|---:|---:|
| train | 45 | 45 | 1.000 |
| val | 9 | 9 | 1.000 |

## Dataset

```text
train: 3,637 confirmed release events, 45 Tier A positives
val:     476 confirmed release events,  9 Tier A positives
```

References:

```text
train prevalence:       0.0124
train all-positive F1:  0.0244
val prevalence:         0.0189
val all-positive F1:    0.0371
```

## Metrics

| Model | Split | AUPRC | AUROC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| logistic | train OOF | 0.1050 | 0.7274 | 0.1899 | 0.1327 | 0.3333 |
| logistic | val | 0.2380 | 0.8958 | 0.2857 | 0.2105 | 0.4444 |
| boosting | train OOF | 0.2611 | 0.8770 | 0.3826 | 0.3143 | 0.4889 |
| boosting | val | 0.6484 | 0.9912 | 0.6154 | 0.4706 | 0.8889 |

The best model is boosting. It exceeds both null references and the
pre-declared val F1 floor of `0.50`.

## Gate

```text
train Tier A trigger recall >= 0.95:       pass (1.000)
val Tier A trigger recall >= 0.95:         pass (1.000)
best val AUPRC above prevalence:            pass
best val F1 above all-positive F1:          pass
best val F1 >= 0.50:                       pass (0.6154)
revised online pilot overall:               PASS
official test available:                    false
final benchmark ready:                      false
```

## Reproduction

```text
summary structure differences:       0
boosting score max difference:       1.11e-16
logistic score max difference:       2.60e-4
reported metric differences:         0 after rounding
```

The logistic difference is environment-specific solver arithmetic.

## Interpretation

An online pipeline now exists:

```text
stable contact -> release trigger -> pre-release quality classifier
```

It recalls every Tier A release and, after model filtering, reaches
val precision `0.4706` and recall `0.8889`.

The model still has a large train-to-val gap. With only one accessible
val participant and distance-only features, this is a pilot result, not
final evidence.

## Decision

The revised online detector passes. The next step is to add pre-release
hand-root and object-motion features, evaluate across participant
leave-one-out folds, and check that the gain is not specific to val
subject `s05`.

Human handover labels and official ARCTIC test performance remain
unsupported.

## Implementation Hashes

```text
scripts/build_arctic_online_release_windows.py
2f8e284d816f4aee5d82d8551556395c6e9009fde9f831003d65c4c362aacb64

scripts/train_arctic_role_switch_candidates.py
2739a4b407cad50f60f27e9acad9bdfc63a72a1a1552e2e8b56b39f1612b102c
```
