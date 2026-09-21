# ARCTIC Online Role-Switch Detector Result

Date: 2026-09-21

Status: online pilot gate FAIL

Protocol:

```text
docs/experiments/arctic-online-role-switch-detector-protocol-2026-09-21.md
```

## Trigger

The online rule requires:

```text
15-frame stable contact before release
release followed by five consecutive raw non-contact frames
```

It never reads the receiving hand's future onset.

At `3 mm`:

| Split | Tier A Events | Triggered | Recall |
|---|---:|---:|---:|
| train | 45 | 36 | 0.800 |
| val | 9 | 7 | 0.778 |

The pre-declared `0.90` recall gate fails.

## Trigger Failure Mechanism

The nine train and two val misses are not code errors. Their outgoing
hand briefly re-contacts the object one to four frames after the Tier A
release boundary. They are therefore not clean online release events
under the five-frame confirmation rule.

This exposes a real difference between:

```text
Tier A retrospective quality: allows a short raw re-contact
online release confirmation:  rejects that re-contact
```

## Online-Clean Subset

After excluding those trigger-incompatible positives:

```text
train: 36 Tier A positives, 2792 confirmed non-Tier-A releases
val:    7 Tier A positives,  366 confirmed non-Tier-A releases
```

The trigger recall for this online-clean subset is effectively `1.00`.

## Model Metrics

| Model | Split | AUPRC | AUROC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| logistic | train OOF | 0.1689 | 0.7859 | 0.2651 | 0.2340 | 0.3056 |
| logistic | val | 0.3054 | 0.9426 | 0.2105 | 0.1667 | 0.2857 |
| boosting | train OOF | 0.2153 | 0.8702 | 0.3226 | 0.3846 | 0.2778 |
| boosting | val | 0.6093 | 0.9879 | 0.4286 | 0.4286 | 0.4286 |

References:

```text
val prevalence:       0.0188
val all-positive F1:  0.0368
required val F1:      0.5000
```

Boosting clearly improves ranking and precision, but its val F1 is
`0.4286`, below the pre-declared `0.50` threshold. The online-clean
pilot gate therefore also fails.

## Reproduction

The trigger, window dataset, and both models were reproduced on the
server.

```text
summary structure differences:       0
boosting score max difference:       2.78e-17
logistic score max difference:       5.21e-5
reported metric differences:         0 after rounding
```

## Decision

The current distance-only model is not sufficient for the online
role-switch gate. It is a useful ranking signal but does not satisfy the
event-level F1 requirement.

Do not weaken the frozen threshold. The next justified step is a
motion-aware online model that adds pre-release hand-root and object
velocity/acceleration features, because the event review already showed
strong object motion and hand retreat in Tier A events.

Official ARCTIC test remains unavailable, so the next model remains a
participant-disjoint pilot.

## Implementation Hashes

```text
scripts/build_arctic_online_release_windows.py
2f8e284d816f4aee5d82d8551556395c6e9009fde9f831003d65c4c362aacb64

tests/test_build_arctic_online_release_windows.py
7e0bdb636af559effb1436b7ada0eee5fe9f66ffd56b7084003e22bd4df36992
```
