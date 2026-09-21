# ARCTIC Motion-Aware Release Detector Result

Date: 2026-09-21

Status: pilot frozen-comparison PASS; participant robustness mixed

Protocol:

```text
docs/experiments/arctic-motion-release-detector-protocol-2026-09-21.md
```

## Dataset

The instant online trigger and labels remain unchanged:

```text
train: 3,637 release events, 45 Tier A positives
val:     476 release events,  9 Tier A positives
trigger recall: 1.000 train, 1.000 val
```

Feature shape:

```text
30 frames x 16 features
```

## Metrics

| Model | Split | AUPRC | AUROC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| logistic | train OOF | 0.1472 | 0.7463 | 0.2083 | 0.1515 | 0.3333 |
| logistic | val | 0.2527 | 0.8558 | 0.4800 | 0.3750 | 0.6667 |
| boosting | train OOF | 0.2758 | 0.8911 | 0.4337 | 0.4737 | 0.4000 |
| boosting | val | 0.7170 | 0.9848 | 0.6316 | 0.6000 | 0.6667 |

## Frozen Comparison

| Metric | Distance Only | Motion Aware | Change |
|---|---:|---:|---:|
| train OOF AUPRC | 0.2611 | 0.2758 | +0.0147 |
| val AUPRC | 0.6484 | 0.7170 | +0.0685 |
| val F1 | 0.6154 | 0.6316 | +0.0162 |

All four frozen motion gates pass.

## Participant OOF Stability

Boosting out-of-fold AUPRC by training participant:

| Participant | Positives | Distance Only | Motion Aware | Change |
|---|---:|---:|---:|---:|
| s01 | 5 | 0.5590 | 0.6263 | +0.0673 |
| s02 | 3 | 0.0538 | 0.2513 | +0.1975 |
| s04 | 4 | 0.0784 | 0.0650 | -0.0134 |
| s06 | 10 | 0.2514 | 0.3245 | +0.0731 |
| s07 | 3 | 0.4009 | 0.3403 | -0.0607 |
| s08 | 9 | 0.3454 | 0.2880 | -0.0574 |
| s09 | 2 | 0.6667 | 0.7500 | +0.0833 |
| s10 | 9 | 0.3195 | 0.3177 | -0.0018 |

Motion features improve 4 of 8 participants. Mean AP change is `+0.0360`,
but gains are concentrated in a subset rather than uniformly robust.

## Gate

```text
trigger recall train/val >= 0.95:          pass
val AUPRC >= 0.6484:                       pass
val F1 >= 0.6154:                          pass
train OOF AUPRC > 0.2611:                  pass
motion-aware pilot overall:                PASS
official test available:                   false
final benchmark ready:                     false
```

## Interpretation

Motion features improve the frozen global comparisons, especially on val
subject `s05`. However, the participant-level result is mixed and the
gain is not uniformly distributed across the eight accessible training
participants.

The result supports motion features as a pilot addition, not as a final
robustness claim.

## Decision

Keep the motion-aware model as the current pilot lead. Before adding
more capacity, test feature stability:

```text
leave-one-participant-out per participant;
feature-group ablations;
seed/fold sensitivity;
calibration of the release-quality score.
```

Human handover labels and official ARCTIC test performance remain
unsupported.

## Artifacts

Server:

```text
/root/autodl-tmp/arctic_motion_release_windows_v1_20260921
/root/autodl-tmp/arctic_motion_release_model_v1_20260921
```

## Implementation Hashes

```text
scripts/build_arctic_motion_release_windows.py
86524df5ce09861c5b731691dcce73f03decf7bb9ed009f2942da349f2754b9f

tests/test_build_arctic_motion_release_windows.py
03bda77b7059832ad7e4f7887b2a44d9797d51106d3ee831a46ab1f61ddabd0f
```
