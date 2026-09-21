# ARCTIC Motion Feature Stability Result

Date: 2026-09-21

Status: stability gate FAIL; distance-only remains more robust

Protocol:

```text
docs/experiments/arctic-motion-feature-stability-protocol-2026-09-21.md
```

## Primary-Seed Ablation

At seed `20260921`:

| Feature Set | Train OOF AUPRC | Val AUPRC | Val F1 |
|---|---:|---:|---:|
| distance only | 0.3004 | 0.6484 | 0.6667 |
| motion only | 0.0222 | 0.1311 | 0.1022 |
| all features | 0.2779 | 0.7170 | 0.6316 |
| all minus hand speed | 0.2972 | 0.6321 | 0.6000 |
| all minus relative speed | 0.2927 | 0.7046 | 0.5556 |
| all minus object motion | 0.3059 | 0.6334 | 0.5556 |

The motion-only features are weak by themselves.

On train OOF, distance-only or all-minus-object-motion outperform the
full all-feature vector. The strong all-feature val AUPRC is therefore
not accompanied by a stable train-side advantage.

## Seed Sensitivity

Five participant-balanced fold seeds were tested:

| Feature Set | Mean OOF AUPRC | Std | Mean Val AUPRC | Std | Mean Val F1 | Std |
|---|---:|---:|---:|---:|---:|---:|
| distance only | 0.2492 | 0.0291 | 0.6484 | 0.0000 | 0.5559 | 0.1144 |
| all features | 0.2612 | 0.0205 | 0.7170 | 0.0000 | 0.5469 | 0.0589 |

Val AUPRC is unchanged across seeds because the final model and val
scores do not change; only the OOF-selected F1 threshold varies.

All features exceed distance-only OOF AUPRC in four of five seeds, but
the selected hard threshold is unstable. The mean val F1 is `0.5469`,
below the frozen requirement `0.6154`.

## Calibration

Primary all-feature model:

| Score | Brier | ECE |
|---|---:|---:|
| raw | 0.0119 | 0.0122 |
| isotonic | 0.0104 | 0.0045 |
| Platt | 0.0128 | 0.0021 |

Calibration improves the descriptive error, but does not change the
frozen classification gate.

## Gate

```text
mean all-feature OOF > distance-only:          pass
all-feature OOF better in >= 4 of 5 seeds:     pass
mean all-feature val AUPRC >= 0.6484:          pass
mean all-feature val F1 >= 0.6154:             fail
stable motion overall:                         FAIL
official test available:                      false
final benchmark ready:                        false
```

## Decision

Do not promote the motion-aware model as the robust pilot lead. The
earlier single-seed improvement is real for ranking on val, but the
hard-threshold F1 gain is not stable across participant folds.

Keep the distance-only instant-release model as the more conservative
pilot baseline until additional participant data can support a stable
feature decision.

The next data decision should target additional participant coverage,
not more model capacity or threshold tuning.

## Artifacts

Server:

```text
/root/autodl-tmp/arctic_motion_feature_stability_v1_20260921
```

## Implementation Hashes

```text
scripts/analyze_arctic_motion_feature_stability.py
2dd86c40691957e93b38d4a6f65baea38ceb4b7f8c1e475bcd363b3fca3d8b38

tests/test_analyze_arctic_motion_feature_stability.py
cde54cbdf1759150bf4e9031807ed9eeab2660ece770d4d6cd376b8d554a63df
```
