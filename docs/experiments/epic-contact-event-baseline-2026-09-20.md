# EPIC-Contact Strict Event Baseline Result

Date: 2026-09-20

Status: complete with strict learned-event gate PASS

Protocol:

```text
docs/experiments/epic-contact-event-baseline-protocol-2026-09-20.md
```

Input:

```text
docs/experiments/epic_contact_strict_frames_v1.jsonl.gz
SHA-256 d202ebbdd3c2d907eb0d607794cd6d4cfa7ffdc9f61796628d7a8f26abaa9e36
```

## Samples

The frozen participant-disjoint strict split yields:

| Split | Participants | Samples | Next Contact | Onset | Release |
|---|---:|---:|---:|---:|---:|
| train | 25 | 7,537 | 6,171 | 654 | 648 |
| dev | 3 | 3,405 | 2,612 | 343 | 357 |
| test | 3 | 3,963 | 3,224 | 370 | 364 |

Hard F1 thresholds were selected on dev and frozen before test
evaluation. No test labels or test scores were used for threshold
selection.

## Test Metrics

| Arm | Target | AUROC | AUPRC | F1 |
|---|---|---:|---:|---:|
| copy current | next contact | 0.6963 | 0.8839 | 0.8972 |
| distance linear | next contact | 0.6960 | 0.8746 | 0.9074 |
| learned logistic | next contact | 0.8245 | 0.9450 | 0.9105 |
| copy current | onset | 0.5000 | 0.0986 | 0.1708 |
| distance linear | onset | 0.1220 | 0.0584 | 0.1707 |
| learned logistic | onset | 0.9668 | 0.6841 | 0.7201 |
| copy current | release | 0.5000 | 0.0940 | 0.1682 |
| distance linear | release | 0.4468 | 0.0767 | 0.1674 |
| learned logistic | release | 0.8143 | 0.2972 | 0.3784 |

The learned arm improves next-contact AUPRC over both trivial arms.
Its next-contact F1 is only `0.0031` above distance linear, but the
paired AUPRC difference excludes zero. The learned arm therefore does
not rely only on copying the current state.

## Participant-Clustered Differences

95% bootstrap intervals for learned minus baseline:

| Target | Baseline | Metric | Mean Difference | 95% CI |
|---|---|---|---:|---:|
| next contact | copy current | AUPRC | 0.0598 | [0.0417, 0.0737] |
| next contact | copy current | F1 | 0.0133 | [0.0086, 0.0186] |
| next contact | distance linear | AUPRC | 0.0699 | [0.0650, 0.0721] |
| next contact | distance linear | F1 | 0.0031 | [-0.0007, 0.0055] |
| onset | copy current | AUPRC | 0.5907 | [0.5226, 0.6610] |
| onset | copy current | F1 | 0.5495 | [0.5029, 0.5855] |
| onset | distance linear | AUPRC | 0.6250 | [0.5647, 0.6723] |
| onset | distance linear | F1 | 0.5491 | [0.5033, 0.5856] |
| release | copy current | AUPRC | 0.2047 | [0.1505, 0.2359] |
| release | copy current | F1 | 0.2082 | [0.1847, 0.2367] |
| release | distance linear | AUPRC | 0.2211 | [0.1698, 0.2517] |
| release | distance linear | F1 | 0.2095 | [0.1841, 0.2379] |

## Reproduction

Training was repeated locally and on the server with the same input hash
and seed:

```text
server: 18 cores, 50 GB RAM, RTX 4090 D
python: 3.10.8
numpy:  1.24.1
```

The summaries have identical structure and identical metrics at
reported precision. Twelve floating-point values differ only in their
last bits, with maximum absolute difference `2.22e-16`. Across the
`3,963` test rows and `8,000` model-score cells, the maximum prediction
difference is `5.55e-16`.

Server output:

```text
/root/autodl-tmp/epic_contact_event_baselines_v1_20260920
```

## Gate

```text
onset AUPRC exceeds both trivial arms:       pass
release AUPRC exceeds both trivial arms:     pass
onset paired CI excludes zero:               pass
release paired CI excludes zero:             pass
next-contact AUPRC exceeds copy-current:     pass
next-contact AUPRC exceeds distance-linear:  pass
overall:                                     PASS
```

## Decision

The strict EPIC-Contact event route continues to a stronger sequence
model. The next model must retain the same participant-disjoint split,
dev-only threshold selection, participant-clustered paired intervals,
and the required `0.5-3 mm` threshold sensitivity curve.

This result supports short-horizon contact lifecycle prediction. It
does not establish handover, receiver-role switching, or cross-hand
object transfer; the C handover route remains NO-GO.

## Implementation Hashes

```text
scripts/train_epic_contact_event_baselines.py
476ec704887c57a4862d11c38e0aafe81dbf081260d8e52b231daaa5f45c7f0c

tests/test_train_epic_contact_event_baselines.py
e48de81c0e44406b7cfd00008b61331ea461bf653885a4df36b229d0fd846ad6
```
