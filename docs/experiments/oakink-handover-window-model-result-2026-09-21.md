# OakInk Handover Window Model Result

Date: 2026-09-21

Status: participant-disjoint handover-onset pilot PASS

Protocol:

```text
docs/experiments/oakink-handover-window-model-protocol-2026-09-21.md
```

## Data

```text
primary handover events: 106 / 126 retained sequences
history: 30 frames
features: 10 per frame
```

Window counts:

| Split | Sequences | Samples | Positives | Prevalence |
|---|---:|---:|---:|---:|
| train | 44 | 3,971 | 495 | 0.1247 |
| val | 41 | 3,418 | 540 | 0.1580 |
| test | 41 | 1,907 | 555 | 0.2910 |

Hard F1 thresholds were selected on participant-pair-grouped train OOF
scores. Val and test were not used for fitting or threshold selection.

## Metrics

| Model | Split | AUPRC | AUROC | F1 | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| logistic | train OOF | 0.5032 | 0.9265 | 0.6683 | 0.5574 | 0.8343 |
| logistic | val | 0.8010 | 0.9647 | 0.7993 | 0.7472 | 0.8593 |
| logistic | test | 0.9146 | 0.9442 | 0.8711 | 0.8908 | 0.8523 |
| boosting | train OOF | 0.7105 | 0.9514 | 0.6782 | 0.6448 | 0.7152 |
| boosting | val | 0.7672 | 0.9605 | 0.7045 | 0.7613 | 0.6556 |
| boosting | test | 0.9112 | 0.9527 | 0.7921 | 0.8791 | 0.7207 |

Logistic is the best model under the frozen val/test gates.

## Gate

```text
val AUPRC above prevalence:                 pass
val F1 above all-positive F1:               pass
test AUPRC above prevalence:                pass
test F1 above all-positive F1:              pass
test recall >= 0.50:                        pass (0.8523)
pilot overall:                              PASS
official OakInk benchmark claim:            false
final benchmark ready:                      false
```

## Boundary

All retained examples come from OakInk sequences already carrying the
handover intent label. The model is therefore a **handover-onset
predictor inside known handover sequences**, not a model that
distinguishes handover from use, hold, or lift-up.

The next required diagnostic is specificity: run the same window model
on all OakInk intents and measure whether it fires selectively around
handover transitions rather than around ordinary contact events.

## Artifacts

Server:

```text
/root/autodl-tmp/oakink_handover_windows_v1_20260921
/root/autodl-tmp/oakink_handover_model_v1_20260921
```

Local review copies:

```text
C:\Users\何炯乐\Documents\HOI项目\oakink_handover_model_v1_20260921\server_output
```

## Decision

OakInk supports participant-disjoint handover-onset modeling. The next
experiment is cross-intent specificity on OakInk `use`, `hold`,
`liftup`, and `handover`, with the same participants and model. This
determines whether the current signal is handover-specific or merely a
generic contact-transition detector.

## Implementation Hashes

```text
scripts/build_oakink_handover_windows.py
ad3d38ce033a5ad71e798ae7c9a51940890fab0b0149f225d13ff7fd6708fb79

scripts/train_arctic_role_switch_candidates.py
4ae315c71038d2aa4747ab7ecfd6d3cb8fb57308c0b06d7b175a2fb44fcca8c7

tests/test_build_oakink_handover_windows.py
e79cb5a0ac0b93a877561cb47196fe18bc2c5ef7d1749806dabd5e252df20e38

server summary
8fd76c09a272e5f2e76469b3e1ccb9f9d652c212ef43344f9c87f53a795602f1
```
