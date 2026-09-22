# DWM P0-R Geometry-Residual Protocol

Date: 2026-09-22

Status: frozen before implementation

## Goal

Test whether a learned decision residual can improve on the strong
geometry-only ranking prior for the completed D0-C dataset.

This is a new post-P0 experiment. It does not reopen the target-free P0
run and does not authorize P1 until it passes.

## Motivation

The target-free P0 model failed because candidate ranking is
target-dependent. The target-conditioned P0 model improved top-1 and regret,
but still lost to geometry-only on exact top-1. Oracle physical context did
not repair ranking. The next hypothesis is that geometry already supplies
the dominant direction prior, and the learned model should predict only a
correction to that score.

## Model

The P0-R model receives the same inputs as the target-conditioned P0 model
and additionally receives geometry features describing the candidate wrist
displacement, target error vector, and geometry logit.

```text
geometry_logit[B, K]
```

The geometry logit is:

```text
-L1(
    candidate wrist translation
    - hold wrist translation,
    target translation
 ) / 0.02
```

Candidate score is:

```text
candidate_score = geometry_logit + learned_residual_score
```

The residual score head is zero-initialized, so the model begins exactly at
the geometry baseline. The model also predicts centered object delta for
the auxiliary loss.

## Training

P0-R uses:

```text
listwise ranking loss
+ 0.5 * top-1 cross-entropy toward the oracle-best candidate
+ 0.1 * centered object-delta SmoothL1
```

Training settings remain:

```text
hidden size:       128
response latent:    16
seeds:              11, 23, 37
optimizer:          AdamW
learning rate:      1e-3
weight decay:       1e-4
batch groups:       64
maximum epochs:     100
early stopping:     15 on validation regret
```

## Baselines

P0-R is compared against:

```text
geometry-only
target-conditioned P0 listwise
no-probe
absolute object-delta
quotient pairwise
oracle response context
oracle utility
```

## Promotion Gate

The P0 gate is unchanged:

1. top-1 at least 10 percentage points above geometry-only;
2. top-1 at least 5 percentage points above no-probe;
3. regret at least 20% lower than geometry-only;
4. regret at least 10% lower than no-probe;
5. paired top-1 and regret bootstrap 95% CI lower bounds above zero;
6. all three seeds agree in direction;
7. slip and unintended release do not worsen by more than 2 percentage
   points.

Failure stops the D line before P1.

## Outputs

Result paths:

```text
docs/experiments/dwm-p0r-result-2026-09-22.md
docs/experiments/dwm-p0r-gate-2026-09-22.json
```

Server results should be written under:

```text
/root/autodl-tmp/dwm_p0r_results_v1_20260922
```
