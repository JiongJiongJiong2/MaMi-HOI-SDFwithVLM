# DWM P0 Failure Analysis

Date: 2026-09-22

Status: diagnostic analysis of the completed P0 `NO-GO`

## Summary

P0 failed for two different reasons across two runs. The first run exposed a
model-interface bug: the candidate scorer did not receive the task target.
After conditioning on the target, top-1 improved from `5.74%` to `10.28%`
and regret improved against every baseline. The revised model still failed
the frozen top-1 gate because it did not learn the strong directional
inductive bias available to the geometry-only heuristic.

The remaining failure is not a probe-identifiability failure: giving the
model ground-truth object mass, friction, and shape in the oracle-context
arm did not improve top-1 (`9.91%`), and the absolute object-delta arm was
only slightly better (`11.30%`). Probe context improves average decision
quality, but not exact candidate identification.

## Run 1: Missing Target Condition

The original P0 scorer received state, probe context, and candidate action,
but not `target_translation`. Candidate ranking is target-dependent, so one
target-free scalar score cannot represent the correct order across resets.

The frozen run produced:

| Arm | Top-1 | Regret |
|---|---:|---:|
| listwise probe | 0.0574 | 0.002322 |
| geometry-only | 0.2306 | 0.002975 |
| no-probe | 0.0657 | 0.004381 |
| absolute object-delta | 0.1083 | 0.003561 |
| quotient pairwise | 0.0361 | 0.001744 |
| oracle response context | 0.0889 | 0.002384 |

The oracle-context result is decisive: access to the true physical response
context did not repair ranking. This ruled out "probe cannot identify the
object" as the primary explanation.

## Run 2: Target-Conditioned Model

The model now receives the normalized target translation. All other data,
splits, seeds, baselines, and promotion thresholds are unchanged.

| Arm | Top-1 | Regret |
|---|---:|---:|
| listwise probe | 0.1028 | 0.002126 |
| geometry-only | 0.2306 | 0.002975 |
| no-probe | 0.0713 | 0.004461 |
| absolute object-delta | 0.1130 | 0.003667 |
| quotient pairwise | 0.0630 | 0.001890 |
| oracle response context | 0.0991 | 0.002448 |

Target conditioning is therefore a real required fix. Relative to the
no-probe model, the revised P0 model improves top-1 by `3.15` percentage
points, reduces regret by `52.35%`, and has a positive paired regret
bootstrap interval. It still loses to geometry-only on top-1 by `12.78`
percentage points, and the top-1 bootstrap interval is negative.

## Why It Still Fails

### 1. Geometry is a strong prior for this task

Candidate actions are wrist and finger position targets, and the evaluation
target is object translation. The geometry-only baseline directly compares
the candidate wrist displacement with the target direction. In a rigid
push task, this direction provides most of the exact-winner signal.

The learned scorer receives raw state and action vectors but is not built to
copy or refine this directional comparison. It can learn average contact
response, but it does not reliably preserve the target-direction ordering.

### 2. Ranking loss improved regret, not exact top-1

The listwise and quotient losses optimize the whole candidate distribution.
They reduce distance to the oracle utility, but exact top-1 is a discrete
and brittle metric. The model often ranks a near-optimal branch above a
weak one without selecting the single best branch.

This explains the otherwise unusual result:

```text
regret improves substantially
top-1 remains below geometry-only
```

### 3. The oracle response context does not improve ranking

The oracle-context arm knows mass, friction, and shape, but its top-1 is
only `9.91%`, similar to the target-conditioned probe model. The bottleneck
is therefore the candidate-effect scorer, not probe-based inference of the
physical response.

### 4. Unseen response changes exact winners

The test split changes mass and friction. Average response trends transfer
well enough to reduce regret, but the exact argmax across 13 contact-rich
branches is unstable. Near-tied candidates and contact-mode switches make
top-1 sensitive to small dynamics errors.

## Required Next Design

The next version should not be another probe-context variant. It should
preserve the geometry prior and learn only the residual decision effect:

1. use the geometry score as an explicit input or additive baseline;
2. predict a target-conditioned correction to that score;
3. train with a top-1-aware pairwise margin in addition to regret loss;
4. compare against a geometry-plus-learned-residual model with matched
   parameters and data.

This is a new method hypothesis. It is not a repair of the frozen P0 gate.

## Decision

```text
D0-C data gate:             GO
P0 target-free model:       NO-GO
P0 target-conditioned model: NO-GO
P1 active probe:            not authorized
MaMi integration:           not authorized
```

The frozen D line stops at P0. Any continuation must use a new protocol and
must explicitly beat geometry-only on top-1, not only reduce average regret.
