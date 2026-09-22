# DWM MaMi Contact Residual Reranker Result

Date: 2026-09-22

Status: complete with NO-GO for the new downstream reranker direction

## Purpose

This is the first downstream experiment for the redefined D direction:

```text
geometry-only candidate selection
vs geometry + learned residual reranker
vs base/random/oracle
```

It uses the existing saved MaMi/ContactOpt candidate pool rather than the
synthetic 13-branch D0-C ranking task.

## Data

The input was the combined validation cohorts:

```text
sequences 10-30
sequences 31-50
sequences 51-70
```

There are 78 hand contact events across 43 sequences. Each event has 12
candidate actions and saved geometry/contact proxy features. True
candidate contact F1 is used only as the held-out evaluation label, never
as a model input.

The reranker was trained with leave-one-sequence-out evaluation. The
geometry baseline is the saved frame-contact minus penetration score.

## Results

| Arm | Mean contact F1 | Mean predicted penetration |
|---|---:|---:|
| base | 0.4131 | 9.79 mm |
| random candidates | 0.3696 | 10.01 mm |
| geometry-only | 0.4909 | 7.36 mm |
| geometry + residual | 0.4830 | 7.07 mm |
| oracle candidate | 0.5451 | 10.48 mm |

The residual model improved penetration slightly but reduced mean contact
F1 by `0.72` percentage points relative to geometry-only. The sequence
bootstrap interval for residual minus geometry F1 was
`[-3.10, +0.69]` percentage points, so it does not support a positive F1
claim.

## Decision

```text
geometry + residual beats geometry on contact F1:   NO
residual is safe/noninferior overall:               PARTIAL
additional D world-model investment authorized:     NO
```

This confirms the earlier stopping rule: the current D formulation does not
justify continuing as a MaMi world-model decision module. The geometry
baseline remains useful, and the negative result is retained as a
diagnostic boundary.

## Artifact

The runnable analysis is:

```text
scripts/train_mami_contact_residual_reranker.py
```

The local result JSON is:

```text
C:/Users/何炯乐/Documents/HOI项目/dwm-literature-20260922/mami_residual_reranker_v1.json
```
