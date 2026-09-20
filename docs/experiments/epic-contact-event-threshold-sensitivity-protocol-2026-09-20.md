# EPIC-Contact Strict Event Threshold Sensitivity Protocol

Date: 2026-09-20

Status: frozen before execution

## Goal

Test whether the strict learned-event result is specific to the
sensitivity-selected `1 mm` contact definition.

This is a sensitivity analysis. It does not replace the official
`3 mm` epistemic protocol and does not create a new primary threshold.

## Frozen Inputs

```text
docs/experiments/epic_contact_strict_frames_v1.jsonl.gz
SHA-256
d202ebbdd3c2d907eb0d607794cd6d4cfa7ffdc9f61796628d7a8f26abaa9e36
```

The participant-disjoint split stored in each frame row is fixed. No
split is recomputed at a new threshold.

## Thresholds

The curve is evaluated at:

```text
0.5 mm
1.0 mm
1.5 mm
2.0 mm
3.0 mm
```

For each threshold, the existing history, feature vector, logistic
optimizer, threshold-selection rule, and metrics are reused. Only the
contact labels and distance normalization change.

The two trivial arms remain:

```text
copy-current-state
distance-linear
```

## Primary Sensitivity Gate

The strict B route survives the sensitivity check only if:

1. the `1 mm` logistic test metrics reproduce the frozen baseline;
2. at both `0.5 mm` and `1.5 mm`, learned onset AUPRC exceeds the best
   trivial arm;
3. at both `0.5 mm` and `1.5 mm`, learned release AUPRC exceeds the
   best trivial arm.

The `2 mm` and `3 mm` points are reported for the full requested curve,
but their event counts are too low to serve as independent pass/fail
criteria.

## Decision Rule

Failure of the primary gate stops the strict B route and moves data work
to ARCTIC/GRAB. Passing establishes only threshold robustness for a
short-horizon lifecycle diagnostic; it does not establish handover.
