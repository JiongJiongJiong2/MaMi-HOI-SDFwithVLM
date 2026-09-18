# ContactOpt Stronger Temporal Smoothing Protocol: Phase 14

Date: 2026-09-18

Status: frozen before execution

## Trigger

The three-tap `[1, 2, 1] / 4` smoother in Phase 13 reduced acceleration and
jerk but passed only one of four development windows. This phase tests one
fixed wider support without searching multiple kernels.

## Frozen Method

Use the same four Phase 12 windows and the same optimized ContactOpt pkl files.
Replace only the temporal kernel with:

```text
[1, 4, 6, 4, 1] / 16
```

Smooth ContactOpt output finger PCA coefficients `3:18`, keep global pose
coefficients `0:3` and `hand_mTc` unchanged, rerun MANO forward geometry, and
recompute contact and temporal metrics.

No per-case kernel, adaptive support, grid search, or learned temporal model is
allowed.

## Gate

Use the Phase 12 promotion thresholds unchanged:

```text
contact-improved frames >= ceil(0.7 * eligible frames)
mean nearest-distance relative change <= +10%
maximum wrist-root drift <= 0.01 mm
maximum object drift <= 0.001 mm
refined/input speed ratio <= 2.0
refined/input acceleration ratio <= 2.0
refined/input jerk ratio <= 3.0
```

Decision:

```text
4/4 passes: stronger smoothing survives the development gate
3/4 passes: CONDITIONAL
0-2/4 passes: reject the fixed stronger smoother
```

Even a 4/4 result remains development data and requires a held-out temporal
validation before integration.
