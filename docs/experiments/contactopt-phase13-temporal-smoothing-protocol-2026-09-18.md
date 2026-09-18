# ContactOpt Temporal Smoothing Protocol: Phase 13

Date: 2026-09-18

Status: completed with partial failure

Result report:

```text
docs/experiments/contactopt-phase13-temporal-smoothing-2026-09-18.md
```

## Trigger

Phase 12 found that independent frame-wise ContactOpt preserves contact but
greatly amplifies acceleration and jerk. This phase tests the smallest
post-processing fix before introducing any learned temporal model.

## Data

Use the four dense windows frozen in:

```text
docs/experiments/contactopt_phase12_manifest.json
SHA-256 e453cc9a7581c4e6fc9903c79028c015ec3578ea8f4866aa35c21854b7c92fec
```

This is a development set. A pass does not count as independent validation
because the smoothing design was chosen after observing Phase 12.

## Frozen Method

For each sequence:

1. keep the ContactOpt global pose coefficients `0:3` unchanged;
2. smooth the ContactOpt-optimized output finger PCA coefficients `3:18`
   over time with the fixed kernel `[1, 2, 1] / 4` and edge replication;
3. keep each frame's `hand_mTc` unchanged;
4. rerun the MANO forward model to obtain the smoothed hand vertices and
   joints;
5. recompute ContactOpt capsule contact and nearest hand-object distance.

No coefficient grid search, learned model, per-case kernel, or contact-specific
adjustment is allowed in this phase.

## Gate

Apply the Phase 12 contact and temporal thresholds unchanged:

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
4/4 passes: temporal smoothing survives the development gate
3/4 passes: CONDITIONAL, inspect the remaining window
0-2/4 passes: reject this smoothing arm
```

## Required Follow-Up

Even a 4/4 development pass only authorizes a held-out temporal validation on
windows that were not used to design the smoother. It does not authorize WM
training or claim temporal generation.
