# ContactOpt Temporal Smoothing Result: Phase 13

Date: 2026-09-18

Protocol:

```text
docs/experiments/contactopt-phase13-temporal-smoothing-protocol-2026-09-18.md
```

## Method

The fixed `[1, 2, 1] / 4` kernel was applied to ContactOpt output finger PCA
coefficients `3:18`. Global pose coefficients `0:3` and `hand_mTc` were kept
unchanged, followed by a fresh MANO forward pass. Contact and temporal metrics
were recomputed on the same four Phase 12 windows.

## Results

| Case | Raw accel ratio | Smoothed accel ratio | Raw jerk ratio | Smoothed jerk ratio | Smoothed contact change | Promotion |
|---|---:|---:|---:|---:|---:|---|
| monitor | 5.59 | 1.73 | 7.16 | 1.56 | +43.8% | PASS |
| largetable | 5.64 | 2.70 | 7.77 | 2.85 | +56.3% | FAIL |
| plasticbox | 8.12 | 3.14 | 11.71 | 3.80 | +15.3% | FAIL |
| smallbox | 11.54 | 3.73 | 19.92 | 5.53 | +69.5% | FAIL |

Aggregate:

```text
temporal passes:        1/4
contact+temporal pass:  1/4
```

The smoother reduced acceleration and jerk by roughly a factor of two to four
and preserved substantial contact gain. It did not bring three windows below
the frozen temporal thresholds.

## Decision

The three-tap smoothing arm is rejected as the final integration method. It is
useful evidence that smoothing the output PCA parameters reduces jitter without
destroying contact, but stronger temporal regularization is required before
integration.

## Next Step

Test one stronger pre-registered smoother with a wider fixed support. Do not
perform a per-case kernel search.
