# Sequence ContactOpt and Fixed Smoothing Result

Date: 2026-09-19

Status: complete with raw and fixed-smoothing NO-GO

## Scope

Run the frozen ContactOpt refinement on the train and dev portions of:

```text
docs/experiments/contactopt_sequence_dataset_v1_20260919.json
SHA-256 0605026647b1dabb44ee13723d16c97297524f7ffc2df3083665512eaf541054
```

The run covered 49 sequence chunks and 376 windows. The 28 test windows were
not read or refined.

Raw evidence:

```text
docs/experiments/contactopt_sequence_dataset_batch_2026-09-19.json
docs/experiments/contactopt_sequence_dataset_temporal_2026-09-19.json
docs/experiments/contactopt_sequence_dataset_smoothing_binomial5_2026-09-19.json
```

## Raw Frame-wise ContactOpt

All 49 chunks completed. The frozen contact gate passed for 364 of 376
windows, but the temporal gate passed for only 93 windows.

| Metric | Windows | Pass rate |
|---|---:|---:|
| Contact gate | 364 / 376 | 96.8% |
| Temporal gate | 93 / 376 | 24.7% |
| Contact + temporal | 92 / 376 | 24.5% |

Mean refined/input ratios:

```text
speed        1.386
acceleration 5.050
jerk         7.532
```

This is a clear NO-GO for direct independent frame-wise integration. Contact
improvement is nearly universal, but it comes with large frame-to-frame
excursions.

## Fixed Five-Tap Smoothing

The pre-existing `[1,4,6,4,1]/16` kernel was applied only to optimized finger
PCA coefficients. Global pose and `hand_mTc` remained unchanged.

| Split | Windows | Temporal pass | Contact pass | Combined pass |
|---|---:|---:|---:|---:|
| train | 256 | 229 | 246 | 220 |
| dev | 120 | 91 | 117 | 88 |
| all | 376 | 320 | 363 | 308 |

Mean ratios improve substantially:

```text
speed        1.088
acceleration 1.655
jerk         1.698
```

The remaining problem is the tail rather than the average:

```text
ratio percentiles      50%     90%     95%     99%     max
speed                 1.008   1.202   1.454   2.516   4.591
acceleration          1.237   2.530   3.675   7.343  19.279
jerk                  1.223   2.552   3.876   8.647  23.923
```

Of the 68 combined failures:

```text
contact-only failures   12
temporal-only failures  55
both                     1
```

The worst object is `floorlamp` with 14 combined failures out of 32 windows
and mean acceleration/jerk ratios `3.672/3.703`. `largebox`, `trashcan`, and
`whitechair` are substantially more stable.

## Decision

Fixed five-tap smoothing is a useful baseline but is not sufficient to freeze
as the temporal solution.

```text
raw ContactOpt integration:       NO-GO
fixed five-tap integration:       NO-GO
sequence-level temporal solver:   required
```

The next objective should target tail stability and contact preservation, not
just lower mean jerk. A useful solver must reduce the worst windows without
using the test split or trading away the 363/376 contact passes.

## Boundaries

This experiment uses right-hand geometry only. It does not recover MaMi
`pose_hand`, establish finger-level WM, change the object trajectory, or
demonstrate physical contact. The test split remains untouched and still has
the narrow object coverage documented in the dataset result.
