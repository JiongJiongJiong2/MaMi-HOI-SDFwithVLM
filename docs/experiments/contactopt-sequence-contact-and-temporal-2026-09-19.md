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
docs/experiments/contactopt_sequence_dataset_smoothing_binomial5_segments_summary_2026-09-19.json
```

The segment-wise correction completed on the server:

```text
/root/autodl-tmp/contact_action_20260914/sequence_contactopt_v1_20260919/analysis/per_window_smoothing_binomial5_segments.json
```

The earlier smoothing file is retained as history only. It smoothed across
four-frame gaps between windows and is superseded by the segment-wise result.

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
| train | 256 | 229 | 245 | 220 |
| dev | 120 | 95 | 117 | 92 |
| all | 376 | 324 | 362 | 312 |

Mean ratios improve substantially:

```text
speed        1.090
acceleration 1.635
jerk         1.679
```

The remaining problem is the tail rather than the average:

```text
ratio percentiles      50%     90%     95%     99%     max
speed                 1.008   1.234   1.417   2.673   4.840
acceleration          1.216   2.564   3.603   7.078  22.392
jerk                  1.195   2.590   3.586   9.132  23.266
```

Of the 64 combined failures:

```text
contact-only failures   12
temporal-only failures  50
both                     2
```

The worst object is `floorlamp` with 15 combined failures out of 32 windows
and mean acceleration/jerk ratios `3.827/3.738`. `largebox`, `trashcan`, and
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
