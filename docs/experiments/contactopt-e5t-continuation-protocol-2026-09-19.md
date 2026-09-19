# E5-T Temporal Continuation Protocol

Date: 2026-09-19

## Goal

Continue from the completed E5 sequence solver and further reduce
acceleration/jerk failures without regressing contact, wrist, or object
state.

This is a second-stage per-segment optimization, not world-model training.

## Frozen Inputs

Use the E5 optimized poses:

```text
/root/autodl-tmp/contact_action_20260914/e5_solver_v1_20260919/optimized
```

Use the same train/dev sequence-disjoint manifest and the same 376 windows.
The 28 test windows remain untouched.

## Initialization

For every contiguous frame segment:

1. load the E5 optimized pose coefficients;
2. optimize a residual on finger coefficients `3:18`;
3. keep global coefficients `0:3` and `hand_mTc` fixed.

## Objective

```text
L =
    0.2   * smooth_l1(hand_vertices, ContactOpt_vertices)
  + 0.05  * mean(delta_pose^2)
  + 2.0   * mean(acceleration)
  + 2.0   * mean(jerk)
  + 8.0   * (top_10pct(acceleration) + top_10pct(jerk))
  + 10.0  * mean(relu(current_distance - baseline_distance - 1 mm)^2)
```

Object distances use 4,096 deterministic samples per frame.

## Optimization

```text
optimizer: Adam
learning rate: 0.003
iterations: 150
initial pose: E5 optimized pose
initial smoothing: none
seed: 20260919
```

## Frozen Gates

The E5 gates are unchanged:

```text
speed ratio          <= 2.0
acceleration ratio   <= 2.0
jerk ratio           <= 3.0
contact improvement  >= max(3, ceil(0.7 * eligible_frames))
mean distance change <= +10%
wrist drift          <= 1e-5 m
object drift         <= 1e-6 m
```

E5-T is promoted only if all of the following hold:

1. combined passes are higher than E5;
2. contact passes are not lower than E5;
3. the dominant `floorlamp` temporal failures improve;
4. no test split data is read.

## Train Pilot

Three train sequences containing E5 temporal failures:

```text
sub17_floorlamp_020
sub16_largebox_014
sub16_clothesstand_002
```

Results on their 24 windows:

| Arm | Combined | Contact | Temporal |
|---|---:|---:|---:|
| raw ContactOpt | 5 | 23 | not used |
| E5 | 14 | 23 | 15 |
| E5-T | 18 | 23 | 19 |

`floorlamp_020` improved from `2/8` to `4/8` combined. Contact did not
regress in the pilot.

## Full Run

Run all 49 chunks and 376 train/dev windows. The test split remains unread.
