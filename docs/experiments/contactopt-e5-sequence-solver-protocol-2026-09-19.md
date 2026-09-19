# E5 Sequence-Level ContactOpt Solver Protocol

Date: 2026-09-19

## Goal

Reduce acceleration and jerk tails in ContactOpt hand trajectories while
preserving its contact improvement and freezing wrist/object state.

This is a per-segment optimization, not a world-model training run.

## Frozen Inputs

Use the train/dev portion of:

```text
docs/experiments/contactopt_sequence_dataset_v1_20260919.json
SHA-256 0605026647b1dabb44ee13723d16c97297524f7ffc2df3083665512eaf541054
```

Batch outputs:

```text
/root/autodl-tmp/contact_action_20260914/sequence_contactopt_v1_20260919
```

The 28 test windows remain untouched.

## Initialization

For each contiguous frame segment:

1. load ContactOpt output pose coefficients;
2. apply the fixed binomial-5 smoother to finger coefficients `3:18`;
3. optimize a residual on those coefficients.

Global coefficients `0:3` and `hand_mTc` remain fixed.

## Objective

```text
L =
    0.2   * smooth_l1(hand_vertices, ContactOpt_vertices)
  + 0.05  * mean(delta_pose^2)
  + 1.0   * mean(acceleration)
  + 1.0   * mean(jerk)
  + 2.0   * (top_10pct(acceleration) + top_10pct(jerk))
  + 10.0  * mean(relu(current_distance - baseline_distance - 1 mm)^2)
```

Distances use 2,048 deterministic object-surface samples per frame. Final
contact metrics are recomputed from the complete ContactOpt capsule/metric
path.

## Optimization

```text
optimizer: Adam
learning rate: 0.003
iterations: 100
seed: 20260919
```

## Frozen Gates

Per eight-frame window:

```text
speed ratio          <= 2.0
acceleration ratio   <= 2.0
jerk ratio           <= 3.0
contact improvement  >= max(3, ceil(0.7 * eligible_frames))
mean distance change <= +10%
wrist drift          <= 1e-5 m
object drift         <= 1e-6 m
```

All four conditions must hold for a combined pass. No threshold may be
changed after seeing the full train/dev result.

## Development Pilot

Four dev sequences were used only to verify that the solver changes the
intended behavior:

```text
sub16_clothesstand_001
sub16_largebox_005
sub17_floorlamp_004
sub16_whitechair_012
```

Result:

```text
raw combined:       10/32
optimized combined: 28/32
contact passes:     32/32
```

`floorlamp` remains the worst case at `4/8` combined and mean acceleration
ratio `3.754`. This pilot is not the final result and does not authorize test
evaluation.

## Full Train/Dev Run

After this protocol is frozen, run all 49 chunks and 376 train/dev windows.
The test split remains unread.
