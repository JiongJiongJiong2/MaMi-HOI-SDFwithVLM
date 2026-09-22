# E5-T Direction-Constrained Projection Protocol

Date: 2026-09-22

Status: frozen before the full train/dev run

Result:
`docs/experiments/contactopt-e5t-direction-projection-result-2026-09-22.md`

## Goal

Test whether constraining outward near-surface motion while retaining a
matched temporal objective preserves the E5-T acceleration and jerk gains
without the observed contact regression. This is a fixed optimization
comparison on the existing train/dev cohort, not world-model training.

The 28 test windows remain unread.

## Frozen Inputs

```text
batch:
/root/autodl-tmp/contact_action_20260914/sequence_contactopt_v1_20260919/cases/batch_summary.json
geometry:
/root/autodl-tmp/contact_action_20260914/sequence_contactopt_v1_20260919/geometry
E5:
/root/autodl-tmp/contact_action_20260914/e5_solver_v1_20260919
E5-T:
/root/autodl-tmp/contact_action_20260914/e5t_solver_v1_20260919
```

The frozen batch, E5 result, and E5-T result hashes are:

```text
cb0785954c7f04c4ee0dbee97ecab008f1b38c3e2ff61e7f21dd3c1b219bd07e8
4c218fc51073870f46420321d07216935ed13d8f1e9cf16abf6af5ccf5fce734b
46714e9460bb4d924745ef4dc66652c8e29c0789bcf94637bdebd33fa91bcb30d
```

The full run must contain 49 chunks and 376 train/dev windows.

## Fixed Arms

```text
ordinary_smoothing
contact_projection
unconstrained_continuation
strong_contact
direction_constrained
```

`ordinary_smoothing` applies the fixed binomial-5 kernel `[1,4,6,4,1]` to
finger coefficients `3:18` from E5. `contact_projection` applies the same
smoothing and then one damped normal projection. `unconstrained_continuation`
reuses the frozen E5-T solver path. `strong_contact` uses the same temporal
objective with contact weight `100`. `direction_constrained` adds the
near-surface outward penalty and contact projection.

## Matched Budgets

All optimized arms use:

```text
initialization:          E5 optimized poses
optimized coefficients:  3:18
frozen coefficients:      global pose 0:3 and hand_mTc
optimizer:                Adam
learning rate:            0.003
iterations:               150
object samples:           4,096 per frame
pose L2 cap:              1.8 per frame
seed:                     20260919
```

The pose cap covers the maximum frozen E5-to-E5-T per-frame L2 displacement
of `1.663` on the full train/dev cohort.

The temporal weights remain:

```text
0.2  * smooth_l1(hand_vertices, ContactOpt_vertices)
0.05 * mean(delta_pose^2)
2.0  * mean(acceleration)
2.0  * mean(jerk)
8.0  * (top_10pct(acceleration) + top_10pct(jerk))
```

Contact preservation uses:

```text
10.0 * mean(relu(distance - E5_distance - 1 mm)^2)
```

`strong_contact` changes only the contact term to weight `100`.

## Direction Constraint

For every E5 hand vertex within `2 cm` of the sampled object surface:

```text
normal = unit(E5_vertex - nearest_object_vertex)
outward = dot(candidate_vertex - E5_vertex, normal)
penalty = smooth_l1(relu(outward - 0.5 mm) * 1000, beta=1 mm)
```

The train smoke protocol fixes `normal_weight = 20`. This value is frozen
before reading the full train/dev results and will not be changed on dev.

## Contact Reprojection

After the direction-constrained optimization, a selected vertex must be:

1. within `2 cm` of the object at E5;
2. more than `0.5 mm` outward relative to E5;
3. more than `1 mm` farther from the object than E5.

For each selected frame, the solver computes the finger-pose Jacobian of the
selected normal displacements and solves:

```text
(J^T J + 1e-4 I) dp = J^T residual
```

The step is capped at L2 `0.1`. A four-step half-line search rejects a step
unless it strictly reduces normal excess.

Projection is reverted to the pre-projection candidate when it breaks a
previously passing temporal window or exceeds the pose cap.

## Gates

The E5 gates remain unchanged:

```text
speed ratio          <= 2.0
acceleration ratio   <= 2.0
jerk ratio           <= 3.0
contact improvement  >= max(3, ceil(0.7 * eligible_frames))
mean distance change <= +10%
wrist drift          <= 1e-5 m
object drift         <= 1e-6 m
```

The direction arm is promoted only if all acceptance conditions in the
preregistered plan hold on train and dev. No test split is read.

## Smoke Check

Train chunk `sub16_largebox_008_c000` was used only as a smoke check.

```text
unconstrained continuation vs frozen E5-T max error: 0.0
contact projection normal excess: 16.720 mm -> 3.483 mm
direction combined/contact/temporal: 8/8, 8/8, 8/8
```

The smoke chunk is not discriminative: ordinary smoothing and strong contact
also pass all eight windows. The full train/dev run is required for a
mechanism decision.

## Outputs

For each arm:

```text
optimized/<chunk_id>_optimized.npz
frames/<chunk_id>.json
result.json
run.log
```
