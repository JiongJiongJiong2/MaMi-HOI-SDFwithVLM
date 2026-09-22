# E5-T Direction-Constrained Projection Result

Date: 2026-09-22

Status: complete

Protocol:

```text
docs/experiments/contactopt-e5t-direction-constrained-projection-protocol-2026-09-22.md
```

Compact result:

```text
docs/experiments/e5t_direction_projection_v1_20260922.json
```

Server analysis:

```text
/root/autodl-tmp/contact_action_20260914/e5t_direction_projection_v1_20260922/analysis.json
```

## Scope

The fixed arms were run on all 49 train/dev chunks and 376 windows. The 28
test windows were not read. All arms kept global pose and `hand_mTc` frozen,
optimized only finger coefficients `3:18`, and used the same iterations,
learning rate, seed, pose cap, and evaluation windows.

The MANO reconstruction maximum error was `2.38e-7 m`, below the frozen
`1e-6 m` gate. Maximum wrist drift and object drift were both zero for every
arm. The window set matched E5 exactly.

`unconstrained_continuation` reproduced the frozen E5-T pass decisions on
all 376 windows: combined, contact, and temporal pass differences were `0`.
The maximum pose-coefficient difference was `0.0101` from parallel-GPU
numeric variation, with maximum mean acceleration-ratio difference `8.24e-4`
and jerk-ratio difference `1.96e-3`. The isolated smoke run was exactly
`0.0`.

## Aggregate

| Arm | Combined | Contact | Temporal | Accel ratio | Jerk ratio | Distance vs raw |
|---|---:|---:|---:|---:|---:|---:|
| E5 | 343 | 362 | 357 | 1.180 | 0.991 | 0.728 |
| E5-T | 341 | 350 | 367 | 0.955 | 0.784 | 0.740 |
| ordinary smoothing | 347 | 361 | 362 | 1.148 | 1.092 | 0.729 |
| contact projection | 347 | 361 | 362 | 1.147 | 1.090 | 0.729 |
| strong contact | 348 | 357 | 367 | 0.959 | 0.787 | 0.733 |
| unconstrained continuation | 341 | 350 | 367 | 0.955 | 0.784 | 0.740 |
| direction constrained | 351 | 362 | 365 | 1.019 | 0.834 | 0.726 |

The direction arm gained 12 combined windows over E5-T and lost two. The two
losses were both temporal-gate losses:

```text
sub16_largetable_000_c000 frames 63:70
sub17_floorlamp_020_c000 frames 32:39
```

Direction-constrained train/dev counts:

| Split | Combined | Contact | Temporal |
|---|---:|---:|---:|
| train | 239 | 245 | 250 |
| dev | 112 | 117 | 115 |

The corresponding E5-T counts were `231/236/251` on train and
`110/114/116` on dev.

## Projection

Direction-constrained optimization already prevented most outward drift.
Contact reprojection selected five segments and six frames. It reduced the
summed near-surface normal excess from `6.828 mm` to `1.765 mm` before
contact recomputation. No projection was reverted by the temporal-gate or
pose-budget fallback.

## Compared With Controls

Paired sequence-clustered bootstrap differences in combined pass count:

| Control | Train mean [95% CI] | Dev mean [95% CI] | All mean [95% CI] |
|---|---:|---:|---:|
| ordinary smoothing | +3 [-2, +9] | +1 [-2, +4] | +4 [-2, +11] |
| contact projection | +3 [-2, +10] | +1 [-2, +4] | +4 [-2, +11] |
| strong contact | +1 [-4, +7] | +2 [-2, +7] | +3 [-4, +10] |

No control interval had a lower bound above zero. Ordinary smoothing and
contact projection also increased combined passes over E5-T, while strong
contact removed nearly all contact regressions at E5-T temporal levels.

## Acceptance

| Condition | Result |
|---|---|
| Combined above E5-T on train and dev | pass (`+8`, `+2`) |
| Contact not below E5-T and within 12 of E5 | pass (`362` vs E5 `362`) |
| Temporal not below E5-T | fail (`365` vs `367`) |
| One control bootstrap lower bound strictly positive | fail |
| Acceleration and jerk not above E5-T | fail (`1.019` vs `0.955`; `0.834` vs `0.784`) |
| Distance within 10% of E5 | pass (`0.998x`) |
| Test unread | pass |

## Decision

```text
direction mechanism: NO-GO
promotion: NO-GO
engineering status: contact regression removed with a small temporal/motion
                    trade, but not separated from the fixed controls
dev retuning: not performed
```

The direction constraint removed the E5-to-E5-T contact regression and
recovered the full E5 contact pass count, but it did not preserve the E5-T
temporal budget and its combined-pass improvement over the matched controls
was not statistically separated. The experiment therefore does not support
the claim that outward-normal control is the main mechanism behind the
combined improvement.

The frozen `normal_weight = 20` is not changed after inspecting dev. A is
retained as a bounded engineering result rather than a paper mechanism. The
separate B lifetime/switching direction is not automatically authorized by
this result because the contact-regression trigger was removed; it requires
its own protocol.

## Implementation Hashes

```text
scripts/optimize_contactopt_direction_projection.py
49d43bb1abec23fa92e1414d319466de65754a949574484b509ffef531650472

scripts/analyze_contactopt_direction_projection.py
097e3a720be7a9a527391c383f9cf0f034b6ca9bc1d0c4c1f6bb5d11f5af1c0d

docs/experiments/e5t_direction_projection_v1_20260922.json
3339c6a1da3a673606df22c181295eaf760eb39a84023c4274aa6214228593ee
```
