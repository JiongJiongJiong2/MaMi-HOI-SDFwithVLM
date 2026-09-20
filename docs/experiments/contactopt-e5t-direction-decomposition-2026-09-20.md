# E5-T Update Direction Decomposition Result

Date: 2026-09-20

Status: complete with GO for the direction hypothesis

Protocol:

```text
docs/experiments/contactopt-e5t-direction-decomposition-protocol-2026-09-20.md
```

Compact result:

```text
docs/experiments/e5t_direction_decomposition_v1_20260920.json
```

Server summary:

```text
/root/autodl-tmp/contact_action_20260914/e5t_direction_decomposition_v1_20260920/summary.json
```

## Scope

The CPU-only run reconstructed MANO vertices from the frozen E5 and E5-T
poses for all 49 train/dev chunks and all 376 windows. The 28 test windows
were not read.

The maximum MANO reconstruction error against the stored ContactOpt
vertices was `2.38e-7 m`, below the frozen `1e-6 m` gate.

For each E5 near-contact hand vertex, the E5-to-E5-T displacement was
decomposed using the unit vector from that vertex to its nearest object
vertex. Only vertices within `0.02 m` of the object contributed to the
primary statistics.

## Frozen Groups

| Group | Windows | Sequences |
|---|---:|---:|
| contact stable | 350 | 49 |
| contact regression | 12 | 10 |
| contact common failure | 14 | 8 |
| contact gain | 0 | 0 |

All 12 contact regressions are losses from E5 pass to E5-T fail.

## Primary Comparison

Values are millimeters per window, first averaged over its eight frames.

| Metric | Stable | Regression | Regression - stable | 95% clustered CI |
|---|---:|---:|---:|---:|
| signed normal | 0.309 | 2.075 | +1.766 | [+0.876, +2.603] |
| outward component | 0.986 | 2.969 | +1.983 | [+1.089, +2.772] |
| inward component | 0.677 | 0.894 | +0.218 | [-0.036, +0.459] |
| tangential motion | 2.887 | 4.673 | +1.785 | [+0.267, +3.899] |
| total motion | 3.565 | 6.619 | +3.053 | [+1.100, +5.320] |
| nearest-distance change | 0.603 | 3.228 | +2.625 | [+1.450, +3.751] |

The outward-vertex fraction is `0.538` for stable windows and `0.649` for
regressions, a difference of `+0.112` with interval
`[+0.052, +0.166]`.

The regressions occur on four object types:

```text
largebox    4
trashcan    4
plasticbox  3
smallbox    1
```

## Decision

```text
direction hypothesis: GO
```

E5-T contact regression is strongly associated with near-surface hand
motion away from the object. The outward fraction also increases, so the
effect is not explained only by a larger global displacement. However,
the regressions move substantially more in both normal and tangential
directions. Direction alone is not isolated yet.

This is a geometric association, not proof that outward motion causes the
contact-gate failures. The nearest-object-vertex normal is a proxy, not a
signed distance field or a physical contact model.

## Next Experiment

The next run must compare direction-constrained smoothing against:

1. ordinary smoothing;
2. nearest-surface contact projection;
3. a stronger contact-constraint objective;
4. an unconstrained E5-T continuation.

All arms must use the same temporal budget, pose-deviation cap, and
contact window set. A passing result must reduce contact regressions
without giving back the E5-T temporal gains. A result obtained only by
reducing total motion is a matched-budget engineering improvement, not
evidence that the normal-direction mechanism is the main cause.

## Implementation Hash

```text
scripts/analyze_contactopt_sequence_direction.py
2b68b895ec77d037ff451274cf07b5785d6c72a93fce8fcdb5c43e5e6d34548c
```
