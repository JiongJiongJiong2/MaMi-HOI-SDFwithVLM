# ContactOpt Temporal Holdout Validation: Phase 15

Date: 2026-09-18

Protocol:

```text
docs/experiments/contactopt-phase15-holdout-validation-protocol-2026-09-18.md
```

Raw evidence:

```text
docs/experiments/phase15_cohort/cohort_summary.json
docs/experiments/phase15_cohort/*.json
docs/experiments/contactopt_phase15_smoothing.json
```

## Results

The fixed binomial-5 smoother was applied unchanged to four held-out dense
windows.

| Case | Frames | Speed ratio | Acceleration ratio | Jerk ratio | Contact change | Distance change | Promotion |
|---|---:|---:|---:|---:|---:|---:|---|
| floorlamp | 8 | 0.98 | 1.78 | 2.00 | +101.9% | -48.6% | PASS |
| whitechair | 6 | 0.97 | 2.00 | 2.12 | +43.9% | -15.6% | PASS |
| clothesstand | 8 | 1.21 | 2.59 | 3.86 | +32.1% | -27.3% | FAIL |
| largebox | 8 | 1.00 | 1.07 | 1.05 | +81.1% | -42.4% | PASS |

Aggregate:

```text
temporal passes:        3/4
contact+temporal pass:  3/4
```

All four ContactOpt raw refinement runs preserve contact and freeze wrist and
object coordinates. The failure is temporal only: `clothesstand` exceeds the
acceleration and jerk limits.

## Combined Evidence

Binomial-5 results:

```text
development windows: 3/4 pass
held-out windows:    3/4 pass
combined:            6/8 pass
```

The method is directionally useful but not reliable enough to freeze as the
integration smoother. It does not justify a claim that ContactOpt trajectories
are temporally consistent.

## Decision

Phase 15 is CONDITIONAL, not GO.

The two fixed local smoothers tested so far show a clear tradeoff:

```text
three-tap:  preserves more contact, leaves too much acceleration/jerk
five-tap:   removes more jitter, but still fails on fast transitions
```

Continuing to tune kernel width on the same windows would be post-hoc threshold
shopping. The next useful step is a sequence-level temporal solver with an
explicit fidelity and smoothness objective, trained or calibrated only on
development sequences and validated on a new untouched cohort.

## Training Boundary

No learned temporal model was trained in this phase. The current dense data
contain eight development and six held-out windows, which is too small to
justify a world-model training run. A sequence-disjoint dense-window dataset
must be built first.
