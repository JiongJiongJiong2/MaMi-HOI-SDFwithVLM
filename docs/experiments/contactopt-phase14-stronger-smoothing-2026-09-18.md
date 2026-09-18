# ContactOpt Stronger Temporal Smoothing Result: Phase 14

Date: 2026-09-18

Protocol:

```text
docs/experiments/contactopt-phase14-stronger-smoothing-protocol-2026-09-18.md
```

Raw metrics:

```text
docs/experiments/contactopt_phase14_smoothing.json
```

## Results

The fixed `[1, 4, 6, 4, 1] / 16` smoother was applied to the same four
development windows.

| Case | Speed ratio | Acceleration ratio | Jerk ratio | Contact change | Distance change | Promotion |
|---|---:|---:|---:|---:|---:|---|
| monitor | 1.11 | 1.31 | 1.19 | +43.4% | -13.5% | PASS |
| largetable | 1.15 | 2.10 | 2.04 | +50.3% | -30.4% | FAIL |
| plasticbox | 0.99 | 1.99 | 2.13 | +15.1% | -16.9% | PASS |
| smallbox | 1.05 | 1.97 | 2.50 | +70.9% | -13.1% | PASS |

Aggregate:

```text
temporal passes:        3/4
contact+temporal pass:  3/4
```

The only failure is `largetable`, where acceleration ratio is `2.10` against
the frozen limit of `2.0`; jerk is within the pre-registered limit and contact
improves by `50.3%`.

## Decision

Phase 14 is CONDITIONAL. The wider smoother removes most high-frequency jitter
while retaining contact, but it does not pass the original four-window
development gate.

The thresholds and method remain unchanged. Do not relabel this run as a full
pass and do not select a per-case kernel after seeing the failure.

## Next Step

Validate the same fixed binomial-5 smoother on held-out contiguous windows
from the next Phase 11 candidates. This tests whether the near-pass generalizes
or is specific to the four development windows.
