# ContactOpt Dense Temporal Feasibility Result: Phase 12

Date: 2026-09-18

Protocol:

```text
docs/experiments/contactopt-phase12-temporal-protocol-2026-09-18.md
```

Raw metrics:

```text
docs/experiments/contactopt_phase12_temporal.json
```

## Results

| Case | Speed ratio | Acceleration ratio | Jerk ratio | Contact gate | Temporal gate |
|---|---:|---:|---:|---|---|
| monitor | 1.20 | 5.59 | 7.16 | PASS | FAIL |
| largetable | 1.27 | 5.63 | 7.77 | PASS | FAIL |
| plasticbox | 1.21 | 8.12 | 11.71 | PASS | FAIL |
| smallbox | 1.46 | 11.54 | 19.92 | PASS | FAIL |

Aggregate:

```text
contact passes:  4/4
temporal passes: 0/4
```

Independent ContactOpt comfortably preserves contact and trajectory speed but
strongly amplifies frame-to-frame acceleration and jerk. The result is a clear
NO-GO for directly integrating independent frame-wise ContactOpt output.

## Decision

Contact refinement and temporal consistency are separate gates. The teacher
passes the contact gate but fails the temporal gate. Any integration attempt
must add temporal regularization before claiming usable finger refinement.
