# E5 Sequence-Level ContactOpt Solver Result

Date: 2026-09-19

## Scope

Full train/dev run of the frozen E5 sequence-level optimizer:

```text
chunks: 49/49
windows: 376
train: 256
dev: 120
test: untouched
```

Protocol:

```text
docs/experiments/contactopt-e5-sequence-solver-protocol-2026-09-19.md
```

Raw result:

```text
docs/experiments/contactopt_e5_solver_v1_20260919.json
```

Server log:

```text
/root/autodl-tmp/contact_action_20260914/e5_solver_v1_20260919/run.log
```

## Aggregate

| Metric | Raw ContactOpt | E5 solver |
|---|---:|---:|
| Combined passes | `92/376` | `343/376` |
| Temporal passes | not stored in E5 JSON | `357/376` |
| Contact passes | not stored in E5 JSON | `362/376` |
| Mean acceleration ratio | `5.050` in Phase 18 | `1.180` |
| Mean jerk ratio | `7.532` in Phase 18 | `0.991` |

The combined pass rate increased by 251 windows, or 66.8 percentage points.

Failure decomposition:

```text
combined failures:  33
contact-only:       19
temporal-only:      14
both:                0
```

Split results:

| Split | Windows | Raw combined | E5 combined | Accel mean | Jerk mean |
|---|---:|---:|---:|---:|---:|
| train | 256 | `72` | `236` | `1.107` | `0.943` |
| dev | 120 | `20` | `107` | `1.337` | `1.093` |

## Per Object

| Object | Windows | Raw combined | E5 combined | Contact pass | Temporal pass | Accel mean | Jerk mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| clothesstand | 38 | 9 | 36 | 38 | 36 | 1.154 | 1.080 |
| floorlamp | 32 | 4 | 21 | 31 | 22 | 2.415 | 1.858 |
| largebox | 96 | 40 | 93 | 94 | 95 | 0.973 | 0.840 |
| largetable | 40 | 2 | 33 | 35 | 38 | 1.172 | 0.943 |
| monitor | 24 | 2 | 23 | 24 | 23 | 1.145 | 0.979 |
| plasticbox | 46 | 0 | 38 | 41 | 43 | 1.255 | 1.037 |
| smallbox | 31 | 8 | 31 | 31 | 31 | 1.052 | 0.855 |
| trashcan | 52 | 24 | 51 | 51 | 52 | 0.939 | 0.805 |
| whitechair | 17 | 3 | 17 | 17 | 17 | 0.923 | 0.823 |

## Decision

```text
E5 improvement over raw ContactOpt: GO
E5 universal promotion gate:       NO-GO
Test evaluation:                   not run
```

The objective strongly improves temporal behavior and preserves the contact
gate on most windows. It is not ready to freeze because 33 train/dev windows
still fail the combined gate. `floorlamp` is the dominant remaining failure:
`21/32` combined, mean acceleration ratio `2.415`, and 11 failed windows.

The next planned action is review of this frozen result only. No threshold,
objective, or test-set change follows from this report.
