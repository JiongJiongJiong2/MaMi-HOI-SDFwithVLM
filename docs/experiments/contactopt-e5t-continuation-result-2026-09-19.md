# E5-T Temporal Continuation Result

Date: 2026-09-19

## Scope

Full train/dev continuation from the E5 optimized poses:

```text
chunks: 49/49
windows: 376
train: 256
dev: 120
test: untouched
```

Protocol:

```text
docs/experiments/contactopt-e5t-continuation-protocol-2026-09-19.md
```

Raw result:

```text
docs/experiments/contactopt_e5t_solver_v1_20260919.json
```

Server log:

```text
/root/autodl-tmp/contact_action_20260914/e5t_solver_v1_20260919/run.log
```

## Aggregate

| Metric | Raw ContactOpt | E5 | E5-T |
|---|---:|---:|---:|
| Combined passes | `92/376` | `343/376` | `341/376` |
| Temporal passes | n/a | `357/376` | `367/376` |
| Contact passes | n/a | `362/376` | `350/376` |
| Mean acceleration ratio | `5.050` | `1.180` | `0.955` |
| Mean jerk ratio | `7.532` | `0.991` | `0.784` |

E5-T reduced temporal failures from 19 to 9, but increased contact failures
from 14 to 26. Relative to E5 it gained 11 combined windows and lost 13,
for a net change of `-2`.

## Per Object

| Object | Windows | Raw | E5 | E5-T | E5 temporal | E5-T temporal | E5 contact | E5-T contact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clothesstand | 38 | 9 | 36 | 38 | 36 | 38 | 38 | 38 |
| floorlamp | 32 | 4 | 21 | 25 | 22 | 26 | 31 | 31 |
| largebox | 96 | 40 | 93 | 89 | 95 | 95 | 94 | 90 |
| largetable | 40 | 2 | 33 | 35 | 38 | 40 | 35 | 35 |
| monitor | 24 | 2 | 23 | 22 | 23 | 22 | 24 | 24 |
| plasticbox | 46 | 0 | 38 | 38 | 43 | 46 | 41 | 38 |
| smallbox | 31 | 8 | 31 | 30 | 31 | 31 | 31 | 30 |
| trashcan | 52 | 24 | 51 | 47 | 52 | 52 | 51 | 47 |
| whitechair | 17 | 3 | 17 | 17 | 17 | 17 | 17 | 17 |

## Failure Decomposition

| Arm | Combined failures | Temporal-only | Contact-only |
|---|---:|---:|---:|
| E5 | 33 | 19 | 14 |
| E5-T | 35 | 9 | 26 |

The E5/E5-T oracle union passes `354/376` windows, showing that the two
optimizers are complementary, but this is selection headroom, not a
deployable result.

## Decision

```text
E5-T temporal continuation: GO for additional temporal reduction
E5-T universal promotion:   NO-GO
Test evaluation:            not run
```

E5-T should not replace E5. It over-trades contact for smoothness. The next
planned step is a contact-preserving continuation or a validated selector
between E5 and E5-T; neither is authorized to use test data.
