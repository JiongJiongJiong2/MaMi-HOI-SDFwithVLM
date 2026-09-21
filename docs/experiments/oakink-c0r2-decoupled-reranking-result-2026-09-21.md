# OakInk C0-R2 Decoupled Reranking Result

Date: 2026-09-21

Status: complete with frozen promotion gate NO-GO

Protocol:

```text
docs/experiments/oakink-c0r2-decoupled-reranking-protocol-2026-09-21.md
```

Compact summary:

```text
docs/experiments/oakink_c0r2_decoupled_reranking_summary_v1_20260921.json
```

## Scope

C0-R2 repaired the two main C0-R1 protocol weaknesses:

```text
use all 82 reference-valid targets;
condition only on receiver contact-region geometry;
evaluate only on receiver joint geometry not used by the condition;
test paired utility directly rather than selection-change frequency.
```

## Data

```text
eligible target events: 82
candidate sets:         two events each
object groups:          41
```

The pairwise candidate limit remains.

## Independent Oracle

The independent joint-based evaluation selects the observed target-event
giver on:

```text
0.5732
```

This passes the frozen oracle-headroom requirement (`0.55`) and shows
that the evaluation contains some candidate-dependent signal.

## Primary Result

Top-1 observed-event rates:

| Arm | Rate |
|---|---:|
| no future | 0.5000 |
| correct future | 0.5000 |
| shuffled future | 0.5000 |
| evaluation oracle | 0.5732 |

With the primary region weight `0.5`, the correct future condition does
not change a single top-1 selection:

```text
correct vs no-future transitions: 0 changes
correct vs shuffled transitions:  0 changes
```

Paired evaluation differences:

```text
correct - no future:
  mean 0.0, 95% grouped CI [0.0, 0.0]

correct - shuffled:
  mean 0.0, 95% grouped CI [0.0, 0.0]

correct - shuffled collision:
  mean 0.0, 95% grouped CI [0.0, 0.0]
```

## Sensitivity

At the stronger frozen weight `1.0`, the region term moves a few
decisions and produces a small positive mean evaluation change:

```text
correct minus no-future mean: 0.00892
correct minus shuffled mean:  0.00878
```

The primary weight and the low weight have zero mean change. This is not
enough to claim a stable incremental effect.

## Gate

```text
at least 60 eligible events:                    pass
independent oracle top-1 >= 0.55:               pass
correct beats no-future with positive CI:       fail
correct beats shuffled with positive CI:        fail
top-1 not dropped:                              pass
collision not worse than shuffled:              fail
sensitivity direction at weights 0.5 and 1.0:   fail
overall:                                        NO-GO
```

## Interpretation

The independent evaluation oracle shows that the data contains a small
amount of future compatibility structure (`0.5732` versus `0.5000`
chance). However, the decoupled receiver-region condition is not strong
enough to move the primary ranking relative to the current contact
score.

This is a cleaner negative result than C0-R1:

- evaluation is no longer circular;
- target coverage is larger;
- the gate tests utility rather than change frequency.

The current observed pairwise grasp universe therefore does not justify
a full future-conditioned optimizer. The result does not show that all
future conditioning representations are useless; it shows that a
receiver contact-region-only condition has negligible marginal utility
over the current grasp score.

No post-hoc weight increase or gate change was used to convert the
result.

## Server Output

```text
/root/autodl-tmp/oakink_c0r2_decoupled_v1_20260921/summary.json
```

Four C0-R2 unit tests passed on the no-GPU server.

## Implementation Hashes

```text
scripts/evaluate_oakink_c0r2_decoupled_reranking.py
d17d87fb84e4cf4f39515a7326919f5c5a2161ada8c967cede8e4f26f4d57cf6

tests/test_evaluate_oakink_c0r2_decoupled_reranking.py
9afd0ee06e855f5e60c7cfee384d412f82fd77c3df46597ef4b879f5d8b80a7e

server full summary
778183b8e2ff78c79fe3259a78692332802b01da5f919a281f4e481395e022c5
```

## Decision

```text
receiver-region-only future conditioning: NO-GO
full future-conditioned optimizer:        not authorized
independent evaluation headroom:          present but small
next C rescue under current data:         not justified
```
