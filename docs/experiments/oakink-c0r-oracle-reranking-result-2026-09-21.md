# OakInk C0-R1 Oracle Reranking Result

Date: 2026-09-21

Status: complete with frozen promotion gate NO-GO

Protocol:

```text
docs/experiments/oakink-c0r-oracle-reranking-protocol-2026-09-21.md
```

Compact summary:

```text
docs/experiments/oakink_c0r_oracle_reranking_summary_v1_20260921.json
```

## Scope

C0-R1 used observed same-object giver grasps as candidates and tested
whether the correct future receiver target changed the ranking relative
to:

```text
no future condition;
correct same-object receiver target;
shuffled same-object receiver target.
```

This is a CPU-only oracle retrieval experiment. It does not generate a
new grasp, train a model, or run ContactOpt.

## Data

```text
eligible target events: 54
candidates per event:   2 exactly
object grouping:        participant-disjoint C0 object ids
```

Only target events with at least two complete same-object candidates
were eligible. The small candidate count is a direct property of the
available observed event set.

## Primary Result

Selection change against the no-future ranking:

```text
changed events: 8 / 54
rate:           0.1481
required:       0.2500
```

Correct versus shuffled future evaluation:

```text
mean paired gain: 0.05806
95% grouped CI:   [0.00608, 0.12886]
```

Top-1 observed-event rates:

| Arm | Rate |
|---|---:|
| no future | 0.5000 |
| correct future | 0.6111 |
| shuffled future | 0.3889 |

The correct condition improves the target-specific evaluation score
over the shuffled condition and raises the observed-event top-1 rate.
However, it selects a different candidate from the no-future arm only
`8/54` times.

## Sensitivity

The frozen sensitivity weights give:

| Weight | Selection Change Rate | Correct Minus Shuffled |
|---:|---:|---:|
| 0.25 | 0.0000 | 0.0000 |
| 0.50 | 0.1481 | 0.0581 |
| 1.00 | 0.2963 | 0.0539 |

The direction is positive at the primary and stronger weight, but the
lowest weight leaves the ranking unchanged. The frozen secondary
consistency requirement therefore fails.

## Gate

```text
at least 40 eligible events:                    pass
selection change rate >= 0.25:                  fail
correct exceeds shuffled with CI lower > 0:     pass
observed-event top-1 drop <= 0.10:              pass
secondary direction consistency:                fail
overall:                                        NO-GO
```

## Interpretation

The future receiver target carries a measurable signal: when it changes
the decision, the correct condition performs better than a same-object
shuffled condition. The effect is not strong or common enough to
authorize the full future-conditioned optimizer under the frozen gate.

The result is limited by the candidate set containing exactly two
observed grasps per target event. It is not evidence that future
receivers cannot matter; it is evidence that the current observed
candidate universe is too coarse for the proposed promotion threshold.

No post-hoc threshold, weight, or candidate selection was changed after
seeing the result.

## Server Output

```text
/root/autodl-tmp/oakink_c0r_oracle_reranking_v1_20260921/summary.json
```

The server ran without GPU. Seven unit tests passed with `unittest`.

## Implementation Hashes

```text
scripts/evaluate_oakink_c0_oracle_reranking.py
8b0eec3bb7e87c9a048292855ec4d41ad411da8dad4a21f8ad6d047548772593

tests/test_evaluate_oakink_c0_oracle_reranking.py
a982eee7b31d069d86e15b58d6cd8eb4ba3b8474bed9bb2ee373137826638a05

server full summary
733eacad3268e9628f2959d4332576b606d0c29018b54210da474fd85fde0ec6
```

## Decision

```text
C0-R1 promotion:                    NO-GO
full future-conditioned optimizer:  not authorized under this evidence
future-conditioning signal:         present but not strong/common enough
```
