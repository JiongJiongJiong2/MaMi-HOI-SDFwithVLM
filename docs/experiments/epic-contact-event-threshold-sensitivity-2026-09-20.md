# EPIC-Contact Strict Event Threshold Sensitivity Result

Date: 2026-09-20

Status: complete with primary sensitivity gate PASS

Protocol:

```text
docs/experiments/epic-contact-event-threshold-sensitivity-protocol-2026-09-20.md
```

Input:

```text
docs/experiments/epic_contact_strict_frames_v1.jsonl.gz
SHA-256 d202ebbdd3c2d907eb0d607794cd6d4cfa7ffdc9f61796628d7a8f26abaa9e36
```

The participant split, feature vector, logistic optimizer,
dev-only threshold selection, and metric definitions are unchanged.
Only the contact threshold and distance normalization change.

## Event Counts

| Threshold | Train Onset | Train Release | Test Onset | Test Release |
|---:|---:|---:|---:|---:|
| 0.5 mm | 1,189 | 1,201 | 609 | 588 |
| 1.0 mm | 654 | 648 | 370 | 364 |
| 1.5 mm | 157 | 150 | 74 | 76 |
| 2.0 mm | 47 | 41 | 34 | 25 |
| 3.0 mm | 7 | 8 | 14 | 13 |

The `2 mm` and `3 mm` points have too few train events for independent
model claims. They are retained to complete the requested sensitivity
curve.

## Test AUPRC

| Threshold | Onset Logistic | Onset Best Trivial | Release Logistic | Release Best Trivial |
|---:|---:|---:|---:|---:|
| 0.5 mm | 0.3674 | 0.1548 | 0.6831 | 0.1477 |
| 1.0 mm | 0.6841 | 0.0986 | 0.2972 | 0.0940 |
| 1.5 mm | 0.4942 | 0.0159 | 0.1236 | 0.0220 |
| 2.0 mm | 0.4933 | 0.0074 | 0.1006 | 0.0128 |
| 3.0 mm | 0.3800 | 0.0032 | 0.1820 | 0.0127 |

## Test F1

| Threshold | Onset Logistic | Release Logistic |
|---:|---:|---:|
| 0.5 mm | 0.4237 | 0.7197 |
| 1.0 mm | 0.7201 | 0.3784 |
| 1.5 mm | 0.5506 | 0.2394 |
| 2.0 mm | 0.6136 | 0.2449 |
| 3.0 mm | 0.4000 | 0.0166 |

The learned model beats the best trivial AUPRC at every threshold. The
primary `0.5 mm` and `1.5 mm` checks pass. The curve is therefore
directionally robust, but not flat: release AUPRC falls from `0.6831`
at `0.5 mm` to `0.2972` at `1 mm` and `0.1236` at `1.5 mm`.

This means the strict event signal is real under the frozen controls,
but its effect size is highly dependent on how strictly contact is
defined.

## Reproduction

The `1 mm` curve point reproduces the frozen baseline evaluation
exactly, with zero numeric difference.

The complete curve was repeated on the server:

```text
server: 18 cores, 50 GB RAM, RTX 4090 D
python: 3.10.8
numpy:  1.24.1
```

The local and server summaries have identical structure, counts, gate
outcomes, and metrics at reported precision. Thirty-seven floating-point
values differ only in their last bits, with maximum absolute difference
`2.22e-16`.

Server output:

```text
/root/autodl-tmp/epic_contact_event_threshold_sensitivity_v1_20260920
```

## Gate

```text
1 mm point present and reproduced:                 pass
0.5 mm onset beats best trivial:                   pass
0.5 mm release beats best trivial:                 pass
1.5 mm onset beats best trivial:                   pass
1.5 mm release beats best trivial:                 pass
overall:                                           PASS
```

## Decision

Phase 2 is complete. The strict EPIC-Contact route survives as a
threshold-qualified, short-horizon contact lifecycle diagnostic. The
logistic model remains the strongest justified model; nonlinear
boosting did not improve release prediction.

The result is not stable enough to support a universal `1 mm` claim.
Future EPIC-Contact reporting must include the full `0.5-3 mm` curve and
state that the effect size is threshold-sensitive.

The result still does not establish handover, receiver-role switching,
or cross-hand object transfer. The next data track for those questions
is ARCTIC/GRAB. The C handover route remains NO-GO until an external
dataset provides verified role-switch events.

## Implementation Hashes

```text
scripts/train_epic_contact_event_baselines.py
0103c9f5fdefa4652a31e8442e8173ce48e75709468c36a8029f24e302db729f

scripts/evaluate_epic_contact_event_threshold_sensitivity.py
f0add498c52d0e70fcd9892e1e3c0c055d7ee9f3bf225bc6958228066b37369b

tests/test_train_epic_contact_event_baselines.py
4fca883578b5b1482b401e1547bd282e3f4625548ac600831cea5e4af32d9711

tests/test_evaluate_epic_contact_event_threshold_sensitivity.py
eaa77b77457f4b72bef440a6562f6682cf961f2c6f135cb254f8c25fe9c6807b
```
