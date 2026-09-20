# EPIC-Contact Strict Nonlinear Event Result

Date: 2026-09-20

Status: complete with nonlinear gate FAIL

Protocol:

```text
docs/experiments/epic-contact-event-boosting-protocol-2026-09-20.md
```

Frozen baseline:

```text
commit 8b52863
docs/experiments/epic-contact-event-baseline-2026-09-20.md
```

Input:

```text
docs/experiments/epic_contact_strict_frames_v1.jsonl.gz
SHA-256 d202ebbdd3c2d907eb0d607794cd6d4cfa7ffdc9f61796628d7a8f26abaa9e36
```

The nonlinear model used the exact baseline feature vector and the
frozen participant-disjoint split. No feature or threshold selection was
performed on test.

## Test Metrics

| Target | Arm | AUROC | AUPRC | F1 |
|---|---|---:|---:|---:|
| next contact | learned logistic | 0.8245 | 0.9450 | 0.9105 |
| next contact | nonlinear boosting | 0.8239 | 0.9464 | 0.9116 |
| onset | learned logistic | 0.9668 | 0.6841 | 0.7201 |
| onset | nonlinear boosting | 0.9706 | 0.6945 | 0.7053 |
| release | learned logistic | 0.8143 | 0.2972 | 0.3784 |
| release | nonlinear boosting | 0.8066 | 0.2952 | 0.3039 |

The nonlinear arm improves onset AUPRC by `0.0104` and next-contact
AUPRC by `0.0014`, but both paired intervals include zero. It does not
improve release ranking and substantially reduces release F1.

## Participant-Clustered Differences

95% bootstrap intervals for nonlinear boosting minus learned logistic:

| Target | Metric | Mean Difference | 95% CI |
|---|---|---:|---:|
| next contact | AUPRC | 0.0012 | [-0.0010, 0.0027] |
| next contact | F1 | 0.0011 | [-0.0007, 0.0036] |
| onset | AUPRC | 0.0116 | [-0.0060, 0.0383] |
| onset | F1 | -0.0143 | [-0.0285, -0.0041] |
| release | AUPRC | -0.0031 | [-0.0086, 0.0023] |
| release | F1 | -0.0741 | [-0.1100, -0.0207] |

## Reproduction

The experiment was executed locally and on the server with the same
input and seeds:

```text
server: 18 cores, 50 GB RAM, RTX 4090 D
python: 3.10.8
numpy:  1.24.1
scikit-learn: 1.6.1
```

The summaries have identical structure and gate outcomes. Fifteen
floating-point values differ in their last bits, with maximum absolute
difference `2.22e-16`. Across `47,556` test prediction cells, 9,038
values differ only by floating-point representation, with maximum
absolute difference `5.55e-16`.

Server output:

```text
/root/autodl-tmp/epic_contact_event_boosting_v1_20260920
```

## Gate

```text
release AUPRC improves:                    fail
release paired AUPRC lower bound > 0:      fail
onset AUPRC noninferior:                   pass
next-contact AUPRC noninferior:            pass
onset F1 noninferior:                      pass
release F1 noninferior:                    fail
overall:                                   FAIL
```

## Decision

Do not adopt the nonlinear boosting arm. The learned logistic model
remains the strongest justified strict EPIC-Contact event model on this
feature set. Adding model capacity does not solve the weak release
ranking, and post-hoc tuning will not be used to overturn the frozen
gate.

The next strict-route check is the required `0.5-3 mm` threshold
sensitivity curve for the logistic model. If event ranking collapses
away from `1 mm`, the strict B route stops and data work moves to
ARCTIC/GRAB. If the curve is stable, the strict B result remains a
limited short-horizon lifecycle diagnostic, not evidence of handover.

The C handover route remains NO-GO.

## Implementation Hashes

```text
scripts/train_epic_contact_event_boosting.py
287431cc0f54edaf8ef878b521d163b6af16e84ae243201190f8a23eeedd2730

tests/test_train_epic_contact_event_boosting.py
23777c51cfd5d9ebe9280b030a238d781d2f05cef92c0e0a4ff64413a2f43ce8
```
