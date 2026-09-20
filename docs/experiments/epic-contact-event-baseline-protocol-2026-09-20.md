# EPIC-Contact Strict Event Baseline Protocol

Date: 2026-09-20

Status: frozen before training

## Goal

Test whether strict-contact lifecycle events are predictable from local
history beyond trivial state copying and linear distance extrapolation.

## Inputs

```text
docs/experiments/epic_contact_strict_frames_v1.jsonl.gz
```

The existing participant-disjoint split is mandatory and must not be
changed.

## Targets

For each valid hand frame `t`:

```text
next_contact: contact at t+1
onset:        not contact at t and contact at t+1
release:      contact at t and not contact at t+1
```

History length is four contiguous frames. Samples never cross clip or
frame-gap boundaries.

## Model Arms

| Arm | Definition |
|---|---|
| copy-current-state | predict `contact_{t+1}=contact_t` |
| distance-linear | extrapolate distance from the last two deltas and threshold it |
| learned-logistic | weighted logistic regression on history distance/contact plus hand and object |

Thresholds for hard F1 are chosen on dev per arm and target, then frozen
for test.

## Metrics

```text
next contact: AUROC, AUPRC, F1
onset:        AUPRC, F1
release:      AUPRC, F1
```

Paired participant-clustered bootstrap reports learned-minus-baseline
differences with 95% intervals.

## Gate

The strict learned event route continues only if:

1. onset and release AUPRC exceed the best trivial arm;
2. the paired participant-clustered F1 or AUPRC interval excludes zero;
3. next-contact performance does not depend only on copying the current
   state.

Failure stops the strict B route and triggers external data work.
