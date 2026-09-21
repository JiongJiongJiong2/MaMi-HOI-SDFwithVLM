# OakInk Cross-Intent Specificity Protocol

Date: 2026-09-21

Status: frozen before execution

## Goal

Test whether the OakInk handover-onset model is specific to handover
transitions or whether it also fires around ordinary contact releases
in `use`, `hold`, and `liftup` sequences.

## Handover Model

Use the frozen logistic window model and threshold selected from
handover-train participant-pair-grouped OOF scores.

Do not retrain on non-handover events and do not tune on val/test.

## Cross-Intent Negatives

For OakInk single-hand intents:

```text
use:    0001
hold:   0002
liftup: 0003
```

For each sequence:

1. compute active-hand/object distance using camera 0;
2. find stable contact segments at 5 mm with at least 15 frames;
3. create negative windows for the 15 frames before each stable-contact
   release;
4. encode the active hand as the giver history and the absent hand as a
   20 mm no-contact channel.

The absent-receiver encoding is intentional: the model must not treat a
single-hand release as a handover.

## Evaluation

For val and test:

```text
positive windows: handover onset windows
negative windows: handover non-event windows + cross-intent release windows
```

Report:

```text
AUPRC, F1, precision, recall
cross-intent false-positive rate
median cross-intent score
```

## Gate

Specificity passes only if:

1. handover recall remains at least 0.50 on val and test;
2. cross-intent false-positive rate is at most 0.10 on val and test;
3. combined AUPRC exceeds prevalence on val and test.

Failure means the model detects generic contact transitions rather than
handover-specific events.
