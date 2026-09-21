# OakInk Sequence-Aware Handover Detector Protocol

Date: 2026-09-21

Status: frozen after v1 failure, before v2 execution

## Reason for Revision

The raw v1 full-sequence detector failed val precision because one
handover sequence could emit multiple threshold-crossing peaks.

The v1 model, threshold, scores, and event-matching rule are unchanged.
Only inference post-processing changes.

## Eligibility

Run handover detection only for sequences with a second-hand/two-person
stream. In the OakInk manifest this is represented by
`source == handover`.

Single-hand `use`, `hold`, and `liftup` sequences are not eligible for a
handover event.

## Primary Peak

For each eligible sequence:

```text
if max(score) < frozen threshold: emit no event
otherwise: emit argmax(score) + 1
```

Exactly one primary handover transition can be emitted per sequence.

## Ground Truth and Matching

Ground truth remains the primary `5 mm` giver release per handover
sequence. Prediction matching remains one-to-one within 15 frames in the
same sequence.

## Gate

Val and test must each reach:

```text
precision >= 0.50
recall    >= 0.50
F1        >= 0.50
```

The execution is deterministic post-processing of the frozen v1 scores;
no model fitting or threshold tuning occurs.
