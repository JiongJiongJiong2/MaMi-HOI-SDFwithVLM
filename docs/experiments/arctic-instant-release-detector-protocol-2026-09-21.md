# ARCTIC Instant-Release Detector Protocol

Date: 2026-09-21

Status: revised after the five-frame-confirmation gate failure

## Reason for Revision

The original online detector required five consecutive non-contact
frames after release. It excluded 9 train and 2 val Tier A events that
briefly re-contacted the object within four frames.

The original result remains recorded as a FAIL. This revision does not
replace it.

## Revised Trigger

For each hand:

```text
stable contact: 15 consecutive raw-contact frames
release:        first raw non-contact frame after that stable segment
confirmation:   none
```

The trigger is emitted at the release boundary using only current and
past frames. It does not read receiving-hand onset or future frames.

## Labels

An emitted release is positive only if
`(sequence, outgoing_hand, release_frame)` exactly matches a Tier A
candidate.

## Models

The model, features, participant folds, and threshold-selection rule are
unchanged from the online detector protocol.

## Gate

The revised pilot passes only if:

1. Tier A release recall is at least 0.95 in train and val;
2. best val AUPRC exceeds val prevalence;
3. best val F1 exceeds all-positive F1;
4. best val F1 is at least 0.50.

Official test remains unavailable.
