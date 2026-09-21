# ARCTIC Role-Switch Event Review Result

Date: 2026-09-21

Status: complete; Tier A candidates show object motion and hand retreat

Protocol:

```text
docs/experiments/arctic-role-switch-event-review-protocol-2026-09-21.md
```

Input:

```text
/root/autodl-tmp/arctic_role_switch_quality_v1_20260921/
  tier_a_candidates.jsonl.gz
```

Primary threshold: `3 mm`.

## Event Counts

```text
train: 45 candidates across 35 sequences and 8 participants
val:    9 candidates across  5 sequences and 1 participant
```

## Object Motion

Within `15` frames before release and `30` frames after receiving
onset:

| Split | Translation >= 5 mm | Rotation >= 5 deg | Either |
|---|---:|---:|---:|
| train | 36 / 45 | 30 / 45 | 36 / 45 |
| val | 7 / 9 | 6 / 9 | 7 / 9 |

Train object translation:

```text
median: 33.9 mm
mean:   79.0 mm
q75:    84.2 mm
```

Train object rotation:

```text
median: 14.7 degrees
mean:   28.7 degrees
q75:    37.0 degrees
```

Val object translation:

```text
median: 56.3 mm
mean:   99.6 mm
```

Val object rotation:

```text
median: 39.8 degrees
mean:   51.5 degrees
```

The object is moving in most Tier A events.

## Hand Motion

Outgoing-hand retreat relative to the release frame:

| Split | Retreat >= 5 mm | Median Retreat |
|---|---:|---:|
| train | 45 / 45 | 91.7 mm |
| val | 9 / 9 | 162.4 mm |

Receiving-hand root displacement after onset:

| Split | Displacement >= 5 mm | Median Displacement |
|---|---:|---:|
| train | 44 / 45 | 42.4 mm |
| val | 9 / 9 | 135.3 mm |

The outgoing hand leaves the object contact region and the receiving hand
continues moving. This rules out static contact alternation for the
overwhelming majority of Tier A candidates.

## Interpretation

Tier A candidates are consistent with object role transfer:

1. one hand releases;
2. the other hand establishes sustained contact;
3. the outgoing hand retreats;
4. the receiving hand moves;
5. the object moves or rotates in most events.

This is still geometric evidence, not human-annotated handover ground
truth. Receiver-role validation remains required before any learned
handover claim.

## Artifacts

Server:

```text
/root/autodl-tmp/arctic_role_switch_event_review_v1_20260921
```

Files:

```text
event_review_summary.json
tier_a_event_features.jsonl.gz
```

## Decision

Proceed to a participant-disjoint pilot role-switch event manifest using
Tier A. The official ARCTIC test subject is still unavailable, so model
development must use held-out accessible participants and must not claim
official test performance.

The next implementation should derive event-centered windows from Tier A
and define prediction targets before training any sequence model.

## Implementation Hashes

```text
scripts/review_arctic_role_switch_events.py
6ddbd37745b173c5c920ce04b7240e4c4e661cfe1f1dc4c5d8e11fed6a837468

tests/test_review_arctic_role_switch_events.py
5e958f70a6cfeb839a4b55e7c7a12e19288c100b335f51063e63cd1cecbb1b13
```
