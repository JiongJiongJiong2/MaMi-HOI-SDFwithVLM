# EPIC-Contact Strict Lifecycle Protocol

Date: 2026-09-20

Status: frozen before strict build

## Goal

Convert the strict-contact sensitivity finding into a reproducible dataset
without reusing the failed official `3 mm` protocol.

## Frozen Definition

```text
contact: min_distance_m <= 0.001
maximum continuous frame gap: 1
```

This threshold is selected because it is the only candidate that produces
sufficient lifecycle events, but it remains exploratory and must always
be reported with the `0.5-3 mm` sensitivity curve.

## Split

Participants, not windows, are assigned to train/dev/test. The target
fractions are:

```text
train 0.70
dev   0.15
test  0.15
```

Assignment is deterministic and balances hold, onset, release, and
complete episodes. No participant or video may cross splits.

## Outputs

```text
epic_contact_strict_frames_v1.jsonl.gz
epic_contact_strict_episodes_v1.jsonl.gz
epic_contact_strict_split_v1.json
summary.json
```

The episode table contains onset, release, duration, contact frame count,
object, hand, and clip identity. The frame table contains the strict
contact state under the new participant split.

## Gate

The strict dataset is accepted only if:

1. no participant or video appears in multiple splits;
2. all three splits contain hold, onset, and release labels;
3. train has at least 100 onset and release transitions;
4. dev and test each have at least 20 complete episodes.

Failure returns to external-data acquisition rather than threshold tuning.
