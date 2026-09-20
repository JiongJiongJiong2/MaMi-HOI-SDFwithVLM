# EPIC-Contact Lifecycle Analysis Protocol

Date: 2026-09-20

Status: completed

## Goal

Determine whether the full EPIC-Contact B manifest supports:

- contact hold modeling;
- contact onset modeling;
- contact release modeling;
- bimanual simultaneous-contact analysis.

The analysis decides whether full B model training is justified.

## Inputs

```text
docs/experiments/epic_contact_frames_v1.jsonl.gz
docs/experiments/epic_contact_episodes_v1.jsonl.gz
```

## Metrics

For each split:

- episode total, complete, onset-only, release-only, and truncated counts;
- contact run length distribution;
- valid hand frames and adjacent-frame transition matrix:
  - contact to contact;
  - contact to non-contact;
  - non-contact to contact;
  - non-contact to non-contact;
- same-object bimanual group, frame, and contiguous-run counts;
- train/test video overlap.

## Decision Thresholds

```text
hold labels ready:       >= 1,000 train contact-to-contact transitions
onset labels ready:      >= 100 train non-contact-to-contact transitions
release labels ready:    >= 100 train contact-to-non-contact transitions
                         and >= 50 complete train episodes
split ready:             no train/test video overlap
bimanual analysis ready: >= 20 train groups and >= 1,000 frames
```

Full B modeling requires all four of hold, onset, release, and split gates.
If release fails, model development is limited to hold or bimanual
analysis.
