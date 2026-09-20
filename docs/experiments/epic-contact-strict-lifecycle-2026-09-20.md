# EPIC-Contact Strict Lifecycle Dataset Result

Date: 2026-09-20

Status: complete with participant-disjoint gate PASS

Protocol:

```text
docs/experiments/epic-contact-strict-lifecycle-protocol-2026-09-20.md
```

## Inputs and Method

The strict lifecycle dataset was built from the full 57,686-frame
EPIC-Contact manifest using:

```text
contact threshold:       1 mm
maximum frame gap:       1
train/dev/test target:   0.70 / 0.15 / 0.15
split unit:              participant
```

Assignment deterministically balances hold, onset, release, and complete
episodes. No participant or video crosses splits.

The build was executed on the server and repeated locally. Parsed JSONL
rows and the split assignment were identical.

## Split Counts

| Split | Participants | Videos | Frames | Episodes | Complete |
|---|---:|---:|---:|---:|---:|
| train | 25 | 74 | 28,287 | 15,814 | 593 |
| dev | 3 | 33 | 13,159 | 6,684 | 349 |
| test | 3 | 32 | 16,240 | 9,142 | 318 |

Transition counts:

| Split | Hold | Onset | Release | Noncontact |
|---|---:|---:|---:|---:|
| train | 8,813 | 1,123 | 1,132 | 1,278 |
| dev | 3,715 | 583 | 585 | 753 |
| test | 4,887 | 610 | 630 | 669 |

Same-object bimanual frames:

```text
train: 2,382
dev:     627
test:    966
```

## Gate

```text
participant disjoint:                 pass
video disjoint:                       pass
all splits have hold/onset/release:   pass
train >= 100 onset and release:       pass
dev/test >= 20 complete episodes:     pass
overall:                              PASS
```

## Outputs

```text
docs/experiments/epic_contact_strict_frames_v1.jsonl.gz
docs/experiments/epic_contact_strict_episodes_v1.jsonl.gz
docs/experiments/epic_contact_strict_split_v1.json
docs/experiments/epic_contact_strict_lifecycle_summary_v1_20260920.json
```

Server outputs:

```text
/root/autodl-tmp/epic_contact_strict_lifecycle_v1_20260920
```

Output hashes:

```text
frames:
d202ebbdd3c2d907eb0d607794cd6d4cfa7ffdc9f61796628d7a8f26abaa9e36

episodes:
edd8c699413e04925dfd12d0377376482829cafe523c25fc873d67d1d465c6b8

split:
f7a6f0f3b9434f84eeb3d656a5e21e8843302357d1974c59ced24327ac4b7492

summary:
119e28e4ec7a2b4fa03746920fa3ce074dff1cd00a76f65a0b875ea21f566cb2
```

## Decision

Phase 1 is complete. The strict dataset can proceed to Phase 2, but the
`1 mm` threshold remains a sensitivity-selected protocol. Every model
result must include the `0.5-3 mm` sensitivity curve, and no official
`3 mm` claim may be retroactively replaced.

The next gate compares:

1. copy-current-contact-state;
2. fixed-distance threshold;
3. learned hold/onset/release baseline.

If the learned model does not beat the trivial and fixed-threshold
controls, the strict B route stops.

## Implementation Hash

```text
scripts/build_epic_contact_strict_lifecycle.py
d60097688f7334eae0a549c631bf26b056b70746014aa6b14d99dce930661682
```
