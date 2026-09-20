# EPIC-Contact Lifecycle Analysis Result

Date: 2026-09-20

Status: complete with full B model NO-GO

Protocol:

```text
docs/experiments/epic-contact-lifecycle-protocol-2026-09-20.md
```

Summary:

```text
docs/experiments/epic_contact_lifecycle_summary_v1_20260920.json
SHA-256 a0f9ec0f1fb1fcc61620cbbeb7f842cc9dd4a9123b344b62b05208c898cd5e37
```

Server summary:

```text
/root/autodl-tmp/epic_contact_b_manifest_v1_20260920/lifecycle_summary.json
```

## Episode Statistics

```text
train episodes:                33,290
train complete episodes:           17
train onset-only episodes:         20
train release-only episodes:       16
test episodes:                  3,872
test complete episodes:              0
```

Complete train episodes are short: contact frame count median `1`, mean
`2.71`, maximum `12`.

## Adjacent-Frame Transitions

| Transition | Train | Test |
|---|---:|---:|
| contact to contact | 22,149 | 2,401 |
| contact to non-contact | 33 | 2 |
| non-contact to contact | 37 | 1 |
| non-contact to non-contact | 133 | 22 |
| adjacent pairs | 22,352 | 2,426 |

The data overwhelmingly supports contact persistence. Onset and release
transitions are rare.

## Split Check

```text
train videos:      124
test videos:        29
overlapping videos: 14
video-disjoint:     false
```

The official train/test split is not video-disjoint. A new
sequence-disjoint split must be built before any learned comparison.

## Bimanual Contact

```text
train groups:        90
train bimanual frames: 3,824
train both-contact frames: 3,756
test groups:          6
test bimanual frames: 151
```

Bimanual data is sufficient for descriptive simultaneous-contact analysis,
but it remains evidence of joint support rather than handover.

## Gate Result

```text
hold labels ready:               true  (22,149 train transitions)
onset labels ready:              false (37 < 100)
release labels ready:            false (33 < 100)
complete episode gate:           false (17 < 50)
official split video-disjoint:   false (14 overlapping videos)
bimanual analysis ready:         true
full B model ready:              false
handover analysis ready:         false
```

## Decision

Do not train a full contact lifecycle model on EPIC-Contact. The dataset
can support a narrow hold-persistence or simultaneous-contact diagnostic,
but it should not be used to claim reliable onset/release modeling.

The next B-related work, if retained, must be limited to:

1. a contact-hold persistence baseline;
2. simultaneous bimanual-contact analysis;
3. a new video-disjoint split.

Release or handover claims require additional data.

## Implementation Hash

```text
scripts/analyze_epic_contact_lifecycle.py
906b9a3b1ba2eb691cae413e5a5fb0eb285b999a8a5525e28aaaf8f7a178c065
```
