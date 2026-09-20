# EPIC-Contact Lifecycle Analysis Result

Date: 2026-09-20

Status: official 3 mm lifecycle NO-GO; strict-contact second gate required

Protocol:

```text
docs/experiments/epic-contact-lifecycle-protocol-2026-09-20.md
```

Summary:

```text
docs/experiments/epic_contact_lifecycle_summary_v1_20260920.json
SHA-256 2835ac67b438b50c959f706b15b03ffb7b9e2f215f356b088afe89180f1e30f8
```

Server summary:

```text
/root/autodl-tmp/epic_contact_b_manifest_v1_20260920/lifecycle_summary_v2.json
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
full B model at official 3 mm:   false
handover analysis ready:         false
```

## Threshold Sensitivity

The manifest stores minimum hand-object distance, so contact can be
recounted at stricter thresholds:

| Threshold | Train onset | Train release | Train complete | Test complete |
|---:|---:|---:|---:|---:|
| 0.5 mm | 3,399 | 3,330 | 2,481 | 275 |
| 1.0 mm | 2,100 | 2,132 | 1,160 | 100 |
| 1.5 mm | 534 | 521 | 235 | 11 |
| 2.0 mm | 163 | 159 | 66 | 1 |
| 3.0 mm | 37 | 33 | 17 | 0 |

The strict `1 mm` definition has enough lifecycle events for a B pilot.
This is a threshold-sensitivity result, not permission to replace the
official `3 mm` protocol after seeing the outcome.

## Independent Split Blocker

Even the strict-contact route still has `14` videos shared between train
and test. A new video-disjoint split is required before any learned
comparison.

## Decision

Do not train a full contact lifecycle model under the official `3 mm`
protocol. The dataset can support a narrow hold-persistence or
simultaneous-contact diagnostic, but the official definition does not
support reliable onset/release modeling.

The strict `1 mm` lifecycle is a different, testable mechanism and may be
pursued only through:

1. a newly frozen strict-contact protocol;
2. a video-disjoint train/dev/test split;
3. a threshold sensitivity curve reported alongside the primary result;
4. contact-hold persistence and trivial-copy controls.

Release or handover claims require additional data.

## Implementation Hash

```text
scripts/analyze_epic_contact_lifecycle.py
1653966ad9ccbdb934f96e45bed45f79d48fcf6e6e05cbed6e6ccd48a2189ec10
```
