# EPIC-Contact Data Gate Result

Date: 2026-09-20

Status: B candidate GO, C NO-GO for this dataset

Protocol:

```text
docs/experiments/epic-contact-data-gate-protocol-2026-09-20.md
```

Compact result:

```text
docs/experiments/epic_contact_gate_v1_20260920.json
```

## Inputs

```text
test payload:
E:/HOI/TMP/hopformer_repro_pkls/training_pkls/
epic_contact_v3_01-06-26_parallel_test_20260611.pkl
SHA-256 35df2fea120b3676dc13d6b2d5a7c215c4e11b06ed35e102de7cda08d74456a4

keys:
E:/HOI/TMP/hopformer_repro_pkls/training_pkls/
epic_contact_v3_01-06-26_parallel_20260611_keys.pkl
SHA-256 e30e8fb92202da0251905e77ddaf24f89a2dae2a16dc1a9edacd9f373c6664b1
```

The full training payload was not materialized in memory. Its keys file
contains `62,299` hand-sample records across `2,271` clips and `139`
videos. The test payload contains `6,316` hand-sample records, which merge
to `6,165` unique video frames across `236` clips and `29` videos.

EPIC-Contact stores left and right hand observations as separate records.
The audit first merges them by `(video_id, frame_num)` before evaluating
bimanual state.

## Schema

The test payload contains:

```text
left/right MANO pose, shape, vertices, joints, translation
left/right validity and per-joint validity
object vertices, faces, pose, translation, keypoints, bbox, class
dense hand-to-object distances and nearest indices:
  dist.ro, dist.lo, dist.or, dist.ol
  idx.ro,  idx.lo,  idx.or,  idx.ol
```

The official contact threshold in the HOPformer code is `3 mm`.

## B: Contact Memory

Using the `3 mm` threshold:

| Metric | Value |
|---|---:|
| left contact fraction among valid left frames | 0.9965 |
| right contact fraction among valid right frames | 0.9886 |
| either-hand contact fraction | 0.9932 |
| contact start events | 278 |
| contact end events | 278 |
| adjacent frame pairs | 2316 |
| nonadjacent frame pairs | 3820 |

There is enough contact-transition evidence to build a B pilot. The next
gate is not learning; it is constructing a sequence manifest that rejects
or separates clips with large frame gaps and verifies that start/end
events are true temporal transitions rather than changes between
unrelated clips.

## C: Future Handover

After merging left/right records by video and frame:

```text
same-object bimanual frames:       151
both-contact frames:               151
video-object groups:                 6
both-contact start/end events:       6 / 6
direct left-to-right switches:        0
direct right-to-left switches:        0
```

All six groups consist of a single simultaneous-contact run `B`. There is
no observed sequence where one hand releases and the other directly takes
over. EPIC-Contact therefore supplies bimanual cooperation and
simultaneous support examples, not a handover benchmark.

## Decision

```text
contact schema:             pass
B contact memory:           candidate GO, pending sequence-gap manifest
C future handover:          NO-GO on EPIC-Contact
GPU required for next step: no
```

C now requires either explicit handover annotations or another complete
sequence dataset. It must not be inferred from the six simultaneous
bimanual-contact groups.

## Next Step

Build an EPIC-Contact B manifest with:

1. clip-level start/end and frame gaps;
2. contact episode onset, hold, and release per hand;
3. object identity continuity;
4. explicit rejection of cross-clip jumps;
5. train/dev/test assignment that never treats overlapping clips as
   independent.

Only after this manifest passes should contact-memory model code begin.

## Implementation Hash

```text
scripts/audit_epic_contact_gate.py
c5e325dc2de30060ef92c2944c37f3751ef0f11c2fd22590ed9562dc227f3b15
```
