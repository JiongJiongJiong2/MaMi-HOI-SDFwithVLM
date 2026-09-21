# OakInk Handover Data-Gate Protocol

Date: 2026-09-21

Status: frozen before execution

## Goal

Determine whether OakInk provides a sufficiently large,
participant-disjoint, geometrically verifiable handover dataset.

OakInk intent id `0004` is an explicit two-person handover label. The
sequence id contains:

```text
object_id_intent_id_giver_subject_id_receiver_subject_id
```

## Data Boundary

```text
handover sequences:  194
subjects:             11
objects:             100
unique frames:     19,700
views per frame:        4
```

Only camera 0 is used for the first geometry gate. Image files are not
used.

## Participant-Disjoint Split

Subject groups are fixed before geometry processing:

```text
train: {0,5,6,9,10}
val:   {2,4,7,8}
test:  {1,3}
```

Only sequences whose giver and receiver are in the same group are
retained. Cross-group sequences are dropped.

Expected sequence counts:

```text
train: 44
val:   41
test:  41
```

## Geometry

For each frame:

```text
hand vertices: hand_v, 778 vertices per subject
object model: OakInkObjectsV2/<name>/align_ds/*.obj
object pose:  obj_transf, T_camera_object
```

The minimum Euclidean distance from each subject's hand vertices to the
transformed object vertices defines contact.

Primary contact threshold:

```text
5 mm
```

Sensitivity thresholds:

```text
1 mm, 3 mm, 5 mm, 10 mm
```

A stable contact segment requires at least 15 consecutive frames.

## Role-Switch Detection

A sequence is role-switch detectable when a stable giver segment and a
stable receiver segment have onset/release timing within 30 frames,
using only the two subjects' hand-object contact trajectories.

No image-based annotation is used in this gate.

## Gate

The OakInk data gate passes only if, at `5 mm`:

1. each split has at least 20 retained handover sequences;
2. each split has at least two distinct participants;
3. each split has at least five object categories;
4. at least 90% of retained sequences contain contact for both the
   giver and receiver;
5. at least 50% of retained sequences contain a detectable giver-to-
   receiver role switch.

Failure stops OakInk handover modeling and triggers the next dataset
audit.

## Outputs

```text
summary.json
handover_trajectories.npz
role_switch_candidates.jsonl.gz
```
