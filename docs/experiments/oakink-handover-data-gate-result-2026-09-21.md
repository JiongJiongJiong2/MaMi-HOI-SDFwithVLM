# OakInk Handover Data-Gate Result

Date: 2026-09-21

Status: complete with participant-disjoint gate PASS

Protocol:

```text
docs/experiments/oakink-handover-data-gate-protocol-2026-09-21.md
```

Summary:

```text
docs/experiments/oakink_handover_gate_summary_v1_20260921.json
```

## Data

OakInk intent id `0004` provides explicit two-person handover labels:

```text
handover sequences:  194
subjects:             11
objects:             100
camera-0 frames:   19,700
```

The geometry gate uses:

```text
hand_v:              778 hand vertices
OakInkObjectsV2:     downsampled object meshes
obj_transf:          object pose in camera space
```

## Participant-Disjoint Split

Subject groups were fixed before geometry processing:

```text
train: {0,5,6,9,10}
val:   {2,4,7,8}
test:  {1,3}
```

Cross-group subject pairs were dropped.

Retained sequences:

| Split | Subjects | Sequences | Frames | Objects |
|---|---:|---:|---:|---:|
| train | 5 | 44 | 5,189 | 23 |
| val | 4 | 41 | 4,830 | 21 |
| test | 2 | 41 | 3,192 | 21 |

No subject or ordered giver/receiver pair crosses a split.

## Primary 5 mm Results

| Split | Both Contact Sequences | Role-Switch Sequences | Candidates | Giver Contact Frames | Receiver Contact Frames |
|---|---:|---:|---:|---:|---:|
| train | 44 / 44 | 33 / 44 | 36 | 3,292 | 2,016 |
| val | 41 / 41 | 36 / 41 | 37 | 2,709 | 2,110 |
| test | 41 / 41 | 37 / 41 | 37 | 1,944 | 1,452 |

All three splits retain both-subject object contact and most sequences
contain a geometrically detectable giver-to-receiver transition.

## Threshold Sensitivity

Role-switch-detectable sequences:

| Threshold | Train | Val | Test |
|---:|---:|---:|---:|
| 1 mm | 9 / 44 | 4 / 41 | 2 / 41 |
| 3 mm | 34 / 44 | 36 / 41 | 38 / 41 |
| 5 mm | 33 / 44 | 36 / 41 | 37 / 41 |
| 10 mm | 32 / 44 | 32 / 41 | 39 / 41 |

The transition signal is stable at `3-10 mm`; `1 mm` is too strict for
this dataset.

## Gate

```text
each split has >= 20 sequences:                  pass
each split has >= 2 participants:                pass
each split has >= 5 object categories:           pass
both-contact rate >= 0.90:                       pass
role-switch sequence rate >= 0.50:               pass
overall:                                         PASS
```

## Relationship to ARCTIC

ARCTIC supplied geometric role-switch candidates but no explicit
handover labels and only one accessible official test subject.

OakInk supplies:

```text
explicit handover intent;
explicit giver and receiver identities;
100 object categories;
participant-disjoint train/val/test with 41-44 sequences each;
geometry-verifiable contacts.
```

OakInk is therefore the primary external handover dataset. ARCTIC
remains a supplementary bimanual-contact dataset.

## Execution Boundary

The large raw OakInk archives remain local because the server data disk
cannot safely hold both raw packages and derived data. The compact
derived manifest was uploaded to the server; all subsequent model
training and evaluation will run on the server.

Server paths:

```text
/root/autodl-tmp/oakink_handover_manifest_v1_20260921
/root/autodl-tmp/oakink_scripts
```

Server manifest verification:

```text
sequences:  126
gate:       train PASS, val PASS, test PASS
failures:   0
```

## Decision

Proceed with OakInk as the main handover route. The next step is to
build event-centered giver/receiver transition windows from this
manifest and train participant-disjoint handover prediction models.

## Implementation Hashes

```text
scripts/build_oakink_handover_manifest.py
3461a6d0ab57b0da7825137f9bcc14d3ec8c8647ca3718c1ac199dc841c46784

tests/test_build_oakink_handover_manifest.py
8dd7fe5466ae0a7adad022900c4251cb6735c8d19dc5809be2ac6c0d4e9f21a0

summary
53b6326aa57e0d1797276bbf49e73a59d07b7e55c1101975c7c1a299cc50574e
```
