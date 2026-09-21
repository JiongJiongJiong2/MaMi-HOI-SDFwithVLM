# OakInk C0 Future-Conditioning Data Result

Date: 2026-09-21

Status: C0 data-feasibility gate PASS; oracle optimization not run

Protocol:

```text
docs/experiments/oakink-c0-future-conditioning-protocol-2026-09-21.md
```

Summary:

```text
docs/experiments/oakink_c0_feasibility_summary_v1_20260921.json
```

## Scope

C0 tested whether OakInk contains the event-centered geometry needed for
a later controlled comparison between:

```text
no future condition;
endpoint object pose only;
correct receiver-hand target;
shuffled receiver target from the same object.
```

The audit is CPU-only. It does not train a model, run ContactOpt, or
claim that future conditioning improves a generated grasp.

## Dataset

The raw OakInk annotation archive was reduced locally to compact event
arrays before upload. The server was used for the full audit because it
was running in no-card mode and the user requires experiments to run
there.

```text
primary events:       106
split events:         33 train, 36 val, 37 test
unique object meshes: 65
event window:         release - 30 to release + 30
compact data size:    about 94 MB
```

Server paths:

```text
/root/autodl-tmp/oakink_c0_scripts
/root/autodl-tmp/oakink_c0_event_dataset_v1_20260921
```

## Geometry Validation

Twelve complete events were reconstructed from the compact local object
meshes and stored per-frame object transforms. The reconstructed
giver/receiver minimum hand-object distances were compared with the
frozen manifest distances:

```text
events checked:                 12
maximum absolute error:         1.64e-08 m
required tolerance:             1e-05 m
```

This confirms that the stored hand vertices, local object meshes, and
object transforms reconstruct the original geometry convention within
numerical precision.

## Event Validity

```text
events with valid window >= 45 frames: 82 / 106
split complete events:                 23 train, 30 val, 29 test
valid frame range:                     36 to 61
current giver contact regions:         106 / 106
future receiver target regions:        106 / 106
contact-region coverage fraction:      1.0000
```

All events contain a non-empty `5 mm` giver region at the pre-release
reference and a non-empty receiver target over the receiver contact
interval.

## Conditioning Contrast

There are 41 object groups containing at least two C0 events. In 21 of
those groups, two receiver target centroids differ by at least 5% of the
object bounding-box diameter:

```text
same-object groups:              41
groups supporting target contrast: 21
minimum required:                20
```

The event data also provides within-sequence controls in which both
hands are near the object before release while the giver is still the
active holder:

```text
events with hard control: 79
minimum required:         20
```

## Gate

```text
at least 60 complete events:                    pass
at least 12 complete events in every split:     pass
geometry reconstruction within tolerance:       pass
contact regions in at least 80% of events:      pass
at least 20 same-object contrast groups:         pass
at least 20 hard-control events:                 pass
overall:                                         PASS
```

## Interpretation

OakInk supports construction of the required controlled states. It is
now possible to express the current giver grasp, the future receiver
target, and a shuffled target in one object frame while preserving
participant-disjoint event provenance.

This result does not establish causal benefit. The observed receiver
target is an oracle from one successful interaction, not a
counterfactual outcome. The next stage must freeze an optimizer and
compare correct versus shuffled future conditions under equal budgets.

The current OakInk test split participated in earlier post-processing
design. C0 may use it for development, but a final external claim still
requires a fresh outer split.

The MaMi task mapping also remains open: OakInk is cross-person, while
MaMi is not automatically the same action schema.

## Server Verification

The no-card server executed the complete feasibility audit:

```text
Python: /root/miniconda3/bin/python3
GPU:    none
tests:  5 passed with unittest
```

## Implementation Hashes

```text
scripts/build_oakink_c0_event_dataset.py
345c3e209164fed5df340a39c476fbb972a3dcd1510dab888cff496fdd64d6a7

scripts/analyze_oakink_c0_feasibility.py
45f6db4fbcfdbb62f65e0585fc44f5880051ea89f86dc3a6f91c7df7a8ec6d15

tests/test_build_oakink_c0_event_dataset.py
baab75ba9c1680df81b2e42bdbc259a5570dd5c804991420eb192d7325eb3e2b

events.npz
fb628f7c9e50d380ca11bf0a35c4c2d14a35ddd2fcb65a8b4ffe41210b5964f8

objects.npz
2ac3408005dc8cf818756c458ecc512c59f3ac14c7f8e0f0492e482403136304

targets.npz
64ec9fbcc267d9885f239f9d132f2ca7cb47e6ce1e94b5cfd8fc61beeec0c3f4

manifest.jsonl
dd823deef245e2244224ab1a4b5a7115a91364a71918fc95463f1245586e17d3

server feasibility.json
6e77024c642c662b8cf3fe01cbbbcfe8ed0e66aac289154f58d2247157d69dd7
```

## Decision

```text
C0 data feasibility:        GO
future-conditioned optimizer: authorized as the next stage
causal or paper claim:      not yet supported
```
