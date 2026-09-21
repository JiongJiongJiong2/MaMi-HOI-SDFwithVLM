# OakInk C0 Future-Conditioning Protocol

Date: 2026-09-21

Status: frozen before execution

## Goal

Determine, using CPU-only data processing, whether OakInk contains the
event geometry needed for a later controlled experiment on whether a
future receiver requirement changes an earlier giver grasp.

C0 does not train a generative model, does not run ContactOpt, and does
not claim that future conditioning is beneficial.

## Research Question

Given the same object and an oracle-fixed transfer/endpoint object pose,
does adding the correct future giver/receiver role and target contact
region change the preferred pre-transfer giver grasp more than:

```text
no future condition;
endpoint object pose only;
shuffled receiver target from the same object.
```

Only the data requirement for this question is tested in C0.

## Inputs

Primary source:

```text
E:/HOI/TMP/OakInk-v1/Image/anno_v2.1.zip
E:/HOI/TMP/OakInk-v1/shape/OakInkObjectsV2.zip
E:/HOI/TMP/OakInk-v1/shape/metaV2.zip
```

Frozen event source:

```text
oakink_handover_manifest_v1_20260921/
  handover_trajectories.npz
  role_switch_candidates.jsonl.gz
  summary.json
```

The participant split, primary-event rule, and `5 mm` contact threshold
are unchanged.

## Event Definition

For each retained handover sequence, select one primary `5 mm`
role-switch candidate using the existing frozen rule:

1. smallest absolute onset-minus-release offset;
2. earliest giver release as the tie-breaker.

The event center is the selected giver release. The event window is:

```text
[release - 30, release + 30]
```

Frames outside the sequence are padded and marked invalid. No frame at
or after release is used to define the current giver state.

## Stored State

For every valid frame in the event window:

```text
giver hand vertices, 778 x 3
receiver hand vertices, 778 x 3
giver hand joints, 21 x 3
receiver hand joints, 21 x 3
object transform, 4 x 4
```

The local object mesh is stored once per object id. Camera-space object
vertices are reconstructed as:

```text
object_transform @ homogeneous(local_object_vertices)
```

## Contact Regions

For a hand and object vertex set:

```text
distance(v) = minimum Euclidean distance from object vertex v
              to the hand vertices
contact(v)  = distance(v) <= 5 mm
```

The current giver region is computed at:

```text
release - 15
```

The future receiver region is the stable union of receiver contact
vertices over:

```text
max(release, receiver_start) to receiver_end
```

If the selected receiver endpoint lies outside the event window, the
intersection with valid window frames is used and recorded.

## Future-Conditioning Fields

Each event stores object-frame summaries so a later optimizer can build
correct and shuffled conditions without reading the raw zip:

```text
current giver contact-vertex indices and distances
future receiver contact-vertex indices and distances
current giver palm/joint pose in object frame
future receiver palm/joint pose in object frame
object release pose and transform
receiver target centroid, covariance, and radius in object frame
```

These fields are oracle targets from the observed sequence. They are not
claimed to be causally necessary for a different valid handover.

## C0 Feasibility Gates

C0 passes only if:

1. at least 60 primary events have a valid 61-frame event window with at
   least 45 valid frames;
2. every retained split has at least 12 such events;
3. all event windows have valid giver vertices, receiver vertices,
   object transforms, and local object meshes;
4. at least 80% of events have non-empty giver and receiver `5 mm`
   contact regions at their respective reference frames;
5. at least 20 same-object groups contain two events whose receiver
   target centroids differ by at least 5% of the object's canonical
   diameter;
6. at least 20 events contain a pre-transfer frame in which both hands
   are within 20 mm of the object but the giver is still the active
   holder, providing within-sequence hard controls.

Failure stops C0 before optimizer implementation. Partial failure is
reported as CONDITIONAL only when the missing condition can be repaired
without changing the source event definition.

## Required Outputs

```text
events.npz
objects.npz
manifest.jsonl
summary.json
```

The output must be compact enough to upload to the existing no-card
server without copying the raw OakInk archives.

## Boundary

The observed receiver target is not a counterfactual outcome. C0 only
establishes whether the data can construct the required controlled
states. Whether the condition changes a generated giver grasp is a later
oracle optimization experiment.

The current OakInk test has already participated in earlier
post-processing choices. It remains useful for C0 development but cannot
serve as a fresh final external test.
