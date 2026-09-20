# ARCTIC Handover Data-Gate Protocol

Date: 2026-09-21

Status: frozen before execution

## Goal

Determine whether the publicly released ARCTIC sequences contain enough
geometrically verified bimanual contact and hand-to-hand transfer
candidates for a role-switch study.

This is a data gate, not a model experiment and not a handover claim.

## Data Boundary

The public raw release contains:

```text
subjects:                 9
sequences:              301
official train:         267
official val:            34
official test:            0
```

Official test subject `s03` is withheld. No public ARCTIC result may be
described as an official test result.

All accessible sequences use the official participant split:

```text
train: 8 subjects
val:   1 subject
```

## Geometry

Hand vertices come from the MANO parameters in `*.mano.npy`.

Object vertices come from `meta/object_vtemplates/*/mesh.obj`. The top
articulated part is transformed by the object articulation angle, then
all vertices receive the object global orientation and translation.

For each frame and hand:

```text
min_distance_m = minimum Euclidean distance from a MANO vertex
                 to an object-template vertex
```

The primary contact threshold is the ARCTIC official value:

```text
3 mm
```

Threshold sensitivity is reported at:

```text
1 mm, 3 mm, 5 mm, 10 mm
```

## Event Definitions

A stable contact segment requires:

```text
minimum contact duration:  15 consecutive frames
false-gap bridging:         0 frames
```

A bimanual overlap is a frame where both stable contact masks are true.

A role-switch candidate is emitted when one stable hand segment releases
and the other hand starts a stable segment within:

```text
release to receiving-onset window: 15 frames
```

The outgoing and receiving segments must each satisfy the stable-contact
duration rule. These are candidates only and must not be called verified
handovers without additional receiver-role validation.

## Gate

At the primary `3 mm` threshold, the pilot gate passes only if:

1. train has at least 10 role-switch candidates;
2. train candidates span at least three participants;
3. train contains at least two candidates in each direction;
4. train candidates cover at least three object categories;
5. train and val each contain at least one sequence with bimanual stable
   overlap;
6. val contains at least one role-switch candidate.

The official-test condition is always reported separately as
unavailable. A pilot pass cannot become a final benchmark claim.

## Execution

The server job writes:

```text
summary.json
distance_trajectories.npz
role_switch_candidates.jsonl.gz
run.log
SUCCESS or FAILED
exit_code
```

After the job exits, the launcher records the exit status, synchronizes
the filesystem, waits 30 seconds, and requests automatic shutdown.
