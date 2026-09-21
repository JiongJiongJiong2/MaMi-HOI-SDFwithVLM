# OakInk C0-R1 Oracle Reranking Protocol

Date: 2026-09-21

Status: frozen before execution

## Goal

Test whether an oracle future receiver target changes the ranking of
plausible pre-transfer giver grasps relative to:

```text
no future target;
correct same-object receiver target;
deterministically shuffled same-object receiver target.
```

This is a retrieval diagnostic. It does not train a model or generate a
new grasp.

## Data

```text
/root/autodl-tmp/oakink_c0_event_dataset_v1_20260921/events.npz
/root/autodl-tmp/oakink_c0_event_dataset_v1_20260921/objects.npz
/root/autodl-tmp/oakink_c0_event_dataset_v1_20260921/targets.npz
```

Only complete events with at least 45 valid frames and valid current
giver and receiver references are eligible.

## Candidate Construction

For a target event `i`, its candidate set is every eligible event with
the same `object_id`, including `i` itself.

The candidate giver state is the hand pose at event position 15:

```text
release - 15
```

For candidate event `j`, its giver vertices are transformed into the
object canonical frame:

```text
H_j_local = inverse(T_j_current) @ H_j_world
```

Candidates are compared only after this object-frame alignment.

The target receiver state is the first valid frame in the target's
receiver interval. Its hand vertices are also transformed into the
object canonical frame.

## Scores

For each candidate:

```text
contact_score
  = negative mean of the 50 nearest candidate-hand distances
    to the object mesh

contact_coverage
  = fraction of candidate hand vertices within 5 mm of the object

region_clearance
  = clipped minimum candidate-hand distance to the correct or shuffled
    receiver contact region

hand_clearance
  = clipped 10th percentile candidate-hand distance to the target or
    shuffled receiver hand
```

All score components are standardized across the candidate set, with
zero variance replaced by zero.

The frozen primary future score is:

```text
contact_z
+ 0.5 * region_clearance_z
+ 0.5 * hand_clearance_z
```

The no-future score is:

```text
contact_z
```

The shuffled condition replaces both the contact region and receiver
hand with a deterministic different event from the same object.

Secondary frozen sensitivity weights are `0.25` and `1.0` for both
future terms. They are reported but do not replace the primary gate.

## Endpoint Pose Control

The target object pose is shared by all three arms because candidates
are expressed in the object canonical frame. Therefore "endpoint object
pose only" is a shared control, not a separate discriminator, at this
stage. An independent endpoint-only arm becomes meaningful only when a
world-space optimizer or trajectory generator is evaluated.

## Evaluation

The condition score does not use all target receiver vertices. The
evaluation uses:

```text
true target receiver hand clearance;
true target receiver contact-region clearance;
candidate contact coverage.
```

Evaluation score:

```text
contact_coverage
+ 0.5 * clipped(true_region_clearance / 30 mm)
+ 0.5 * clipped(true_hand_clearance / 50 mm)
```

## Metrics

For every eligible target event with at least two candidates:

```text
top-1 selected candidate;
whether top-1 is the observed target-event giver;
rank of the observed target-event giver;
selection change from no-future;
evaluation score of the selected candidate;
paired correct-minus-shuffled evaluation score.
```

Participant/object-grouped bootstrap uses 2,000 deterministic draws and
reports the 95% paired interval.

## Gate

C0-R1 passes only if:

1. at least 40 eligible events contain at least two same-object
   candidates;
2. the correct future score changes top-1 selection relative to
   no-future on at least 25% of eligible events;
3. the mean paired correct-minus-shuffled evaluation score is positive
   and its grouped 95% interval excludes zero;
4. the correct and no-future top-1 observed-event rates differ by no
   more than 0.10;
5. the primary result is directionally consistent at both secondary
   frozen weights.

Failure stops the full future-conditioned optimizer stage. A pass
authorizes only a bounded optimizer with matched budgets.

## Boundary

The candidate set contains observed successful grasps, not multiple
actions generated from one initial state. A positive result is oracle
ranking evidence, not physical causality, not a final OakInk benchmark,
and not evidence of direct MaMi transfer.
