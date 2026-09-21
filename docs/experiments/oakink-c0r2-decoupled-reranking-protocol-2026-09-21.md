# OakInk C0-R2 Decoupled Reranking Protocol

Date: 2026-09-21

Status: frozen before execution

## Goal

Test whether an oracle future receiver contact region changes the
ranking of pre-transfer giver grasps, using an evaluation that does not
reuse the conditioning geometry.

## Eligibility

Unlike C0-R1, an event does not need at least 45 valid window frames. It
is eligible when:

```text
current giver position 15 is valid;
release position 30 is valid;
the receiver reference position is valid;
the receiver contact region is non-empty.
```

This is the `reference-valid` universe established by the C0-R1 audit.

## Candidate Set

For target event `i`, use every reference-valid event with the same
`object_id`. Candidate giver vertices are expressed in the object
canonical frame:

```text
H_j_local = inverse(T_j_current) @ H_j_world
```

All current candidate sets are expected to contain two events.

## Conditioning

Conditioning uses only the future receiver object-contact region:

```text
region_clearance
  = clipped minimum candidate-hand distance to the target or donor
    receiver contact region
```

The correct condition uses the target event's region. The shuffled
condition uses the next deterministic different same-object event.

Condition score:

```text
contact_z + weight * region_clearance_z
```

Primary weight is `0.5`. Secondary weights are `0.25` and `1.0`.

No receiver hand or joint geometry is used for conditioning.

## Evaluation

Evaluation uses the target receiver's 21 hand joints, which never enter
the condition:

```text
joint_clearance
  = clipped 10th percentile nearest distance from candidate giver
    vertices to receiver joints

collision_fraction
  = fraction of candidate giver vertices within 5 mm of a receiver joint

evaluation_score
  = contact_coverage
  + 0.5 * joint_clearance
  - collision_fraction
```

This metric is independent of the conditioning field, although both are
derived from the same observed successful event.

## Metrics

For every target with at least two candidates:

```text
top-1 observed-event rate for no-future, correct, shuffled, and
independent-evaluation oracle;
paired correct-minus-no-future evaluation score;
paired correct-minus-shuffled evaluation score;
paired collision-fraction differences;
exact McNemar checks for top-1 transitions.
```

Object-grouped bootstrap uses 2,000 deterministic draws.

## Gate

GO requires all of:

1. at least 60 eligible target events;
2. the independent evaluation oracle selects the observed target-event
   giver on at least 55% of events;
3. correct-minus-no-future evaluation has positive mean and grouped 95%
   interval lower bound above zero;
4. correct-minus-shuffled evaluation has positive mean and grouped 95%
   interval lower bound above zero;
5. correct top-1 observed-event rate is no more than 0.05 below
   no-future;
6. correct selected collision fraction is no worse than shuffled, with
   grouped 95% interval upper bound below zero for
   `correct - shuffled`;
7. both weights `0.5` and `1.0` show positive correct-minus-both-baseline
   mean evaluation.

Failure is NO-GO for C under this data. A pass authorizes a bounded
future-conditioned optimizer with matched budgets.

## Boundary

The evaluation geometry comes from the same successful event and is not
a counterfactual outcome. The result is oracle ranking evidence, not
physical causality, a final benchmark, or direct MaMi transfer.
