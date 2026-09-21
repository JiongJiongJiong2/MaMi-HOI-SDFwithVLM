# ARCTIC Role-Switch Candidate Quality Protocol

Date: 2026-09-21

Status: frozen before refined analysis

## Goal

Separate raw hand-transition candidates from stronger role-switch
candidates by requiring a clean release, a timely receiving onset, and
sustained receiving contact.

This refinement was motivated by the raw pilot result, which passed its
data gate but produced a large number of transitions that could include
ordinary bimanual support or temporary hand switching.

## Inputs

```text
/root/autodl-tmp/arctic_handover_gate_v1_20260921/
  distance_trajectories.npz
  role_switch_candidates.jsonl.gz
```

Primary contact threshold remains `3 mm`; threshold sensitivity remains
`1/3/5/10 mm`.

## Tier Definitions

All tiers require at least 15 consecutive receiving-contact frames after
the outgoing release.

### Tier A

```text
receiving onset offset from outgoing release: -5 to +5 frames
stable bimanual overlap before release:        <= 10 frames
outgoing hand remains off:                     >= 15 frames
```

### Tier B

```text
receiving onset offset from outgoing release: -10 to +10 frames
stable bimanual overlap before release:        <= 20 frames
outgoing hand remains off:                     >= 15 frames
```

### Raw

The existing candidate rule with a `-15` to `+15` frame onset window.

Tiers are nested in the order `Tier A -> Tier B -> Raw`.

## Primary Gate

At `3 mm`, Tier A passes the pilot refinement gate only if:

1. train has at least 10 Tier A candidates;
2. train Tier A candidates span at least three participants;
3. train Tier A contains at least two candidates in each direction;
4. train Tier A candidates cover at least three object categories;
5. val contains at least one Tier A candidate.

The official-test condition remains unavailable and is reported
separately. Passing this gate creates a pilot role-switch manifest, not
a final ARCTIC benchmark.

## Outputs

```text
candidate_quality_summary.json
tier_a_candidates.jsonl.gz
tier_b_candidates.jsonl.gz
```

Each refined candidate records onset offset, bimanual-overlap length,
outgoing-hand gap, receiving duration, and stable frames after release.
