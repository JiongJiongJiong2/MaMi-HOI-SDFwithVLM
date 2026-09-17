# Contact Episode Baseline

Date: 2026-09-18

Status: complete

## Purpose

Extend the frozen analytic L1 baseline from frame-wise contact F1 to stable
contact episode metrics. Candidate generation, candidate selection, object
trajectory, checkpoint, and 50 mm contact threshold are unchanged.

## Protocol

```text
stable_min_frames: 3
horizon: 8
candidate_seed: 1
scorer_mode: geom_only
rollout_mode: analytic
penetration_weight: 20.0
```

The run uses the same 43 sequences and 78 hand events as the canonical
analytic baseline.

## Results

| Metric | Base | Random | Selected |
|---|---:|---:|---:|
| contact F1 | 0.4131 | 0.3696 | 0.4909 |
| contact precision | 0.4189 | 0.3737 | 0.4137 |
| contact recall | 0.7297 | 0.6747 | 0.8718 |
| stable contact success rate | 0.7051 | 0.6506 | 0.8077 |
| stable false-positive rate | 0.0256 | 0.0449 | 0.0256 |
| predicted contact episodes per event | 0.7692 | 0.7660 | 0.9103 |
| longest predicted episode, frames | 5.3974 | 5.1314 | 6.2436 |
| early contact frames before GT onset | 3.3333 | 3.2853 | 3.8590 |
| dropout count inside GT span | 0.0000 | 0.0000 | 0.0000 |

Release metrics are undefined for this first-onset cohort because GT contact
does not release inside most eight-frame evaluation windows.

## Bimanual Pair Headroom

The aligned pair analysis covers 22 sequences where left and right futures
overlap for at least two frames.

| Pair policy | joint F1 | joint precision | joint recall | both stable success |
|---|---:|---:|---:|---:|
| independent per-hand argmax | 0.4079 | 0.3581 | 0.8295 | 0.2273 |
| maximin balanced score | 0.3789 | 0.3782 | 0.7784 | 0.2273 |
| pair oracle | 0.5364 | 0.5259 | 0.8523 | 0.1818 |

The pair oracle is `+12.85 pp` joint F1 above the independent policy, with a
sequence-clustered 95% interval of `[+5.42, +22.10] pp`. Maximin changes 13
pairs but does not improve F1. This shows that joint candidate-pool headroom
exists, but a simple balance rule is not the right joint signal.

The pair oracle must not be interpreted as evidence for a bimanual interaction
signal. Because joint F1 is the mean of the two per-hand F1 values, this
oracle decomposes exactly into independent per-hand F1 oracles. The measured
headroom is therefore per-hand scorer miscalibration, not proof that the left
and right hands require a joint dynamics model.

## Learned Event-Head Check

The Stage 2 B checkpoint was used as a learned event-head scorer with
`event_weight=0.25`, while keeping the same candidates and penetration
weight. Seeds 11, 12, and 13 selected the same candidates and produced
identical aggregate metrics.

| Selection scorer | selected F1 | precision | recall | stable success |
|---|---:|---:|---:|---:|
| geometry only | 0.4909 | 0.4137 | 0.8718 | 0.8077 |
| B event head, weight 0.25 | 0.4880 | 0.4119 | 0.8590 | 0.7949 |

The learned event head does not improve candidate selection. The current
geometry-only scorer remains the selection baseline. This result also means
E1 should not proceed directly to a learned joint reranker.

## Decision

The analytic selector improves stable contact success from `0.7051` to
`0.8077` and raises recall from `0.7297` to `0.8718`. However, precision is
essentially unchanged and selected actions create more early contact frames.
The current scorer buys contact coverage and episode length rather than
removing wrong or premature contact. There is a large per-hand scoring gap,
but neither maximin nor the learned B event head captures it. E1 is therefore
closed with the geometry-only scorer retained.

The next experiment should either recalibrate per-hand candidate ranking
against episode-level utility, or move to E2 forward-inverse consistency.
There is not yet evidence that a joint bimanual reranker would help.

## Evidence

Server summary:

```text
/root/autodl-tmp/contact_action_20260914/analytic_contact_episode_v2_20260918/summary.json
```

Server cohorts:

```text
/root/autodl-tmp/contact_action_20260914/analytic_contact_episode_v2_20260918/{10_30,31_50,51_70}
```

Bimanual headroom analysis:

```text
/root/autodl-tmp/contact_action_20260914/analytic_contact_episode_v2_20260918/bimanual_headroom.json
```

Learned event-head runs:

```text
/root/autodl-tmp/contact_action_20260914/analytic_contact_episode_hybrid_b{11,12,13}_v1_20260918
```

Implementation commit:

```text
26d6013 fix(world-model): correct contact episode edge metrics
f97f357 feat(world-model): persist ground-truth contact windows
e302c1f feat(world-model): analyze bimanual pair headroom
```
