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

## Decision

The analytic selector improves stable contact success from `0.7051` to
`0.8077` and raises recall from `0.7297` to `0.8718`. However, precision is
essentially unchanged and selected actions create more early contact frames.
The current mechanism therefore appears to buy contact coverage and episode
length rather than removing wrong or premature contact.

This supports the next E1b step: joint bimanual ranking and contact-risk
scoring must optimize precision, onset delay, and false-contact penalties,
not only maximum frame contact or recall.

## Evidence

Server summary:

```text
/root/autodl-tmp/contact_action_20260914/analytic_contact_episode_v1_20260918/summary.json
```

Server cohorts:

```text
/root/autodl-tmp/contact_action_20260914/analytic_contact_episode_v1_20260918/{10_30,31_50,51_70}
```

Implementation commit:

```text
26d6013 fix(world-model): correct contact episode edge metrics
```

