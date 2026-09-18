# ContactOpt Cross-Object Eligibility Cohort: Phase 10

Date: 2026-09-18

Raw results:

```text
docs/experiments/contactopt_phase10_manifest.json
docs/experiments/phase10_cohort/cohort_summary.json
docs/experiments/phase10_cohort/*.json
```

## Frozen Selection

The object list and lexicographic selection rule were frozen before the run.
Six objects had saved-vertex candidates:

```text
monitor, largetable, plasticbox, smallbox, floorlamp, whitechair
```

Missing saved-vertex candidates:

```text
clothesstand, trashcan, woodchair, suitcase
```

No ContactOpt outcome was used to select the six files. The manifest is
`contactopt_phase10_manifest.json`.

## Frozen Gate

```text
p10_distance <= 0.02 m
at least 3 eligible frames per case
required successful frames = max(3, ceil(0.7 * eligible frames))
w_opt_rot = 0
w_opt_trans = 0
w_obj_rot = 0
rand_re = 0
```

## Cohort Results

| Object | Eligible | Contact improved | Distance improved | Mean distance change | Mean contact change | Gate |
|---|---:|---:|---:|---:|---:|---|
| monitor | 10 | 10/10 | 9/10 | -6.02% | +44.66% | PASS |
| largetable | 10 | 10/10 | 10/10 | -36.89% | +95.02% | PASS |
| plasticbox | 1 | not run | not run | not run | not run | REJECT |
| smallbox | 10 | 7/10 | 9/10 | -27.02% | +83.54% | PASS |
| floorlamp | 10 | 5/10 | 10/10 | -29.20% | +64.97% | FAIL |
| whitechair | 8 | 5/8 | 8/8 | -32.36% | +91.16% | FAIL |

`floorlamp` and `whitechair` did not fail because of coordinate drift or
distance regressions. They reached the required distance improvement but not
the required fraction of contact-weight improvements.

## Aggregate

| Metric | Result | Frozen gate | Status |
|---|---:|---:|---|
| Selected objects with saved vertices | 6 | >= 6 | PASS |
| Eligible cases | 5 | >= 4 | PASS |
| Cases passing the full per-case gate | 3 | >= 3 contact-success cases | PASS |
| Maximum wrist drift | 0.0 m | <= 0.00001 m | PASS |
| Maximum object drift | `2.98e-8 m` | <= `1e-6 m` | PASS |
| Eligible cases with large distance regression | 0 | 0 | PASS |

Phase 10 gate: PASS.

## Decision

The geometry gate is usable as a safety gate on this deterministic cohort:
it rejected the far-from-contact `plasticbox` case before ContactOpt and
preserved wrist/object coordinates in every invoked case.

ContactOpt refinement is not uniformly successful across eligible cases.
Three of five eligible cases met the full contact-improvement criterion;
`floorlamp` and `whitechair` improved nearest distance but not the required
contact fraction. They remain negative results and must not be relabeled as
successes.

This cohort still uses neutral SMPLX hand geometry, one candidate per object,
and a 2 cm p10 threshold selected from an earlier three-case audit. It does
not establish a universal contact detector or justify finger WM training.
