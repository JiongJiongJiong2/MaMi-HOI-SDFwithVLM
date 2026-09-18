# ContactOpt Cross-Object Eligibility Cohort: Phase 10

Date: 2026-09-18

Raw results:

```text
docs/experiments/contactopt_phase10_manifest.json
docs/experiments/phase10_cohort/cohort_summary.json
docs/experiments/phase10_cohort/*.json
docs/experiments/contactopt-contact-metric-repair-2026-09-18.md
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
| smallbox | 10 | 10/10 | 9/10 | -27.02% | +71.63% | PASS |
| floorlamp | 10 | 10/10 | 10/10 | -29.20% | +98.64% | PASS |
| whitechair | 8 | 8/8 | 8/8 | -32.36% | +79.99% | PASS |

The initial report marked `floorlamp` and `whitechair` as failures because a
single non-finite ContactOpt capsule value made the direct array mean `NaN`.
The metric repair replaces undefined capsule contacts with zero while retaining
the finite fraction. Re-evaluating the saved pkl files does not rerun
optimization and changes no hand or object coordinates. See
`contactopt-contact-metric-repair-2026-09-18.md`.

## Aggregate

| Metric | Result | Frozen gate | Status |
|---|---:|---:|---|
| Selected objects with saved vertices | 6 | >= 6 | PASS |
| Eligible cases | 5 | >= 4 | PASS |
| Cases passing the full per-case gate | 5 | >= 3 contact-success cases | PASS |
| Maximum wrist drift | 0.0 m | <= 0.00001 m | PASS |
| Maximum object drift | `2.98e-8 m` | <= `1e-6 m` | PASS |
| Eligible cases with large distance regression | 0 | 0 | PASS |

Phase 10 gate: PASS.

## Decision

The geometry gate is usable as a safety gate on this deterministic cohort:
it rejected the far-from-contact `plasticbox` case before ContactOpt and
preserved wrist/object coordinates in every invoked case.

After correcting the undefined-contact aggregation, all five eligible cases
met the full contact-improvement criterion. This conclusion is conditional on
treating non-finite capsule outputs as zero contact rather than as missing
whole-frame observations. The raw finite fractions are retained in every row.

The earlier 3/5 result was an analysis-code false negative, not a new
refinement outcome. The original negative JSON and report remain in git
history. No ContactOpt optimization was rerun during the correction.

This cohort still uses neutral SMPLX hand geometry, one candidate per object,
and a 2 cm p10 threshold selected from an earlier three-case audit. It does
not establish a universal contact detector or justify finger WM training.
