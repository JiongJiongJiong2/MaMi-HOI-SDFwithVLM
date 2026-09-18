# ContactOpt Contact Metric Repair

Date: 2026-09-18

## Trigger

Phase 10 initially reported that `floorlamp` and `whitechair` were eligible for
ContactOpt refinement but failed the required fraction of contact-improved
frames. Their nearest-distance metrics improved in every or nearly every
frame, which made the contact failure suspicious.

## Root Cause

ContactOpt computes contact as a normalized capsule SDF. Its source notes that
the capsule-axis normalization can produce invalid values when the capsule
length is zero. These values propagate as `NaN` in `HandObject.hand_contact`.

The original evaluator called `.mean()` on the full contact array. One `NaN`
therefore made the mean for the whole hand `NaN`, and the frame was treated as
having no valid contact measurement. Across the five invoked Phase 10 cases,
11 of 48 frames contained at least one non-finite contact value.

This was an aggregation error, not evidence that ContactOpt failed to improve
contact.

## Repair

The metric now:

1. replaces non-finite capsule contact values with zero contact;
2. computes the mean over all hand vertices;
3. records the finite fraction separately for input and refined arrays.

The repair was applied to:

```text
scripts/contactopt_contact_metrics.py
scripts/evaluate_contactopt_smoke.py
scripts/run_mami_contactopt_case.py
scripts/recompute_contactopt_case_contact.py
```

Existing `optimized_phase10_*.pkl` files were re-evaluated directly. No
ContactOpt optimization and no coordinate-changing computation were rerun.

## Corrected Cohort

| Object | Eligible | Contact improved | Distance improved | Mean contact change | Gate |
|---|---:|---:|---:|---:|---|
| monitor | 10 | 10/10 | 9/10 | +44.66% | PASS |
| largetable | 10 | 10/10 | 10/10 | +95.02% | PASS |
| plasticbox | 1 | not run | not run | not run | REJECT |
| smallbox | 10 | 10/10 | 9/10 | +71.63% | PASS |
| floorlamp | 10 | 10/10 | 10/10 | +98.64% | PASS |
| whitechair | 8 | 8/8 | 8/8 | +79.99% | PASS |

Aggregate:

```text
eligible cases:              5/5
eligible contact successes:  5/5
invoked frames:              48
contact-improved frames:     48
distance-improved frames:    46
maximum wrist drift:         0.0 m
maximum object drift:        2.98e-8 m
```

The corrected Phase 10 gate remains PASS. The conclusion changes from three
successful eligible cases to five; one requested case (`plasticbox`) is still
rejected before refinement by the frozen geometry gate.

## Boundary

Replacing an undefined capsule value with zero is a conservative contact
encoding consistent with the capsule definition, but it does not recover true
surface contact. The corrected result still uses aligned neutral MANO hands,
one MaMi candidate per object, and a 2 cm p10 eligibility threshold. It does
not justify finger world-model training by itself.

## Evidence

```text
docs/experiments/phase10_cohort/cohort_summary.json
docs/experiments/phase10_cohort/*.json
```
