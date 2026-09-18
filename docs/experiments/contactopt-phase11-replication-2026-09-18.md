# ContactOpt Cross-Object Replication: Phase 11

Date: 2026-09-18

Protocol and frozen manifest:

```text
docs/experiments/contactopt-phase11-protocol-2026-09-18.md
docs/experiments/contactopt_phase11_manifest.json
```

Raw results:

```text
docs/experiments/phase11_cohort/cohort_summary.json
docs/experiments/phase11_cohort/*.json
```

## Selection

Phase 11 used the pre-registered manifest with 108 saved-vertex candidates
inventory-wide and nine selected sequences. All Phase 7, Phase 8, and Phase 10
sequences were excluded before selection. The manifest SHA-256 is
`8c679a5e80bca1f63842819e988ba28ed94402ab3cbc425352577ead04d65fa5`.

Three selected objects were absent from the Phase 10 cohort:
`clothesstand`, `largebox`, and `trashcan`.

## Results

| Object | Eligible | Contact improved | Distance improved | Mean contact change | Gate |
|---|---:|---:|---:|---:|---|
| monitor | 10 | 10/10 | 10/10 | +63.83% | PASS |
| largetable | 10 | 10/10 | 10/10 | +80.86% | PASS |
| plasticbox | 10 | 8/10 | 10/10 | +9.36% | PASS |
| smallbox | 10 | 10/10 | 10/10 | +138.35% | PASS |
| floorlamp | 10 | 10/10 | 10/10 | +93.59% | PASS |
| whitechair | 1 | not run | not run | not run | REJECT |
| clothesstand | 6 | 6/6 | 6/6 | +64.35% | PASS |
| largebox | 10 | 10/10 | 10/10 | +71.71% | PASS |
| trashcan | 10 | 10/10 | 10/10 | +54.80% | PASS |

Aggregate across the eight invoked cases:

```text
eligible cases:               8/9
full gate passes:             8/8
eligible contact successes:   8/8
eligible frames:              76
contact-improved frames:      74/76
distance-improved frames:     76/76
mean contact change:          +72.52%
mean distance change:         -28.45%
maximum wrist drift:          0.0 m
maximum object drift:         1.19e-7 m
```

The lowest finite-contact fraction in an invoked frame was `78.41%`; 31 of
76 frames contained at least one non-finite capsule value. The repaired
contact policy handles these frames by treating undefined capsule contacts as
zero, while retaining the finite fraction in the raw JSON.

## Decision

Phase 11 promotion gate: PASS.

The teacher result replicates on all eight eligible pre-registered cases.
All three newly added objects pass, including `clothesstand` with six eligible
frames and `trashcan` and `largebox` with ten each. The previously rejected
`plasticbox` outcome does not generalize: a new sequence is eligible and
passes, although its contact margin is low at 8/10 frames and a mean contact
change of `+9.36%`.

`whitechair_002` is a preserved rejection, not a refinement failure. It had
only one frame satisfying the frozen `p10 <= 2 cm` eligibility rule, so
ContactOpt was not invoked.

The corrected Phase 10 and Phase 11 cohorts together support a GO for a
controlled finger-refinement integration study. They do not show object
motion, physical force, articulated hand generation, or temporal smoothness.
Those remain separate gates.
