# Geometry-Only Contact Eligibility Gate: Phase 9

Date: 2026-09-18

Raw results:

```text
docs/experiments/contact_eligibility_phase9.json
docs/experiments/phase9_monitor.json
docs/experiments/phase9_largetable.json
docs/experiments/phase9_plasticbox.json
```

## Frozen Rule

Eligibility uses geometry only:

```text
p10_distance = 10th percentile of nearest
               right-hand-vertex-to-object-vertex distance

eligible_frame = p10_distance <= 0.02 m
eligible_case  = at least 3 eligible frames
```

ContactOpt is not invoked when the case has fewer than three eligible frames.
The rule was frozen before the gated reruns.

## Feature Audit

The audit covers all 30 previously executed frames.

| Case | Mean p10 distance | Mean 3 cm contact fraction | Eligible frames |
|---|---:|---:|---:|
| monitor | 17.755 mm | 21.51% | 7/10 |
| largetable | 5.623 mm | 74.00% | 10/10 |
| plasticbox | 23.438 mm | 15.41% | 1/10 |

The plasticbox-only eligible frame is frame `111`, which had low proximity
support (`p10 = 8.766 mm`) and did improve under the ungated run. The other
nine plasticbox frames were excluded.

## Gated Results

| Case | Eligible frames | Contact improved | Distance improved | Mean distance change | Mean contact change | Gate |
|---|---:|---:|---:|---:|---:|---|
| monitor | 7 | 7/7 | 7/7 | -38.47% | +137.50% | PASS |
| largetable | 10 | 7/10 | 7/10 | -11.54% | +43.65% | PASS |
| plasticbox | 1 | not run | not run | not run | not run | REJECT |

All invoked refinements retained:

```text
maximum wrist-root drift:  0.0 m
maximum object drift:      <= 1.67e-8 m
```

## Decision

GO for the geometry-only eligibility gate as a prerequisite for ContactOpt.

The E3D action contract now includes:

```text
ContactEligible_t =
    1[p10(hand-surface-to-object-surface distance) <= 0.02 m]

FingerTeacherApplied_t =
    ContactEligible_t * ContactOptFingerRefinement_t
```

The teacher remains forbidden from changing wrist translation, wrist rotation,
or the object frame.

## Limitations

The threshold was selected from three cases and 30 frames. It is a safety
gate, not a calibrated universal contact detector. It must be rechecked across
more subjects, object scales, and hand sizes before being frozen for training.

This does not recover real `pose_hand`, does not create object motion, and
does not justify finger world-model training by itself.
