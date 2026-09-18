# ContactOpt on Real MaMi Outputs: Phase 8 Replication

Date: 2026-09-18

Raw results:

```text
docs/experiments/contactopt-mami-phase8-largetable-2026-09-18.json
docs/experiments/contactopt-mami-phase8-plasticbox-2026-09-18.json
```

## Scope

Repeat the frozen Phase 7 protocol on two additional objects and sequences.
The comparison tests whether the Phase 7 success was object-specific and
whether ContactOpt remains safe when the input hand is not actually near the
object.

## Frozen Protocol

All runs use:

```text
10 selected frames
--min-frame-separation=5
w_opt_rot=0
w_opt_trans=0
w_obj_rot=0
rand_re=0
n_iter=250
```

The wrist root, object frame, and both `rand_re` and object rotation are
frozen. Only the 15-dimensional MANO finger PCA pose is optimized.

## Results

| Case | Object | Canonical alignment | MANO alignment | Contact improved | Distance improved | Mean distance change | Mean contact change | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Phase 7 | monitor | 1.564 mm | 11.463 mm | 10/10 | 10/10 | -40.22% | +154.20% | PASS |
| Phase 8A | largetable | 1.294 mm | 11.543 mm | 7/10 | 7/10 | -19.38% | +56.69% | PASS |
| Phase 8B | plasticbox | 2.033 mm | 11.469 mm | 3/10 | 4/10 | -0.79% | +2.93% | FAIL |

In every run, the maximum wrist drift was exactly `0.0 m` and the maximum
object-vertex drift was at most `1.67e-8 m`.

## Failure Interpretation

The `plasticbox` candidate did not fail because of coordinate drift,
optimization instability, or a wrist/object change. Its selected frames have
input mean hand-object distances of roughly `8.6-15.2 cm`, and the
ContactOpt contact signal is effectively absent. The model therefore refined
toward a weak or invalid contact target and changed the hand pose by about
`24.4 mm` on average without producing credible contact improvement.

The `monitor` and `largetable` cases were in a contact-relevant regime and
passed the same frozen gate.

## Decision

Conditional GO for ContactOpt as a finger/contact refinement teacher, but
only behind an explicit contact-eligibility gate.

The next implementation requirement is:

```text
if the input right hand is not in a valid contact/proximity regime:
    do not invoke ContactOpt as a contact teacher
```

This is now part of the E3D action contract rather than an optional
preprocessing detail. A refinement module that is called on a no-contact
sample can create a plausible-looking finger pose with no contact evidence.

The result still does not establish generalization across all objects, does
not recover true `pose_hand`, and does not justify finger WM training.
