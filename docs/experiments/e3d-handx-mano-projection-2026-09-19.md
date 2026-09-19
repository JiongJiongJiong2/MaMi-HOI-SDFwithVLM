# E3D HandX-to-MANO Projection Diagnostic

Date: 2026-09-19

## Scope

Test a deterministic dual-hand projection from HandX 21-joint sequences to
legal MANO parameters without changing the injected wrist trajectory.

The implementation is:

```text
map MANO joints to HandX joint order
project each bone direction onto the MANO template bone length
fit MANO finger pose to the canonical skeleton
write the MANO state back to world coordinates with an exact wrist transform
```

No HandX backbone weights or world model were trained.

Implementation:

```text
scripts/project_handx_to_mano.py
tests/test_project_handx_to_mano.py
```

## Joint-Order Finding

`manopth.ManoLayer` returns 21 joints in a visualization order, not the
HandX order defined by `JOINT_NAME_INDEX_MAP`. The explicit projection order is:

```text
[0, 5, 6, 7, 9, 10, 11, 17, 18, 19,
 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20]
```

Ignoring this mapping produced fictitious bone-length variation and centimeter
joint errors.

## Results

### Official HandX Wrist Smoke

| Metric | Left | Right | Gate |
|---|---:|---:|---|
| Projected bone CV | `1.15e-7` | `1.30e-7` | PASS |
| Wrist RMSE | `2.25e-5 mm` | `2.37e-5 mm` | PASS |
| Canonical fit mean | `8.63 mm` | `5.70 mm` | PASS |
| Speed ratio | `1.075` | `1.159` | PASS |
| Acceleration ratio | `1.190` | `1.231` | PASS |
| Jerk ratio | `1.246` | `1.303` | PASS |

The official sample passes the deterministic projection gate.

### MaMi Clothesstand Wrist Condition

| Metric | Left | Right | Gate |
|---|---:|---:|---|
| Input bone CV | `16.04%` | `19.30%` | FAIL before projection |
| Projected bone CV | `1.77e-7` | `1.73e-7` | PASS |
| Wrist RMSE | `2.28e-5 mm` | `2.36e-5 mm` | PASS |
| Canonical fit mean | `7.24 mm` | `12.44 mm` | PASS |
| Raw generated fit mean | `102.72 mm` | `84.55 mm` | diagnostic |
| Speed ratio | `2.012` | `1.808` | left FAIL |
| Acceleration ratio | `2.877` | `3.536` | FAIL |
| Jerk ratio | `4.744` | `6.536` | FAIL |

### MaMi Monitor Wrist Condition

| Metric | Left | Right | Gate |
|---|---:|---:|---|
| Input bone CV | `18.54%` | `24.17%` | FAIL before projection |
| Projected bone CV | `3.12e-7` | `2.95e-7` | PASS |
| Wrist RMSE | `3.13e-5 mm` | `2.74e-5 mm` | PASS |
| Canonical fit mean | `9.42 mm` | `17.14 mm` | right FAIL |
| Speed ratio | `1.483` | `1.638` | PASS |
| Acceleration ratio | `2.424` | `4.489` | FAIL |
| Jerk ratio | `4.083` | `7.785` | FAIL |

## Output Smoothing Diagnostics

Two-stage output smoothing was tested on the clothesstand case. It preserved
the projected bone lengths and wrist but did not produce a reliable temporal
gate:

| Output kernel | Left accel | Left jerk | Right accel | Right jerk |
|---:|---:|---:|---:|---:|
| 1 | `2.877` | `4.744` | `3.536` | `6.536` |
| 5 | `2.554` | `3.573` | `3.492` | `6.054` |
| 9 | `2.482` | `3.492` | `3.468` | `6.264` |

Simply increasing fixed smoothing does not remove the worst temporal modes.

## Decision

```text
HandX-to-MANO deterministic canonicalizer: GO for legal geometry
Official HandX wrist smoke:                  PASS
MaMi wrist-conditioned temporal integration: NO-GO
```

The projection fixes the original 14.8-21.5 percent bone-length failure and
keeps the injected wrist exact. It does not make raw MaMi-wrist-conditioned
HandX output temporally usable. The remaining error is not a MANO
parameterization failure; it is a sequence-fidelity and temporal-consistency
failure that must be handled by the E5 sequence-level solver.

## Next Gate

Do not integrate these HandX projections into contact refinement yet. Freeze
the E5 contact-fidelity plus temporal-smoothness objective on train/dev and
validate on the untouched test cohort only after the solver configuration is
frozen.
