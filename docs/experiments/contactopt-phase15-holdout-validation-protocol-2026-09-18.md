# ContactOpt Temporal Holdout Validation Protocol: Phase 15

Date: 2026-09-18

Status: completed with conditional result

Result report:

```text
docs/experiments/contactopt-phase15-holdout-validation-2026-09-18.md
```

## Goal

Validate the fixed binomial-5 temporal smoother on contiguous windows that
were not used to design or inspect Phase 12-14 smoothing behavior.

## Frozen Selection

Source manifest:

```text
docs/experiments/contactopt_phase11_manifest.json
SHA-256 8c679a5e80bca1f63842819e988ba28ed94402ab3cbc425352577ead04d65fa5
```

Exclude the four development sequences:

```text
sub17_monitor_015
sub16_largetable_001
sub16_plasticbox_004
sub17_smallbox_015
```

Select the next four manifest candidates with a geometry-only contiguous run
of at least five frames satisfying `p10_distance <= 0.02 m`. Use the longest
run, tie by earliest, and take a centered window capped at eight frames.

Frozen manifest:

```text
docs/experiments/contactopt_phase15_manifest.json
SHA-256 8771fef65d0bec29a5151397bbde76be47676cbb185b08cab0d8ac3d9f7956f5
```

Windows:

```text
sub17_floorlamp_007    frames 33..40
sub16_whitechair_002   frames 14..19
sub16_clothesstand_001 frames 102..109
sub16_largebox_005     frames 69..76
```

## Frozen Method

1. Run independent ContactOpt on every frame with the Phase 12 settings.
2. Smooth the optimized output finger PCA coefficients `3:18` with
   `[1, 4, 6, 4, 1] / 16`.
3. Keep global pose coefficients `0:3` and `hand_mTc` unchanged.
4. Rerun MANO forward geometry.
5. Recompute contact and temporal metrics.

No threshold, kernel, case list, or frame window may change after execution.

## Gate

Use the Phase 12 promotion gate unchanged:

```text
contact-improved frames >= ceil(0.7 * eligible frames)
mean nearest-distance relative change <= +10%
maximum wrist-root drift <= 0.01 mm
maximum object drift <= 0.001 mm
refined/input speed ratio <= 2.0
refined/input acceleration ratio <= 2.0
refined/input jerk ratio <= 3.0
```

Decision:

```text
4/4 passes: GO for short-window temporal integration study
3/4 passes: CONDITIONAL
0-2/4 passes: NO-GO for this smoother
```

Passing this phase still does not justify world-model training by itself. It
only establishes a temporally stable local refinement baseline on held-out
windows.
