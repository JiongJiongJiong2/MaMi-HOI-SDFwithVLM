# ContactOpt Dense Temporal Feasibility Protocol: Phase 12

Date: 2026-09-18

Status: frozen before execution

## Goal

Measure whether independent frame-wise ContactOpt refinement is temporally
usable on contiguous MaMi frames. Phase 10 and Phase 11 established contact
gain on sparse frames. Phase 12 does not add a learned temporal model yet; it
first measures whether such a model is already necessary.

## Frozen Selection

Source manifest:

```text
docs/experiments/contactopt_phase11_manifest.json
SHA-256 8c679a5e80bca1f63842819e988ba28ed94402ab3cbc425352577ead04d65fa5
```

Selection is geometry-only. Iterate the source manifest in order and select
the first four candidates containing a contiguous run of at least five frames
with:

```text
p10(hand-vertex-to-object-vertex nearest distance) <= 0.02 m
```

For each selected candidate, use the longest eligible run, break ties by the
earliest run, then take a centered window capped at eight frames. No
ContactOpt outcome is used for selection.

Frozen output:

```text
docs/experiments/contactopt_phase12_manifest.json
SHA-256 e453cc9a7581c4e6fc9903c79028c015ec3578ea8f4866aa35c21854b7c92fec
```

Selected windows:

```text
sub17_monitor_015      frames 72..79
sub16_largetable_001   frames 59..66
sub16_plasticbox_004   frames 70..77
sub17_smallbox_015     frames 78..85
```

## Method

Run ContactOpt independently on every frame with the Phase 10 parameters:

```text
w_opt_rot = 0
w_opt_trans = 0
w_obj_rot = 0
rand_re = 0
250 iterations
```

Save input and refined hand vertices, hand joints, and object vertices for
every selected frame. No temporal smoothing, parameter sharing, or learned
temporal model is applied in this phase.

## Metrics

For each case and for both the input aligned MANO hand and refined hand:

```text
speed        = mean ||x[t+1] - x[t]||
acceleration = mean ||x[t+1] - 2*x[t] + x[t-1]||
jerk         = mean ||x[t+1] - 3*x[t] + 3*x[t-1] - x[t-2]||
```

Report refined/input ratios in millimeters per frame. Also retain:

```text
contact-improved frame fraction
mean nearest-distance relative change
wrist-root drift
object-vertex drift
per-frame mean hand-vertex motion from the aligned input
```

## Promotion Gate

A case passes only when both gates pass.

Contact gate:

```text
contact-improved frames >= ceil(0.7 * eligible frames)
mean nearest-distance relative change <= +10%
maximum wrist drift <= 0.01 mm
maximum object drift <= 0.001 mm
```

Provisional temporal gate:

```text
refined/input speed ratio        <= 2.0
refined/input acceleration ratio <= 2.0
refined/input jerk ratio         <= 3.0
```

The temporal ratios are deliberately provisional because this is the first
dense-window measurement. They are frozen before seeing ContactOpt output.

Decision:

```text
4/4 contact+temporal passes: GO for short-window independent integration
3/4 passes: CONDITIONAL, inspect the failing window
0-2/4 passes: NO-GO, add temporal regularization before integration
```

## Boundaries

This phase measures temporal consistency. It does not train a finger world
model, recover `pose_hand`, change the object trajectory, or demonstrate
physics. A temporal-pass result only allows the independent refinement to be
used as a short-window integration baseline.
