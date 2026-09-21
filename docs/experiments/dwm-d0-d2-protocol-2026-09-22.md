# DWM D0-D2 Protocol

Date: 2026-09-22

Status: frozen before implementation

## Goal

Build a simulator-backed action-conditioned object-response model using
same-initial-state counterfactual action branches.

The plan stops after D0-D2. It does not claim MaMi improvement, human
tactile transfer, deformable dynamics, or physical fidelity outside the
configured MuJoCo environment.

## D0-A Environment

```text
simulator:       MuJoCo 3.2.3
hand:            SMPL-X right hand, HandX/InterMimic MJCF
physics dt:      0.002 s
control period:  0.030 s
physics steps:   15 per control step
```

The right-hand asset is used under its upstream non-commercial research
license. The repository must retain the license and third-party notice.

### State

The canonical state is 168D:

```text
0:9      object translation + first-two-rows rotation 6D
9:15     object linear/angular velocity
15:24    wrist translation + first-two-rows rotation 6D
24:30    wrist linear/angular velocity
30:75    45 finger joint positions
75:120   45 finger joint velocities
120:168  16 hand-body contact-force triples
```

Units are SI. Rotations are normalized 6D.

### Action

The action chunk is `[H, 51]`. Each row is a requested position target
for the 51 right-hand actuators:

```text
0:3      wrist translation
3:6      wrist rotation
6:51     45 finger joints
```

The action never contains object pose, object delta, contact state, or
contact force.

### PD Control

```text
wrist translation: kp 1000, kd 100
wrist rotation:    kp 50,   kd 5
finger joints:     kp 5,    kd 0.5
```

Torque is clipped to finite actuator control ranges.

### Objects

Three shapes and four mass/friction variants produce 12 configurations:

```text
sphere:   radius 0.040 m
box:      half extents 0.040, 0.030, 0.025 m
cylinder: radius 0.035 m, half length 0.035 m

variant 0: mass 0.10 kg, friction 0.45
variant 1: mass 0.25 kg, friction 0.65
variant 2: mass 0.35 kg, friction 0.80
variant 3: mass 0.55 kg, friction 1.00
```

Split by object configuration:

```text
train: variants 0, 1 for all shapes = 6
val:   variant 2 for all shapes  = 3
test:  variant 3 for all shapes  = 3
```

### Reset States

Each object uses 120 resets. The reset seed is fixed to `20260922`.

```text
object horizontal offset: uniform +/- 0.010 m
object yaw:               uniform +/- pi
contact-state mix:        80% initial contact, 20% near contact
```

The wrist translation is calibrated per shape from the actual capsule
geometry. Initial-contact resets align the wrist over the object and use:

```text
sphere:   0.050 m above the object center
box:      0.035 m above the object center
cylinder: 0.040 m above the object center
```

Near-contact resets add 0.015 m and use small lateral jitter. Finger
joints start at zero. Wrist yaw follows object yaw, with a small jitter
only in the near-contact group.

No reset is filtered by action outcome.

## D0-B Branches

Each reset receives 12 actions:

```text
hold
close
open
push_x+
push_x-
push_y+
push_y-
lift
lower
random_0
random_1
random_2
random_3
```

The primary horizon is 8 control steps. Push/lift/lower offset wrist
targets by 0.010 m. Close/open use automatically signed finger flexion
axes. Random actions use smooth, seeded residuals.

Expected trajectories:

```text
12 objects x 120 resets x 12 actions = 17,280
```

## D0-B Gates

1. repeated reset/action runs have object translation difference below
   `1e-5 m` and rotation difference below `1e-4 rad`;
2. at least 80% of training resets produce four or more distinct
   outcomes across the 12 actions;
3. an outcome differs when translation exceeds 2 mm, rotation exceeds
   0.02 rad, or contact mode differs;
4. every mode has at least 100 transitions across the dataset;
5. train/val/test object configs and reset ids are disjoint;
6. action tensors contain no object/contact outcome fields.

Failure stops D0-B and does not authorize D1.

## D1 Identifiability

Train the same model on true, shuffled, and zero action inputs.

```text
state encoder:       MLP
action encoder:      per-step GRU
output:              object delta, contact mode, contact impulse
```

Gate:

1. h8 true-action translation and rotation error are at least 30% lower
   than shuffled and zero action;
2. true-action contact-mode macro-F1 is at least 0.10 higher than
   shuffled;
3. all three seeds agree in direction.

Failure stops D1 and does not authorize D2.

## D2 Decision Utility

Each reset receives a pre-action object-displacement target. Utility is
computed only from simulator outcomes. Compare model ranking,
geometry-only heuristic, random, and oracle over the same 12 branches.

Gate:

1. model top-1 is at least 10 percentage points above geometry-only;
2. regret is at least 20% lower;
3. object-config grouped 95% bootstrap interval lower bound exceeds
   zero for the paired gain;
4. slip and unintended release do not worsen.

Failure stops D and does not authorize MaMi integration.
