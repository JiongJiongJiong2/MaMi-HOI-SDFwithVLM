# Findings and Decisions

## Existing Evidence

- Old E2 was not action-identifiable in the 34D/15D representation.
- The existing contact-action transition receives object deltas as action
  inputs, so it is not an independent object-response world model.
- Natural observed trajectories do not provide same-initial-state
  counterfactual outcomes.
- HOI-Dyn learned one-step object response, but autoregressive rollout
  was only marginally better than a zero-motion object baseline.

## Environment Audit

The server's `mami_hoi` environment has Torch and PyTorch3D but no
MuJoCo, IsaacGym, PyBullet, SAPIEN, or Gym. The HandX/InterMimic asset
tree contains a right-hand MJCF with 51 motor actuators.

## D0-A Corrections

The D plan's earlier approximate state size was 162D. The concrete
layout is 168D:

```text
object pose + twist               15
wrist pose + twist                15
finger positions + velocities    90
16 body contact-force triples    48
total                            168
```

This concrete layout is authoritative for implementation.

## D0-A Calibration

The initial hand asset already extends well below the wrist. A first
near-contact height of `0.075 m` caused immediate self-penetration and a
solver launch in one sphere reset. The frozen D0-A values are now:

```text
initial contact offsets above object center:
  sphere 0.050 m, box 0.035 m, cylinder 0.040 m
near contact: contact offset + 0.015 +/- 0.004 m
contact mix:  80% initial contact, 20% near contact
wrist yaw:    follow object yaw
```

This is an environment calibration, not an outcome-based filter.

## D0-A/D0-B Smoke Result

```text
server tests:                       8 passed
12-object hold smoke:               finite and stable
branch excitation on 60 resets:     52 / 60 = 86.7%
reduced dataset trajectories:       78
state/action schema:                168D / 51D
```

D0-A is complete. D0-B branch generation is implemented and passes the
small excitation smoke. Full data generation is waiting for the larger
server instance.
