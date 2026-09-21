# DWM D0-A/D0-B Smoke Result

Date: 2026-09-22

Status: D0-A complete; D0-B implementation and small smoke complete;
full 17,280-trajectory generation pending a larger server

Protocol:

```text
docs/experiments/dwm-d0-d2-protocol-2026-09-22.md
```

## D0-A Environment

Implemented:

```text
MuJoCo 3.2.3
SMPL-X right-hand MJCF
12 rigid object configurations
168D canonical state
8 x 51 position-control action chunk
15 physics substeps per 30 Hz control step
```

The upstream asset required MuJoCo-specific adaptation:

1. explicit inertial records were added to the root and wrist-slide body;
2. PD torque is recomputed at every physics substep;
3. contact forces are read with `mj_contactForce` and aggregated over
   the 16 hand bodies;
4. contact height is calibrated per shape;
5. the reset mix is 80% initial contact and 20% near contact, with wrist
   yaw aligned to object yaw.

## D0-A Tests

The server ran eight focused tests:

```text
DWMConfigTest: 2 passed
DWMActionBranchTest: 3 passed
DWMEnvironmentTest: 3 passed
total: 8 passed
```

The 12-object hold smoke was finite for all checked resets. The largest
stable hold displacement was below 3 mm.

## D0-B Branch Smoke

The fixed branch set is:

```text
hold, close, open, push_x+, push_x-, push_y+, push_y-,
lift, lower, random_0, random_1, random_2, random_3
```

For 12 object configurations and 5 resets each:

```text
resets with >= 4 distinct outcome pairs: 52 / 60
excitation rate:                         0.8667
required by D0-B gate:                   0.8000
```

The action branch smoke passes. Outcomes are considered distinct when
translation exceeds 2 mm, rotation exceeds 0.02 rad, or contact mode
differs.

## Dataset Schema Smoke

A reduced split-disjoint dataset was generated on the server:

```text
train: 1 object, 2 resets, 13 branches = 26 trajectories
val:   1 object, 2 resets, 13 branches = 26 trajectories
test:  1 object, 2 resets, 13 branches = 26 trajectories
```

Stored fields:

```text
states          [N, 9, 168]
actions         [N, 8, 51]
object_pose     [N, 9, 9]
object_twist    [N, 9, 6]
contact_forces  [N, 9, 16, 3]
contact_mode    [N, 9]
reset_id, object_id, branch_name, split
```

The reduced smoke validates the writer, split naming, metadata, and
hash recording. It is not the D0-B promotion dataset.

## Server Outputs

```text
/root/autodl-tmp/dwm_d0_20260922
/root/autodl-tmp/dwm_d0b_smoke_20260922
```

Full D0-B generation requires the larger/GPU instance selected in the
plan. The current no-card 0.5-core instance is sufficient for smoke but
not for 17,280 trajectories plus D1 training.

## Implementation Hashes

```text
manip/world_model/dwm/config.py
21c762b547c274e428b2267a9b880285d33ef4adc76983194a68aa8178c5ad80

manip/world_model/dwm/schema.py
10c3b0637b557d32b94b4962cd2e2f9a7eddb034ac5b6a97b08c01d8e8078889

manip/world_model/dwm/xml.py
82b258bcf3b24b87ed72dd7ec3190dce3f3e834d1d3e8e00b050f27749a2037a

manip/world_model/dwm/env.py
66a8407e3a423fb7086cf614ba8741bec1a8cb4173227314abe935a7d5f6f14b

manip/world_model/dwm/branches.py
ae8c45cf67e9a8476583412bbe8b6e638933e5b178cb19883dc267a2f79aafa9

scripts/generate_dwm_counterfactual_dataset.py
44a70022e01bc2492a8ba664d7e9e97467fe2dbfb7b762150560520ac45e2718

reduced metadata.json
68895c0abc401dea52b1a4885fac612adc49daecd5ec6edb8d97ce6426d9b203
```
