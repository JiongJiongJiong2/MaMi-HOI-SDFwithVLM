# DWM P0/P1 Decision-Focused Probe Protocol

Date: 2026-09-22

Status: frozen before implementation

## Goal

Test whether a short interaction probe makes unseen object response
identifiable for ranking candidate hand action chunks. The line stops after
P1. It does not claim MaMi improvement, human tactile transfer, deformable
dynamics, or general physical fidelity.

The prior D1 single-state regression remains a frozen `NO-GO` and is not
reopened.

## D0-C Probe Dataset

The environment, hand, object configurations, reset distribution, state,
and candidate actions remain identical to D0-B.

Each reset produces seven groups:

```text
no probe
fixed probe, budget 1
fixed probe, budget 2
fixed probe, budget 4
random probe, budget 1
random probe, budget 2
random probe, budget 4
```

The fixed probe is the nested prefix of:

```text
push_x+
lift
push_y+
lower
```

The random probe is generated from a reset-specific fixed seed as a
length-four nested sequence of probe primitives. Probe primitives are the
13 existing branch action chunks. A random sequence must not be exactly
equal to the fixed sequence.

Every group stores the initial state, the observed probe trajectory, the
post-probe checkpoint state, and the outcomes of all 13 candidate branches
starting from that same checkpoint. The candidate chunks use the primary
horizon of eight control steps.

## Target and Utility

Each group includes an independent target action generated from a frozen
smooth target-action family distinct from the 13 candidate actions. The
target action is not an input to deployable models.

The target translation is:

```text
target_translation
  = target final object position
  - post-probe object position
```

Candidate utility is:

```text
utility_i
  = -L1(
      candidate_i final object position
      - post-probe object position,
      target_translation
    )
```

Slip and unintended release are separate safety metrics. They are not
folded into the utility used to train or rank candidates.

## Dataset Schema

Each split is a single compressed NPZ file with group-level arrays:

```text
initial_state                 [G, 168] float32
probe_action                  [G, 4, 51] float32
probe_state                   [G, 5, 168] float32
probe_mask                    [G, 4] bool
post_probe_state              [G, 168] float32
candidate_action              [G, 13, 8, 51] float32
candidate_final_object_pose   [G, 13, 9] float32
candidate_contact_mode        [G, 13, 9] int64
candidate_contact_impulse     [G, 13, 8, 3] float32
target_action                 [G, 8, 51] float32
target_translation            [G, 3] float32
utility                       [G, 13] float32
rank                          [G, 13] int64
object_id                     [G] unicode
reset_id                      [G] int64
split                         [G] unicode
probe_mode                    [G] unicode
probe_length                  [G] int64
branch_names                  [13] unicode
```

Expected full counts:

```text
groups:             10,080
candidate rollouts: 131,040
target rollouts:    10,080
train/val/test groups: 5,040 / 2,520 / 2,520
```

The candidate action tensor contains only position targets. It must never
contain object pose, object delta, contact mode, contact force, target, or
utility.

## D0-C Gates

1. small repeated generation is identical for every stored field;
2. train/val/test object configurations and group keys are disjoint;
3. fixed and random probes are nested across budgets;
4. every candidate group contains 13 deterministic outcomes;
5. at least 80% of training groups have a utility range above 2 mm;
6. every contact mode occurs at least 100 times across the full dataset;
7. modifying candidate outcomes does not alter model input tensors;
8. no deployable model input contains mass, friction, object variant,
   target, utility, slip, release, or contact outcome.

Failure stops P0. Target generation is corrected before training rather
than filtering groups by outcome.

## P0 Passive Probe Ranking

The primary model receives:

```text
initial_state
probe_action
probe_state
probe_mask
post_probe_state
candidate_action
```

It predicts a 16D Gaussian response context and one score per candidate.

The primary model uses a pairwise/listwise ranking loss and a centered
object-delta SmoothL1 auxiliary loss with weight 0.1.

Required matched comparisons:

```text
no probe
fixed probe budgets 1, 2, 4
random probe budgets 1, 2, 4
absolute object-delta baseline
quotient pairwise baseline
geometry-only
random
oracle response context
oracle utility
```

The absolute baseline predicts candidate object deltas and derives utility
from those deltas. The quotient baseline predicts signed utility
differences with an antisymmetric pairwise model.

Training settings:

```text
hidden size:       128
response latent:    16
GRU layers:          1
seeds:              11, 23, 37
optimizer:          AdamW
learning rate:      1e-3
weight decay:       1e-4
batch groups:       64
maximum epochs:     100
early stopping:     15
checkpoint metric:  validation regret
```

No test tuning is permitted. Model, budget, baselines, and thresholds are
frozen before the first test evaluation.

## P0 Promotion Gate

Primary budget is two probe steps. Evaluation uses the unseen object
configuration split.

1. top-1 is at least 10 percentage points above geometry-only;
2. top-1 is at least 5 percentage points above the best no-probe model;
3. regret is at least 20% lower than geometry-only;
4. regret is at least 10% lower than the best no-probe model;
5. paired top-1 and regret gains have hierarchical-bootstrap 95% CI lower
   bounds above zero using 10,000 resamples;
6. all three seeds agree in direction for top-1 and regret;
7. slip rate and unintended release worsen by no more than 2 percentage
   points relative to the best passive baseline.

Budgets 1 and 4 are reported for sensitivity. A result that passes only at
budget 4 does not authorize promotion.

Failure freezes D as `NO-GO`.

## P1 Active Probe

P1 starts only after P0 passes.

An ensemble of five probe-outcome models predicts the next state after a
probe primitive. The active selector enumerates the 13 probe primitives,
predicts their response-context effect, and chooses the primitive that
maximizes the predicted reduction in candidate top-1 entropy plus the
minimum predicted pairwise response margin.

Budgets 1, 2, and 4 are compared against fixed, random, and the best passive
probe model at the same budget.

P1 gate:

1. top-1 is at least 5 percentage points above the best passive model at
   the same budget, or regret is at least 10% lower;
2. the paired gain has a 95% bootstrap lower bound above zero;
3. all three seeds agree in direction;
4. slip and unintended release do not worsen by more than 2 percentage
   points.

Active-probe failure removes the active-policy claim but does not invalidate
a successful P0.

## Server Boundary

Full D0-C generation, formal training, and final evaluation run on the
current GPU server. Local execution covers static analysis and tests that
do not need MuJoCo, Torch, or the datasets.
