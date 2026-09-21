# DWM D1 Action-Identifiability Result

Date: 2026-09-22

Status: complete with frozen promotion gate NO-GO

Protocol:

```text
docs/experiments/dwm-d0-d2-protocol-2026-09-22.md
```

## Data

The D0-B counterfactual dataset contains:

```text
trajectories: 18,720
train/val/test trajectories: 9,360 / 4,680 / 4,680
train/val/test reset groups: 720 / 360 / 360
state/action dimensions: 168 / 51
```

The D0-B gate passes:

```text
split-group overlaps:       0
reproducibility difference: 0.0
train excitation rate:      645 / 720 = 0.8958
contact-mode coverage:      > 100 samples for every mode
```

## Identifiability Runs

The first absolute-target runs used unseen object mass/friction in
validation. The state-centered runs removed branch-independent
settling by subtracting the mean outcome of all 13 actions for each
reset.

Seed 11 h8 results:

| Target / split | Arm | Translation L1 | Rotation L1 | Mode macro-F1 |
|---|---|---:|---:|---:|
| absolute / object-held-out | true | 0.0351 | 0.1541 | 0.7695 |
| absolute / object-held-out | shuffled | 0.0396 | 0.1601 | 0.7347 |
| absolute / object-held-out | zero | 0.0373 | 0.1584 | 0.7420 |
| centered / object-held-out | true | 0.0123 | 0.0409 | 0.7539 |
| centered / object-held-out | shuffled | 0.0098 | 0.0372 | 0.7214 |
| centered / object-held-out | zero | 0.0094 | 0.0383 | 0.7277 |
| centered / reset-held-out | true | 0.0094 | 0.0390 | 0.7907 |
| centered / reset-held-out | shuffled | 0.0072 | 0.0381 | 0.7514 |
| centered / reset-held-out | zero | 0.0071 | 0.0374 | 0.7567 |

## Gate

```text
true h8 translation error <= 0.7 * shuffled/zero: FAIL
true h8 rotation error    <= 0.7 * shuffled/zero: FAIL
true mode macro-F1 >= shuffled + 0.10:            FAIL
three-seed direction consistency:                 not run after seed 11 failure
overall:                                          NO-GO
```

The contact-mode result is positive but insufficient: true action
improves mode F1 by roughly 0.03 over the no-action arms, below the
frozen 0.10 requirement. Object response regression is worse than the
no-action baselines in both object-held-out and reset-held-out tests.

## Interpretation

The model learns a useful contact-mode effect, but it does not learn an
object-response effect that survives paired action controls. This is a
real failure of the current D formulation, not a numerical accident:

- action branches create distinct object outcomes;
- data are deterministic and split-disjoint;
- loss weights were corrected to make object response primary;
- common settling was removed with state-centered targets;
- the same failure occurs on held-out resets of seen objects.

The current initial-state-to-action-chunk formulation therefore does not
identify object dynamics well enough to enter D2. Possible future
directions require a different D design, such as a probing history,
known physical parameters, or a more structured contact/force state.
Those are new research hypotheses, not repairs of the frozen D1 test.

## Decision

```text
D1 action identifiability: NO-GO
D2 decision utility:       not run
MaMi integration:          not authorized
```

## Server Outputs

```text
/root/autodl-tmp/dwm_d0b_full_v1_20260922
/root/autodl-tmp/dwm_d1d_indist_seed11_*
/root/autodl-tmp/dwm_d1d_ood_seed11_*
```

## Implementation Hashes

```text
manip/world_model/dwm/model.py
da35293df72f9fb9c7e416df696fdf2e6bb5a99a1e1aaddf31a8d5890f3e9033

scripts/train_dwm_transition.py
d38eff243d72224a00541e88f12f94cbc5180fbf1392950bcd1a232550b383be

scripts/audit_dwm_d0b_dataset.py
e73ae62db4eca5dad826d767ef956d74eba78189ba0c114536be57b0fd35a25d

tests/test_dwm_model.py
d891a491ced61dc18bd73b21e0c29f31d82791768c98cc291752f725b9f2a1d8
```
