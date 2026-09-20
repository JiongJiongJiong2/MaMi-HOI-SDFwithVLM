# E5-T Direction Gate Result

Date: 2026-09-20

Status: complete with NO-GO

Protocol:

```text
docs/experiments/contactopt-e5t-direction-gate-protocol-2026-09-20.md
```

Compact result:

```text
docs/experiments/e5t_direction_gate_v1_20260920.json
```

Server result:

```text
/root/autodl-tmp/contact_action_20260914/e5t_direction_gate_v1_20260920.json
```

## Pre-Execution Revision

The first run selected E5 whenever `near_outward_fraction` was missing or
non-finite. That rule exceeded the 20-window train selection budget before
any threshold could be evaluated. It was changed to keep E5-T for missing
features, and the protocol was updated before any train or dev outcome was
produced.

## Frozen Selector

The train-only fitted threshold is:

```text
near_outward_fraction > 0.6661904752452155
```

The selector uses E5 only when its contact gate passed and the direction
risk exceeds this threshold. Otherwise it keeps E5-T.

## Aggregate Results

| Split | Arm | Contact passes | Combined passes | Selected E5 |
|---|---|---:|---:|---:|
| train | E5 | 245/256 | 236/256 | n/a |
| train | E5-T | 236/256 | 231/256 | n/a |
| train | direction gate | 239/256 | 234/256 | 15 |
| dev | E5 | 117/120 | 107/120 | n/a |
| dev | E5-T | 114/120 | 110/120 | n/a |
| dev | direction gate | 117/120 | 112/120 | 9 |

Dev oracle union is `114/120` combined passes.

Paired differences versus always-E5-T:

```text
train combined: +3, clustered 95% CI [0, +7]
dev combined:   +2, clustered 95% CI [-2, +7]
```

Leave-one-object-out threshold fitting inside train selected 15 E5
windows and reached `233/256` combined passes, compared with `231/256`
for always-E5-T. The fitted threshold stayed near `0.666` across folds.

## Gate

```text
train combined gain:                 pass
dev combined gain:                   pass
dev contact not below E5-T:           pass
dev clustered interval lower > 0:    fail
overall:                              NO-GO
```

## Decision

The direction feature has predictive signal, but the current candidate
set does not support a deployable discrete gate. A `+2/120` dev change is
not separable from sequence-level variation. The result must not be
upgraded by changing the threshold, feature, or selection budget after
seeing dev.

The 12 contact regressions come from only 10 sequences, with three dev
regressions from two sequences. The next direction-related experiment is
not justified until either:

1. a larger independent event set is available; or
2. a continuous projection is compared with ordinary contact projection
   and strong constraint baselines under a predeclared power analysis.

If the continuous projection cannot outperform those baselines on a
larger event set, the outward-direction mechanism should remain a
diagnostic finding rather than become the paper's core claim.

The test split remains untouched.

## Implementation Hash

```text
scripts/analyze_e5t_direction_gate.py
392b05503fa97c1d1e58cd5a4a86cdeb2712e057a9c37b9b7da9bc2412a01bda
```
