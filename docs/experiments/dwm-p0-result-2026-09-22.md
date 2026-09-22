# DWM P0 Passive Probe Result

Date: 2026-09-22

Status: complete with frozen P0 promotion gate `NO-GO`

Protocol:

```text
docs/experiments/dwm-p0-p1-protocol-2026-09-22.md
```

## Data Gate

The revised D0-C dataset contains 10,080 groups:

```text
train / val / test groups: 5,040 / 2,520 / 2,520
candidate rollouts:        131,040
```

The frozen audit passes:

```text
split overlaps:                       0
reproducibility maximum difference:   0.0
primary utility-range rate:           2926 / 3600 = 0.8128
probe nesting:                        pass
rank consistency:                     pass
target/candidate separation:          pass
schema and mode coverage:             pass
```

The primary data gate uses no-probe, one-step, and two-step conditions.
Four-step groups are retained and reported as a sensitivity condition but
do not participate in promotion.

## P0 Runs

The formal suite trained three seeds (`11`, `23`, `37`) for:

```text
listwise probe ranking
listwise no-probe
absolute object-delta
quotient pairwise
oracle response context
```

Primary evaluation uses fixed two-step probes on the unseen object
configuration split.

## Frozen Gate

| Comparison | Main | Baseline | Delta |
|---|---:|---:|---:|
| top-1 vs geometry | 0.0574 | 0.2306 | -17.31 pp |
| top-1 vs no-probe | 0.0574 | 0.0657 | -0.83 pp |
| regret vs geometry | 0.002322 | 0.002975 | 21.97% lower |
| regret vs no-probe | 0.002322 | 0.004381 | 47.01% lower |

The regret gate passes, and the corrected safety check passes. The
top-1, seed-direction, and paired top-1 bootstrap gates fail:

```text
top-1 geometry gain >= 10 pp:       FAIL
top-1 no-probe gain >= 5 pp:        FAIL
regret geometry reduction >= 20%:   PASS
regret no-probe reduction >= 10%:   PASS
top-1 bootstrap CI lower > 0:       FAIL
regret bootstrap CI lower > 0:      FAIL for geometry
all seed directions positive:       FAIL
safety noninferiority:              PASS
overall:                            NO-GO
```

## Interpretation

The decision-focused model learned to reduce average regret relative to
both geometry-only and no-probe baselines. It did not learn reliable
candidate identification: top-1 is below geometry-only and the seed
directions are inconsistent.

This is a real failure of the revised P0 formulation, not a pre-training
data stop. The probe-conditioned score is not yet aligned with the
correct candidate ranking.

The current result does not authorize P1 active probing or MaMi
integration.

## Decision

```text
D0-C data gate:             GO
P0 passive ranking gate:    NO-GO
P1 active probe:            not run
MaMi integration:           not authorized
```

## Artifacts

```text
docs/experiments/dwm-d0c-audit-v3-20260922.json
docs/experiments/dwm-p0-gate-v2-20260922.json
```

Server result root:

```text
/root/autodl-tmp/dwm_p0_results_v2_20260922
```
