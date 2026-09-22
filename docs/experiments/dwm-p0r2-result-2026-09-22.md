# DWM P0-R2 Geometry-Feature Residual Result

Date: 2026-09-22

Status: complete with frozen promotion gate `NO-GO`

Protocol:

```text
docs/experiments/dwm-p0r-residual-protocol-2026-09-22.md
```

## Change from P0-R

P0-R2 adds explicit geometry features to the residual model:

```text
candidate wrist displacement
target error vector
geometry logit
```

The candidate score remains:

```text
geometry_logit + learned residual score
```

The residual score head is zero-initialized, so training starts exactly at
the geometry baseline.

## Formal Result

The formal three-seed run produced:

| Comparison | Main | Baseline | Delta |
|---|---:|---:|---:|
| top-1 vs geometry | 0.2769 | 0.2306 | +4.63 pp |
| top-1 vs no-probe | 0.2769 | 0.0741 | +20.28 pp |
| regret vs geometry | 0.002202 | 0.002975 | 25.99% lower |
| regret vs no-probe | 0.002202 | 0.004788 | 54.01% lower |

All three seeds are positive for top-1 and regret. Safety noninferiority
passes.

## Frozen Gate

```text
top-1 geometry gain >= 10 pp:       FAIL (4.63 pp)
top-1 no-probe gain >= 5 pp:        PASS (20.28 pp)
regret geometry reduction >= 20%:   PASS (25.99%)
regret no-probe reduction >= 10%:   PASS (54.01%)
top-1 bootstrap CI lower > 0:       FAIL
regret bootstrap CI lower > 0:      FAIL for geometry by a small margin
all seed directions positive:       PASS
safety noninferiority:              PASS
overall:                            NO-GO
```

## Interpretation

The geometry-residual formulation is real but not yet sufficient. It
improves average utility more reliably than exact winner identification.
The top-1 gain over geometry is positive in every seed, but its magnitude
is only `2.2` to `6.4` percentage points, below the frozen `10` point
requirement.

The remaining gap is now narrow enough to analyze as a ranking-margin
problem rather than a probe-identification problem. Oracle physical context
still does not provide sufficient top-1 improvement, and the model already
contains the geometry direction prior.

## Decision

```text
D0-C data gate:              GO
P0-R2 residual gate:         NO-GO
P1 active probe:             not authorized
MaMi integration:            not authorized
```

The D line remains stopped at P0-R2 under the frozen gate.
