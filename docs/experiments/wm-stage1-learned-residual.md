# WM Stage 1: Bounded Learned Residual

Date: 2026-09-16

## Decision

**NO-GO for the learned-residual branch in its current form.**

The analytic geometry branch remains the active baseline. Generated-state
training, receding-horizon correction, and MaMi guidance are not entered from
this result.

## Protocol

- Three seeds: 21, 22, 23.
- 30 epochs per seed, batch size 512.
- Geometry mode: local object-SDF patches.
- Residual scale: 0.05 with `tanh` bound.
- Residual mask: palm position, palm velocity, clearance, and normal.
- Scorer cohorts: 10 validation sequences and 19 hand events per seed.

## Training Result

| Metric | Analytic rollout | Learned rollout |
|---|---:|---:|
| H8 contact F1 | 0.7195 | 0.7149 |
| H8 palm L1 | 0.00922 | 0.08876 |
| H1 contact F1 | 0.9292 | 0.9292 |
| H1 palm L1 | 0.00133 | 0.01205 |

The learned residual does not improve contact F1 and increases H8 palm error
by roughly 9.6 times. The residual saturates near its bound: mean absolute
residual is about 0.11 and maximum absolute residual is about 1.0.

## Scorer Result

`learned_state - geom_only`:

```text
mean dense contact F1 delta: -10.17 pp
95% CI: [-18.09, -2.44] pp
mean penetration delta: +0.176 mm
95% CI: [-0.260, +0.733] mm
win/tie/loss: 9 / 29 / 19
```

`hybrid_event - geom_only`:

```text
mean dense contact F1 delta: -10.11 pp
95% CI: [-18.25, -2.47] pp
mean penetration delta: +0.187 mm
95% CI: [-0.266, +0.767] mm
win/tie/loss: 9 / 29 / 19
```

The event head does not change the negative conclusion.

## Interpretation

The failure is consistent with free-running error accumulation. The model is
optimized on mixed teacher forcing, but its residual is applied directly to
the recurrent state at every rollout step. The residual becomes large enough
to fit training windows while degrading generated-state geometry.

This result does not show that learned contact dynamics are impossible. It
shows that this bounded residual parameterization, training distribution, and
scoring usage are not sufficient for the next stage.

## Required Stop

- Do not proceed to generated-state fine-tuning from this checkpoint.
- Do not proceed to receding-horizon correction from this checkpoint.
- Do not connect this residual branch to MaMi guidance or training loss.
- Keep the analytic geometry scorer and local geometry encoder as the active
  baseline.
