# Analytic Contact Baseline

Date: 2026-09-16

## Purpose

Freeze the analytic geometry branch as the reliable engineering baseline
before starting another world-model research branch.

## Frozen Protocol

```text
scorer_mode: geom_only
rollout_mode: analytic
score: frame_contact_fraction - 20 * mean_penetration
checkpoint: large_event_20260914/model_v8_gpu_fast/best.pt
candidate_seed: 1
horizon: 8
history: 4
```

Candidate cohorts:

```text
10_30: mami_l1_heldout_20_20260915
31_50: mami_l1_heldout_31_50_20260915
51_70: mami_l1_heldout_51_70_20260915
```

## Reproduction

```bash
bash scripts/run_analytic_contact_baseline.sh
python scripts/summarize_analytic_contact_baseline.py \
  --root /root/autodl-tmp/contact_action_20260914/analytic_contact_baseline_v1_20260916 \
  --output /root/autodl-tmp/contact_action_20260914/analytic_contact_baseline_v1_20260916/summary.json
```

## Reference Result

The prior 43-sequence, 78-event result remains the reference until the
canonical rerun is complete:

```text
base MaMi        F1 0.4131
random action    F1 0.3696
analytic selected F1 0.4909
oracle           F1 0.5451

selected - base   +7.78 pp
selected - random +12.13 pp

base penetration       9.787 mm
selected penetration   7.363 mm
selected - base        -2.424 mm
```

The rerun must reproduce the same direction and comparable values before
the analytic branch is promoted as the frozen mainline.

## Promotion Gate

- The canonical rerun must reproduce selected greater than random and
  oracle greater than selected.
- The selected-minus-base direction must be non-negative on the combined
  cohort and the independent 51-70 cohort.
- No threshold or weight may be changed after seeing the rerun results.
- The learned-residual and C2 branches remain research-only until they beat
  this frozen baseline on the same candidate set and metrics.

