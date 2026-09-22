# Findings

## Initial Context

- Branch: codex/e5t-direction-decomposition
- Worktree was clean at recovery time.
- The E5-T direction decomposition completed with GO: regressions had a larger
  outward near-contact normal component than stable windows.
- The decomposition isolated association, not causation. The next experiment
  must match temporal, pose, and contact budgets across all arms.
- Planning files did not exist for this experiment before this session.

## Constraints

- Only train/dev, 49 contiguous chunks, and 376 windows may be read.
- The 28 test windows must remain unread.
- E5 and E5-T thresholds and checkpoints are frozen.
- The constraint strength must be chosen on train and applied unchanged to dev.

## Frozen Implementation

- New solver:
  scripts/optimize_contactopt_direction_projection.py
- Focused tests:
  tests/test_contactopt_direction_projection.py
- The default seed is restored to the frozen E5-T value 20260919.
- Finger coefficients 3:18 are optimized; global pose 0:3 and hand_mTc are
  frozen.
- Normal tolerance is 0.5 mm, contact tolerance is 1 mm, pose L2 cap is 1.8,
  projection damping is 1e-4, and projection step cap is 0.1.

## Smoke Findings

- Train chunk: sub16_largebox_008_c000.
- Unconstrained continuation and frozen E5-T have maximum coefficient error
  0.0 on the smoke chunk.
- Contact projection reduced summed normal excess from 16.720 mm to 3.483 mm
  on the selected frames.
- Direction-constrained, ordinary smoothing, and strong contact all passed
  8/8 smoke windows. This chunk is not discriminative for the mechanism.

## Full Result

- All 49 chunks and 376 train/dev windows completed; test was unread.
- Direction-constrained combined/contact/temporal: 351/362/365.
- E5-T combined/contact/temporal: 341/350/367.
- Direction gained 12 combined windows and lost two temporal-only windows.
- Contact regression versus E5 is zero, but direction mean acceleration and
  jerk remain above E5-T.
- No sequence-clustered control difference has a lower bound above zero.
- Decision: NO-GO for mechanism promotion. The contact regression was removed,
  but the temporal and motion budgets did not match E5-T and the improvement
  over controls was not statistically separated.
