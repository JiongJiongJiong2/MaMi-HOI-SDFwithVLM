# E5-T Direction-Constrained Projection

## Goal

Execute the direction-constrained temporal correction experiment promised by
the E5-T direction decomposition GO result. Preserve E5-T temporal gains while
testing whether outward-normal constraints plus contact reprojection recover
the E5 contact pass count. Use only the frozen 376 train/dev windows and do not
read test.

## Frozen Inputs

- Batch summary:
  /root/autodl-tmp/contact_action_20260914/sequence_contactopt_v1_20260919/cases/batch_summary.json
- Geometry:
  /root/autodl-tmp/contact_action_20260914/sequence_contactopt_v1_20260919/geometry
- E5:
  /root/autodl-tmp/contact_action_20260914/e5_solver_v1_20260919
- E5-T:
  /root/autodl-tmp/contact_action_20260914/e5t_solver_v1_20260919

## Phases

### Phase 1: Recover interfaces and freeze the protocol

Status: complete

- Inspect the existing E5/E5-T continuation solver, metrics, gates, and tests.
- Confirm the remote frozen inputs and available runtime without writing to them.
- Freeze arm definitions, optimization budgets, tolerances, and output schema.

### Phase 2: Implement the solver and focused tests

Status: complete

- Add direction-aware objective terms and constrained finger-pose optimization.
- Add damped normal contact reprojection with temporal-gate and pose-budget
  rollback.
- Add unit tests for direction signs, projection decrease, bounded pose change,
  rollback, and regression compatibility.

### Phase 3: Local verification and checkpoint

Status: in_progress

- Run focused tests plus relevant existing solver tests.
- Verify no unintended changes to E5/E5-T artifacts.
- Commit the validated implementation and tests as one checkpoint.

### Phase 4: Server smoke

Status: complete

- Sync the committed implementation to the remote workspace.
- Run one train chunk for every fixed arm.
- Verify reconstruction, wrist drift, object drift, window identity, gate
  evaluation, and resumability.

### Phase 5: Full train/dev experiment

Status: complete

- Run all fixed arms on all 49 train/dev chunks and 376 windows.
- Keep test unread.
- Preserve per-chunk optimized arrays, result JSON, and frame JSON.

### Phase 6: Frozen analysis and decision

Status: complete

- Compute paired pass counts, temporal/contact metrics, and sequence-clustered
  bootstrap intervals against all fixed controls.
- Apply the six acceptance conditions without tuning on dev.
- Record GO for the direction mechanism, engineering-only result, or NO-GO for
  the next lifetime/switching mechanism, as specified by the protocol.

### Phase 7: Result checkpoint

Status: complete

- Write the concise protocol/result documents and machine-readable summary.
- Commit code, tests, protocol, result, and focused metadata.
- Keep bulk remote artifacts out of git.

## Acceptance Gate

1. Direction arm combined pass count exceeds always-E5-T consistently on train
   and dev.
2. Contact pass count is not below E5-T and loses no more than 12 windows
   relative to E5; recovering E5 contact pass is preferred.
3. Temporal pass count is not below E5-T.
4. Versus ordinary smoothing, ordinary projection, and strong contact, at
   least one paired combined-pass 95% sequence-clustered bootstrap lower bound
   is strictly positive.
5. Mean acceleration and jerk are not above E5-T, and mean contact distance is
   not more than 10% above E5.
6. Test is unread and no constraint strength is changed after seeing dev.

## Fixed Arms

- E5
- E5-T
- ordinary binomial5 smoothing from E5
- nearest-surface contact projection after ordinary smoothing
- strong contact objective with matched temporal budget
- unconstrained temporal continuation
- direction-constrained smoothing plus contact reprojection
