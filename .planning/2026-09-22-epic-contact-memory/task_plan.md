# EPIC Contact Lifecycle Memory

## Goal

Run B as an independent mechanism experiment on EPIC-Contact strict 1 mm
participant-disjoint data. Learn per-anchor HOLD/UPDATE/CLOSE/UNKNOWN actions,
apply the resulting material-point memory to deterministic perturbed
trajectories, and determine whether lifecycle-aware memory improves downstream
correction without release-time stickiness.

## Frozen Data

- Raw EPIC train and test pickles:
  E:/HOI/TMP/hopformer_repro_pkls/training_pkls/
- Strict lifecycle outputs:
  docs/experiments/epic_contact_strict_*.jsonl.gz and split JSON.
- Primary threshold: 1 mm.
- Sensitivity: 0.5, 1.0, 1.5, 2.0, 3.0 mm.
- Test is a reused locked confirmation, not a pristine unseen test.

## Phases

### Phase 1: Correspondence manifest

Status: complete

- Stream the raw pickle without loading it wholesale.
- Extract per-frame Top-4 hand/object contact pairs, MANO parameters, object
  topology identity, and strict contact state.
- Enforce participant/video disjointness and no cross-clip or frame-gap memory.

### Phase 2: Memory actions and GRU

Status: complete

- Build train-calibrated HOLD/UPDATE/CLOSE/UNKNOWN reference actions.
- Train the frozen causal GRU and rule/oracle baseline policies.

### Phase 3: Controlled correction evaluator

Status: complete

- Generate deterministic 2/5/10 mm low-pass perturbations.
- Correct with matched optimization budgets under every fixed memory arm.
- Evaluate end-to-end error, hold drift, switch delay, release attraction,
  contact, motion, and participant/object-clustered intervals.

### Phase 4: Local verification

Status: complete

- Run focused unit tests and one-clip smoke tests.
- Verify deterministic perturbations, no future leakage, and exact budgets.

### Phase 5: Server experiment

Status: in_progress

- Build the full manifest, train, evaluate participant test, and run the four
  supported object-held-out folds.
- Record all hashes and resume metadata.

### Phase 6: Frozen gate and result

Status: pending

- Apply oracle-headroom, learned-promotion, event, motion, and generalization
  gates without changing thresholds on test.
- Produce result JSON/report and a focused checkpoint commit.

## Fixed Gate

- Oracle memory must improve end-to-end correction error and post-release
  attraction on dev, otherwise stop before training a larger model.
- Learned memory must beat hysteresis and sticky on test with
  participant-clustered 95% CI lower bounds above zero.
- Hold drift and switch delay must not regress by more than 5%.
- Release AUPRC/F1 must not fall below the frozen logistic baseline.
- At least one HOLD/UPDATE/CLOSE macro-F1 must improve.
- Acceleration/jerk may not exceed hysteresis by more than 5%.
- Object-held-out direction must agree in at least 3 of 4 supported classes:
  bowl, plate, pan, bottle.
