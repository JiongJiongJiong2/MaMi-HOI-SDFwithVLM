# Task Plan: DWM P0/P1 Decision-Focused Probe

## Goal

Implement the frozen D-line P0/P1 plan: generate probe-conditioned
same-state counterfactual data, train and evaluate a passive probe ranking
model under strict decision-utility gates, and only then authorize the
active probe stage.

## Current Phase

Protocol and environment API

## Phases

### 1. Protocol and Environment API
- [x] Freeze the P0/P1 protocol document
- [x] Add deterministic MuJoCo checkpoint/restore support
- [x] Add checkpoint regression tests
- **Status:** complete

### 2. D0-C Probe Dataset
- [x] Implement probe and target action generation
- [x] Implement group-level dataset writer and audit
- [x] Pass a server smoke dataset
- [ ] Generate and audit full D0-C
- [ ] Commit the validated data implementation and result
- **Status:** in_progress

### 3. P0 Passive Probe Ranking
- [x] Implement the ranking model and matched baselines
- [x] Implement metrics, bootstrap, safety checks, and frozen gate
- [x] Pass a two-epoch server smoke
- [x] Run three formal seeds and evaluate test once
- [x] Record P0 GO/NO-GO and commit
- **Status:** complete with NO-GO

### 4. P1 Active Probe
- [ ] Implement the probe-outcome ensemble and selector
- [ ] Pass mock-selector and simulator integration tests
- [ ] Run the active-vs-passive promotion experiment
- [ ] Record P1 GO/NO-GO and commit
- **Status:** blocked by P0 NO-GO; not authorized

## Frozen Decisions

| Decision | Value |
|---|---|
| Candidate branches | Existing 13 fixed D0-B actions |
| Probe budgets | 0, 1, 2, 4 control steps |
| Probe conditions | no probe, nested fixed, nested random |
| Fixed probe | `push_x+`, `lift`, `push_y+`, `lower` |
| Decision target | Target object translation from an independent target action |
| Primary utility | Negative L1 translation error |
| Safety metrics | Slip rate and unintended release |
| Main promotion budget | 2 steps |
| Seeds | 11, 23, 37 |
| MaMi integration | Out of scope |

## Required Server Boundary

All MuJoCo generation, formal training, and final tests run on the current
GPU server. Local execution is limited to static checks and tests that do
not need MuJoCo, Torch, or datasets.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---:|---|
| None yet | 0 | None |
