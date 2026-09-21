# Task Plan: DWM D0-D2

## Goal

Build a simulator-backed action-conditioned object-response model with
same-initial-state counterfactual data. The line stops after D0-D2 gates;
MaMi integration is a separate later plan.

## Current Phase

D0-B

## Phases

### D0-A: MuJoCo Right-Hand Environment
- [x] Vendor the SMPL-X right-hand MJCF and its license/notice
- [x] Implement the 12 object configurations and splits
- [x] Implement reset, 51D action control, state extraction, contact
  mode, and deterministic stepping
- [x] Add focused tests and server smoke
- **Status:** complete

### D0-B: Counterfactual Dataset
- [x] Implement the 12 fixed action branches
- [x] Run branch excitation smoke
- [ ] Generate the split-disjoint trajectory dataset
- [ ] Apply reproducibility, excitation, coverage, and leakage gates
- **Status:** in_progress

### D1: Action Identifiability
- [ ] Train true/shuffled/zero-action arms
- [ ] Evaluate h1/h2/h4/h8 object response and contact mode
- [ ] Record the action-shuffle gap and gate decision
- **Status:** pending

### D2: Decision Utility
- [ ] Build the target object-displacement task
- [ ] Compare model ranking, geometry heuristic, random, and oracle
- [ ] Record regret, top-1, slip/release safety, and GO/NO-GO
- **Status:** pending

## Required Server Boundary

All experiments and unit tests run on the server. The current no-card
instance is suitable only for dependency setup and smoke. Full D0-B
generation requires the larger/GPU instance selected in the plan.

## Decisions Frozen

| Decision | Value |
|---|---|
| Simulator | MuJoCo 3.2.3 |
| Hand | SMPL-X right hand from HandX/InterMimic asset |
| Object scope | sphere, box, cylinder rigid primitives |
| Action | 8 x 51 position controls |
| Outcome | object delta + contact mode + contact impulse |
| D boundary | Stop after D2; no MaMi integration in this plan |
