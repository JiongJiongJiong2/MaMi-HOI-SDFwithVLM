# Task Plan: OakInk C0-R1 Oracle Reranking

## Goal

Test whether a correct future receiver target changes the ranking of
plausible pre-transfer giver grasps relative to a no-future ranking and
a same-object shuffled future target.

C0-R1 is CPU-only retrieval and reranking. It does not train a model,
does not run ContactOpt, and does not claim generated-grasp improvement.

## Current Phase

Phase 1

## Phases

### Phase 1: Freeze Oracle Reranking Protocol
- [x] Define candidate construction and object-frame alignment
- [x] Freeze the no-future, correct-future, and shuffled-future scores
- [x] Freeze paired metrics and promotion gates
- **Status:** complete

### Phase 2: Implement CPU-Only Reranker
- [x] Implement deterministic candidate scoring
- [x] Implement participant/object grouped paired statistics
- [x] Add focused unit tests
- **Status:** complete

### Phase 3: Server Smoke and Full Run
- [x] Upload code and tests to the no-card server
- [x] Run the smoke test and full C0-R1 audit on the server
- [x] Retrieve compact results and verify hashes
- **Status:** complete

### Phase 4: Result, Decision, and Commit
- [x] Record GO, CONDITIONAL, or NO-GO for downstream optimization
- [x] Commit protocol, code, tests, compact result, and hashes
- **Status:** complete

## Boundary

The observed giver grasp is one successful outcome, not a
counterfactual. Same-object observed grasps are candidates for an
oracle ranking test, not independent actions generated from one initial
state. A positive result only authorizes a later bounded optimizer; it
does not establish physical causality or MaMi transfer.
