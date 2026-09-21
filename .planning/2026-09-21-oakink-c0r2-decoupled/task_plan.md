# Task Plan: OakInk C0-R2 Decoupled Reranking

## Goal

Test future-conditioned giver-grasp reranking with a protocol that fixes
the two main weaknesses found by the C0-R1 audit:

1. use all reference-frame-valid events instead of only complete windows;
2. decouple conditioning geometry from evaluation geometry;
3. test paired utility directly instead of requiring selection changes.

C0-R2 remains CPU-only and does not train a generative model.

## Current Phase

Phase 1

## Phases

### Phase 1: Freeze Decoupled Protocol
- [x] Use receiver contact region for conditioning only
- [x] Use receiver joints for independent evaluation only
- [x] Freeze paired utility gates and sensitivity weights
- **Status:** complete

### Phase 2: Implement Decoupled Reranker
- [x] Implement reference-valid candidate construction
- [x] Implement region-conditioned selection
- [x] Implement joint-based evaluation and paired statistics
- [x] Add focused unit tests
- **Status:** complete

### Phase 3: Server Run and Verification
- [x] Upload code and tests to the no-card server
- [x] Run tests and the full C0-R2 audit on the server
- [x] Verify output hashes and data counts
- **Status:** complete

### Phase 4: Decision and Commit
- [x] Record GO, CONDITIONAL, or NO-GO
- [ ] Commit protocol, code, tests, report, and hashes
- **Status:** in_progress

## Boundary

The candidate set remains observed successful grasps. Receiver joints
are independent of the conditioning features but are still derived from
the same successful handover event. A pass authorizes only a bounded
downstream optimizer, not physical causality or direct MaMi transfer.
