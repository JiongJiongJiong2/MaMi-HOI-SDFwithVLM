# Progress Log

## Session: 2026-09-21

### Phase 1: Freeze Decoupled Protocol
- **Status:** complete
- Actions taken:
  - Created C0-R2 after the independent NO-GO audit.
  - Selected reference-valid event eligibility.
  - Decoupled receiver-region conditioning from joint-based evaluation.

### Phase 2: Implement Decoupled Reranker
- **Status:** complete
- Actions taken:
  - Implemented reference-valid candidate construction.
  - Implemented region-only conditioning.
  - Implemented joint-only evaluation with paired utility gates.
  - Corrected one unit-test expectation for a receiver-joint collision.

### Phase 3: Server Run and Verification
- **Status:** complete
- Actions taken:
  - Uploaded code and tests to the no-card server.
  - Ran four C0-R2 unit tests successfully.
  - Ran the complete C0-R2 audit.
- Result:
  - 82 eligible events and pairwise candidates.
  - Independent oracle top-1 rate `0.5732`.
  - Correct and shuffled primary top-1 rates both `0.5000`.
  - Primary paired evaluation changes are exactly zero.
  - Frozen gate is NO-GO.

### Phase 4: Decision and Commit
- **Status:** complete
- Actions taken:
  - Created the result report and compact summary.
  - Committed the frozen C0-R2 result as
    `5464402 test(data): reject decoupled OakInk future conditioning`.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| C0-R2 unit test expected one collision when the candidate point lay on the donor joint | 1 | Corrected the synthetic expectation to `2/3`; implementation was unchanged |

## 5-Question Reboot Check

| Question | Answer |
|---|---|
| Where am I? | C0-R2 result commit |
| Where am I going? | implement, run on server, decide |
| What is the goal? | salvage C with an independent future-conditioning test |
| What have I learned? | C0-R1 failed for protocol reasons as well as data limits |
| What have I done? | completed C0-R2, which is also NO-GO |
