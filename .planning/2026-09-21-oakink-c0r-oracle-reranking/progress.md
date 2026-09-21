# Progress Log

## Session: 2026-09-21

### Phase 1: Freeze Oracle Reranking Protocol
- **Status:** complete
- Actions taken:
  - Created a separate C0-R1 plan.
  - Reviewed the completed C0 metrics and boundaries.
  - Froze candidate construction, score weights, paired statistics,
    and promotion gates.

### Phase 2: Implement CPU-Only Reranker
- **Status:** complete
- Actions taken:
  - Implemented deterministic object-frame candidate scoring.
  - Implemented object-grouped paired bootstrap.
  - Added focused unit tests.

### Phase 3: Server Smoke and Full Run
- **Status:** complete
- Actions taken:
  - Uploaded code and tests to the no-card server.
  - Ran seven unit tests successfully.
  - Ran the complete C0-R1 audit on the server.
- Result:
  - 54 eligible target events with two candidates each.
  - Selection change rate `0.1481`.
  - Correct-minus-shuffled evaluation gain `0.05806` with 95% grouped
    interval `[0.00608, 0.12886]`.
  - Frozen promotion gate is NO-GO.

### Phase 4: Result, Decision, and Commit
- **Status:** complete
- Actions taken:
  - Created the result report and compact summary.
  - Committed the frozen C0-R1 result as
    `3f37cb9 test(data): reject OakInk oracle future reranking`.

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| Full C0-R1 run failed while serializing NumPy `bool_` values | 1 | Added recursive conversion to built-in Python scalars and reran the server audit |

## Test Results

| Test | Input | Expected | Actual | Status |
|---|---|---|---|---|
| C0 prerequisite | server feasibility summary | all data gates pass | pass | ready |

## 5-Question Reboot Check

| Question | Answer |
|---|---|
| Where am I? | C0-R1 result commit |
| Where am I going? | implement reranker, run on server, decide GO/NO-GO |
| What is the goal? | test whether future receiver identity changes giver candidate ranking |
| What have I learned? | C0 data supports deterministic same-object candidate construction |
| What have I done? | completed the server C0-R1 audit, which is NO-GO |
