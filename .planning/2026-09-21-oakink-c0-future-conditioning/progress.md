# Progress Log

## Session: 2026-09-21

### Phase 0: Server Identity and Existing Data Audit
- **Status:** complete
- Actions taken:
  - Connected to the AutoDL no-card instance.
  - Confirmed the same container id used earlier in the day.
  - Verified script and manifest SHA-256 values.
  - Re-read the manifest and candidate arrays.
- Result:
  - Existing OakInk code and data are internally consistent.
  - No stale OakInk artifact was promoted by the audit.

### Phase 1: Freeze C0 Protocol and Data Schema
- **Status:** complete
- Actions taken:
  - Created an isolated C0 plan.
  - Froze the event window, target fields, split policy, and feasibility
    gates.
- Files created/modified:
  - `.planning/2026-09-21-oakink-c0-future-conditioning/`
  - `docs/experiments/oakink-c0-future-conditioning-protocol-2026-09-21.md`

### Phase 2: Build Event-Centered OakInk Dataset
- **Status:** complete
- Actions taken:
  - Extracted 106 primary events from 126 retained sequences.
  - Stored event-centered hand vertices, hand joints, object transforms,
    local object meshes, current giver regions, and future receiver
    regions.
  - Uploaded the compact data to the no-card server.
- Result:
  - 33 train, 36 val, and 37 test events.
  - 65 unique local object meshes.
  - Compact dataset size is about 94 MB before the audit report.

### Phase 3: C0 Data-Feasibility Audit
- **Status:** complete
- Actions taken:
  - Started the full audit on the server after the local extraction.
  - Corrected repeated `npz` decompression by loading every array once.
  - Reran the audit on the no-card server.
- Result:
  - 82 complete events, split 23 train, 30 val, and 29 test.
  - Geometry reconstruction maximum error is `1.64e-8 m`.
  - 21 same-object conditioning-contrast groups and 79 hard controls.
  - All C0 data gates pass.

### Phase 4: Local Run, Server Verification, and Decision
- **Status:** complete
- Actions taken:
  - Completed the full local event extraction.
  - Uploaded the compact dataset and scripts to the server.
  - Ran the five C0 unit tests on the server with `unittest`.
  - Created the result report and machine-readable summary.
- Decision:
  - C0 data feasibility is GO.
  - The oracle optimization stage is authorized, but no causal claim is
    supported yet.
- Commit:
  - `b02b6ed test(data): validate OakInk C0 future-conditioning data`

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| Server C0 audit import failed because `audit_arctic_handover_gate.py` was not copied | 1 | Copied the matching file from `oakink_scripts` and reran the audit |
| C0 audit repeated `npz` decompression inside loops | 1 | Loaded each `npz` once into an in-memory array dictionary; audit then completed in about two minutes |

## Test Results

| Test | Input | Expected | Actual | Status |
|---|---|---|---|---|
| Server identity continuity | container id and same-day artifact timestamps | same instance | matched | pass |
| OakInk code hash audit | five server scripts | match local repository | all matched | pass |
| OakInk manifest hash audit | summary, candidates, trajectories | match audited local copies | all matched | pass |
| Manifest structure | 126 trajectories and 110 candidates | participant-disjoint counts | matched | pass |

## 5-Question Reboot Check

| Question | Answer |
|---|---|
| Where am I? | Phase 1: freeze C0 protocol and schema |
| Where am I going? | build event dataset, audit feasibility, server verify |
| What is the goal? | determine whether C0 can support future-conditioned oracle optimization |
| What have I learned? | the server instance and existing OakInk artifacts are correct |
| What have I done? | completed C0 extraction and server feasibility audit |
