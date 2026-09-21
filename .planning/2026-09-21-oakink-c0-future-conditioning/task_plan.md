# Task Plan: OakInk C0 Future-Conditioning Feasibility

## Goal

Determine whether OakInk can support a controlled experiment in which a
future giver/receiver handover requirement changes the earlier giver
grasp. C0 is a CPU-only data and conditioning feasibility stage; it does
not train a generative model and does not claim that future
conditioning improves MaMi.

## Current Phase

Phase 1

## Phases

### Phase 0: Server Identity and Existing Data Audit
- [x] Confirm the connected AutoDL instance is the correct same-day
  instance
- [x] Verify OakInk code hashes against the main repository
- [x] Verify derived OakInk manifest hashes and structural counts
- **Status:** complete

### Phase 1: Freeze C0 Protocol and Data Schema
- [x] Define event-centered input, target, and control fields
- [x] Freeze the CPU-only feasibility gates
- [x] Define what remains for the later oracle optimization stage
- **Status:** complete

### Phase 2: Build Event-Centered OakInk Dataset
- [x] Extract hand vertices, hand joints, object transforms, and local
  object meshes around primary handover transitions
- [x] Preserve participant-disjoint splits and source provenance
- [x] Emit compact arrays suitable for CPU and server reproduction
- **Status:** complete

### Phase 3: C0 Data-Feasibility Audit
- [x] Measure window completeness and geometry validity
- [x] Measure giver-current versus receiver-future contact-region
  separability
- [x] Quantify same-object future-target contrast
- [x] Record hard-negative candidates and unresolved task-alignment risks
- **Status:** complete

### Phase 4: Local Run, Server Verification, and Decision
- [ ] Run the full C0 dataset builder locally
- [x] Reproduce the feasibility audit on the no-card server
- [ ] Commit the protocol, code, tests, compact report, and hashes
- [x] Decide GO, CONDITIONAL, or NO-GO for the oracle optimization stage
- **Status:** in_progress

## Immediate Data Need

No new download and no GPU are required. The local raw annotations are:

```text
E:/HOI/TMP/OakInk-v1/Image/anno_v2.1.zip
E:/HOI/TMP/OakInk-v1/shape/OakInkObjectsV2.zip
E:/HOI/TMP/OakInk-v1/shape/metaV2.zip
```

The server already has the matching compact OakInk handover manifest.
Only the C0-derived arrays and scripts will be uploaded.

## Decisions Made

| Decision | Rationale |
|---|---|
| C0 is data and conditioning feasibility only | The audit found no evidence yet that future handover changes an earlier grasp |
| Do not retrain or tune the current OakInk detector | Detection is frozen as event extraction infrastructure |
| Keep the current OakInk test as exploratory | Its outcomes participated in earlier post-processing design |
| Keep the MaMi task-alignment decision explicit | OakInk is cross-person; MaMi is not automatically the same action schema |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| Active planning files still described the EPIC route | 1 | Created an isolated C0 plan and switched the active pointer |
