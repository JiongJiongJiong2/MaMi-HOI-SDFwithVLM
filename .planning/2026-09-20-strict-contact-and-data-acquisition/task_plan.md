# Task Plan: Strict Contact Lifecycle and External Data Acquisition

## Goal

Decide and execute the next viable research path after the official EPIC
`3 mm` lifecycle gate failed but strict `1 mm` labels showed sufficient
events.

## Current Phase

Phase 2

## Phases

### Phase 1: Strict 1 mm Manifest and Video-Disjoint Split
- [x] Recompute episodes at `1 mm`
- [x] Build train/dev/test split with no shared videos
- [x] Freeze the strict-contact protocol before modeling
- **Status:** complete

### Phase 2: Strict-Contact B Pilot
- [ ] Train hold/onset/release baselines
- [ ] Compare against copy-current-state and fixed-threshold controls
- [ ] Evaluate by sequence-clustered paired intervals
- **Status:** pending

### Phase 3: External Data Gate
- [ ] Audit GRAB for release lifecycle and MANO contact maps
- [ ] Audit ARCTIC for bimanual release transitions and object topology
- [ ] Audit OakInk/H2O/DexYCB for complementary sequence supervision
- [ ] Record license, access method, fields, and split policy
- **Status:** pending

### Phase 4: Handover Decision
- [ ] Decide whether any acquired source has explicit handover evidence
- [ ] If absent, stop C and do not relabel simultaneous contact as handover
- **Status:** pending

### Phase 5: Mainline Decision
- [ ] Choose strict-contact B, release/re-contact refinement, or return to E5
- [ ] Freeze one paper claim and its baseline
- **Status:** pending

## Immediate Data Need

No new download is needed for Phase 1. It uses the existing EPIC-Contact
frame manifest and minimum-distance fields.

The next downloads are only needed for Phase 3:

1. GRAB, if single-hand release and dense MANO contact maps suffice.
2. ARCTIC, if bimanual sequences and dynamic contact fields are needed.
3. OakInk/H2O/DexYCB, only as secondary checks after the first two.

## Decisions Made

| Decision | Rationale |
|---|---|
| Official 3 mm lifecycle is NO-GO | 37 onset, 33 release, 17 complete train episodes |
| Strict 1 mm is a new candidate, not a replacement | It has enough events but was selected after sensitivity analysis |
| Handover remains blocked | EPIC has simultaneous contact, not direct role switching |

## Errors Encountered

| Error | Attempt | Resolution |
|---|---|---|
| Official split is not video-disjoint | 1 | Require a new split before learned evaluation |
