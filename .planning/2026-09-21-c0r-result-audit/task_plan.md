# Task Plan: C0-R1 Result Audit

## Goal

Determine whether the C0-R1 NO-GO is a valid rejection of the frozen
claim, a consequence of implementation or score-design problems, or an
underpowered test caused by candidate availability.

The audit does not alter the frozen C0-R1 result.

## Current Phase

Phase 1

## Phases

### Phase 1: Code and Logic Audit
- [x] Trace candidate construction, score construction, and evaluation
- [x] Identify circular or non-independent metrics
- [x] Identify assumptions that make the gate hard to interpret
- **Status:** complete

### Phase 2: Server Statistical and Data Audit
- [x] Recompute paired top-1 transition counts and exact McNemar checks
- [x] Quantify candidate availability under alternative valid-frame rules
- [x] Quantify condition/evaluation feature overlap
- **Status:** complete

### Phase 3: Verdict and Scope
- [x] Decide whether the NO-GO is implementation-correct
- [x] Separate the rejected claim from unblocked research branches
- [x] Record implications for C, B, D, and E5
- **Status:** complete

### Phase 4: Report and Commit
- [x] Add an audit report and machine-readable summary
- [ ] Commit the audit without rewriting the original result
- **Status:** in_progress

## Boundary

This audit can explain or localize the NO-GO. It cannot convert the
frozen C0-R1 result into a pass. Any repaired protocol is a new
experiment and must be frozen separately.
