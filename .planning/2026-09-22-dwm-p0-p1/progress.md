# Progress Log

## Session: 2026-09-22

### Initialization
- **Status:** complete
- Actions taken:
  - Created the P0/P1 planning files.
  - Recorded the frozen scope, split, budgets, utility, and stop conditions.

### Current Work
- **Status:** in_progress
- Next action:
  - Wait for full D0-C generation, then run the frozen audit.

### Protocol and Environment
- **Status:** complete
- Result:
  - Added `checkpoint()` / `restore()` using `copy.deepcopy(MjData)`.
  - Verified exact branch reproduction and contact-mode equality on server.

### D0-C Smoke
- **Status:** complete
- Result:
  - Two repeated smoke generations were byte-identical at the stored-field
    maximum-difference level (`0.0`).
  - Probe nesting, rank consistency, target separation, utility spread, and
    split isolation passed.
  - Small-data contact-mode coverage failed as expected because the smoke
    has only 42 groups.
- Current action:
  - Full generation is running in `/root/autodl-tmp/dwm_d0c_probe_v1_20260922`.
