# Progress Log

## Session: 2026-09-22

### D0-A: MuJoCo Right-Hand Environment
- **Status:** complete
- Actions taken:
  - Created the isolated DWM plan.
  - Froze the 168D state and 51D action layouts.
  - Confirmed the HandX/InterMimic right-hand asset has 51 actuators.
  - Added the vendored MJCF, license, and third-party notice.
  - Implemented object configs, MJCF derivation, reset, PD control,
    state extraction, contact forces, and contact mode.
  - Added and passed eight server tests.

## Test Results

| Test | Input | Expected | Actual | Status |
|---|---|---|---|---|
| Asset actuator audit | right-hand MJCF | 51 motor actuators | 51 | pass |
| MuJoCo server tests | 12-object environment | finite and deterministic | 8 passed | pass |
| Branch excitation smoke | 60 resets | >=80% with 4+ distinct outcomes | 86.7% | pass |
| Reduced dataset schema | 78 trajectories | 168D state / 51D action | matched | pass |

### D0-B: Counterfactual Dataset
- **Status:** in_progress
- Actions taken:
  - Implemented all 13 frozen action branches.
  - Calibrated per-shape contact offsets and the contact mix.
  - Added the split-disjoint dataset writer.
  - Generated a reduced server smoke dataset.
- Pending:
  - Run the full 17,280-trajectory generation on the larger server.
- Errors fixed:
  - Added missing MuJoCo inertial records.
  - Replaced zero-order-hold PD with per-substep PD.
  - Replaced `cfrc_ext` with `mj_contactForce` aggregation.
  - Calibrated contact heights and wrist yaw.

## 5-Question Reboot Check

| Question | Answer |
|---|---|
| Where am I? | D0-B full dataset generation |
| Where am I going? | finish D0-B gates, D1 identifiability, D2 utility |
| What is the goal? | build a real same-state multi-action simulator study |
| What have I learned? | existing MaMi action data cannot serve as counterfactual truth |
| What have I done? | completed D0-A and reduced D0-B implementation smoke |
