# Findings and Decisions

## Existing Evidence

- D0-B contains 18,720 single-branch trajectories and passes its original
  data gates.
- D1 is a frozen NO-GO because single-state absolute/centered response
  prediction does not generalize to unseen mass/friction.
- A post-hoc audit found branch-dependent effects on seen response regimes
  but no reliable unseen-response ranking signal.
- Counterfactual quotient learning, active tactile perception, and
  history-conditioned context exist separately, so P0 must compare against
  matched no-probe, absolute, quotient, geometry, and oracle arms.

## Implementation Constraints

- The existing environment is deterministic and uses a 168D state plus a
  51D position-target action.
- Candidate branches must start from exactly the same post-probe MuJoCo
  state, including contact-bookkeeping flags.
- The full D0-C dataset is group-level and includes no-probe plus nested
  fixed/random probes at budgets 1, 2, and 4.
- Test thresholds are frozen before test access.

## Open Questions

- None requiring user input. All product and experiment choices are frozen
  in the approved plan and protocol.
