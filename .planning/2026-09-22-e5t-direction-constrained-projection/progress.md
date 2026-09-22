# Progress

## 2026-09-22 Session

- Recovered the prior GO result and protocol.
- Created a persistent execution plan.
- Started inspection of the existing solver, metrics, and tests.
- Confirmed remote frozen input hashes match the direction-decomposition run.
- Added the direction/projection solver and focused unit tests.
- Passed five local pure-NumPy tests and the same tests in the contactopt venv.
- Ran all arms on train chunk sub16_largebox_008_c000.
- Confirmed unconstrained continuation exactly matches frozen E5-T on that
  chunk after restoring seed 20260919.
- Committed solver and protocol as 5277970.
- Committed acceptance analysis as eda09aa.
- Started full train/dev runs for all five arms on the remote server.
- Completed all five arms and the 5,000-sample sequence-clustered analysis.
- Verified MANO reconstruction error, frozen zero wrist/object drift, and the
  full E5-T pass-decision regression path.
- Wrote the result report and compact analysis JSON.
- Committed the final result checkpoint as fc29cdc.
