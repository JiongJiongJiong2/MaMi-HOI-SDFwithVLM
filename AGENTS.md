# Project Workflow

## Research entry and scope

- Read `RESEARCH_START_HERE.md` at task start and after context recovery. Follow its links to the current protocol and handoff; do not load every historical report.
- Current research decisions live in that entry. Dated reports preserve historical evidence; their "next step" paragraphs are not standing instructions. If code, artifacts and summaries disagree, record the conflict and check evidence before updating status.
- One implementation task answers one named experiment question. Keep the research objective and protocol fixed while repairing implementation. A valid negative result completes an experiment.
- Fix necessary implementation bugs autonomously within the task. Record affected old results and rerun the relevant controls when a correctness fix changes numbers. Do not silently change labels, splits, primary metrics, baselines, candidate/compute budgets or core mechanisms to obtain PASS; document a new protocol version and proposed scope change.
- Record unrelated improvements without implementing them. After a repair, return to the original experiment. Repeated failures require a brief root-cause reassessment, not unbounded refactoring.
- Keep implementation, smoke, formal run and verified scientific result as separate statuses. Preserve NO-GO results and distinguish protocol failure from insufficient evidence; never generalize a failed version to all world models.
- Before ending or handing off, update the current experiment handoff with commits, actual commands, input/config identities, artifacts, running jobs and next action. No invented completed tests or server state.
- Do not use another task's worktree, GPU process, output directory or uncommitted files as your own. Check existing jobs before starting a remote run. Credentials stay outside reports.
- Do not overwrite another task's `.planning/.active_plan`. Use task-specific plan directories when needed; these are work logs, not competing research master plans.

## Git checkpoints

- After every meaningful validated result, create a focused git commit before starting the next experiment.
- Treat a passed test suite for a new module, a completed smoke/formal experiment with a decision, or a verified improvement over a baseline as checkpoint-worthy.
- Stage only files that belong to the result. Never include unrelated user changes.
- Do not commit large datasets, model checkpoints, caches, or bulk generated outputs unless explicitly requested.
- Commit code, configs, focused tests, scripts, and concise result summaries together.
- Use a descriptive message such as `feat(world-model): add bounded learned residual scorer`.
