# Project Workflow

## Git checkpoints

- After every meaningful validated result, create a focused git commit before starting the next experiment.
- Treat a passed test suite for a new module, a completed smoke/formal experiment with a decision, or a verified improvement over a baseline as checkpoint-worthy.
- Stage only files that belong to the result. Never include unrelated user changes.
- Do not commit large datasets, model checkpoints, caches, or bulk generated outputs unless explicitly requested.
- Commit code, configs, focused tests, scripts, and concise result summaries together.
- Use a descriptive message such as `feat(world-model): add bounded learned residual scorer`.
