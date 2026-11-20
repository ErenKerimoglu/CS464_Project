## What
Closes #<issue>

<!-- One or two sentences: what this PR adds or changes. -->

## How it was checked
- [ ] `pytest` and `ruff check .` pass locally
- [ ] New code has tests (synthetic data only)
- [ ] For runs: configs committed, results in the `results/runs/...` schema, row added/updated in `docs/RUNS.md`

## Protocol checklist (PROJECT_PLAN §3)
- [ ] Uses `split_v1` and the shared `GalaxyPreprocess` / `evaluate()`, with no private copies
- [ ] The test set is not used for tuning, early stopping or model selection
- [ ] No data, checkpoints or large files committed
- [ ] Contracts in PROJECT_PLAN §4 unchanged (or the change is agreed in the issue and the plan is updated)

## Results / screenshots (if any)
