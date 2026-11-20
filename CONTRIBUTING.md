# Contributing

## Workflow
1. **Pick your next issue** from [WORK_SEPARATION.md](docs/WORK_SEPARATION.md) (in order). Comment "started" on it.
2. **Branch** from up-to-date `main`: `<type>/<issue>-<short-name>`, e.g. `feat/9-gpu-preprocess`, `exp/17-e1a-tta`,
   `results/19-model-family`, `fix/4-resume-bug`.
3. **Commit** small and often, with an imperative message that references the issue: `Add rotate_batch with exact rot90 path (#9)`.
4. **Open a PR** early (draft is fine), using the PR template. Write `Closes #<n>` only in the PR that finishes the issue.
5. **Review:** at least **one approval**, preferably from the owner of the module you touch (see the roles table).
   Reviewers respond within 24 h. CI must be green.
6. **Merge** with *squash and merge*. Delete the branch.

Do not push directly to `main`. Do not force-push shared branches.

## Code style
* Python ≥ 3.10, type hints on public functions, docstrings on public functions/classes.
* `ruff check .` must pass (line length 120). Run `ruff check --fix .` before committing.
* Reusable logic goes in `galaxy/`. `scripts/` only parse arguments and call into the package. Notebooks only call
  into the package and plot.
* **Tests use synthetic data only** (`tests/conftest.py`). Mark GPU tests `@pytest.mark.gpu` and long ones `@pytest.mark.slow`.
  Optional heavy dependencies go through `pytest.importorskip("escnn")`.

## Protocol rules (non-negotiable, PROJECT_PLAN §3)
* Use `split_v1` only. Never regenerate it and never create a private split.
* Use the shared preprocessing (`GalaxyPreprocess`) and evaluation (`predict` / `compute_metrics` / `save_eval`).
  No private copies.
* **The test set is never used** for tuning, early stopping or model choice. Only `scripts/evaluate.py` touches it.
* Deep results: 3 seeds (0, 1, 2). Every result file stores its config and git commit.
* Changing a contract (PROJECT_PLAN §4) or a protocol decision (§3) needs agreement in an issue and an update of the plan.

## Running experiments
1. Smoke test first: `--set train.max_epochs=1 data.fraction=0.1`.
2. Add a row in [docs/RUNS.md](docs/RUNS.md) (status `planned` → `running`).
3. Run on Kaggle/Colab with the runner notebooks (#10). Pass `--resume` after a session ends.
4. Download `results/runs/<experiment>/<run_name>/` and commit it on a `results/<issue>-<name>` branch → PR.
5. Upload the `best.pt` of shared runs (model family, escnn) to the team checkpoint dataset and note the location in RUNS.md.

## Files that must never be committed
Raw/processed data (`*.h5`, `*.npy`, `*.parquet`), checkpoints (`*.pt`), API keys/tokens, large notebook outputs.
`.gitignore` covers most of these. Check `git status` before committing.

## Communication
* Use the issue thread for anything about a task (decisions stay findable). Use the group chat for quick coordination.
* **Blocked?** Comment on the blocking issue and tag its owner the same day.
* Weekly 15-minute sync: done / next / blockers. Update your issue checklists before it.
