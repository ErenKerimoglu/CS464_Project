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
3. Run on Kaggle or Colab with the runner notebooks (#10):
   * [`notebooks/kaggle_runner.ipynb`](notebooks/kaggle_runner.ipynb): attach the team dataset, set `BRANCH` and
     `RUN_CMDS`, then *Run All* (short runs) or *Save Version → Save & Run All* (long runs, up to 12 h in the
     background). Checkpoints go to `/kaggle/working/checkpoints`.
   * [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb): same, with checkpoints on Google Drive.
   * After a session ends, run the same commands with `--resume`. On Kaggle, first attach the previous version's
     output and set `RESUME_FROM`. Training continues exactly from `last.pt`, including `train_log.csv`.
4. Commit the results (below). Upload the `best.pt` of shared runs (model family, escnn) to the team checkpoint
   dataset and note the location in RUNS.md.

## Results workflow
Result files are small (metrics JSON, float16 predictions, logs) and are committed. Checkpoints never are.
1. The last cell of the runner notebook copies every result file written in that session to `results_export/`,
   zips it, and prints the exact git commands. Download `results_export.zip` (Kaggle: **Output** tab of the saved
   version · Colab: it is in your Drive folder).
2. In your local clone:
   ```bash
   git checkout main && git pull
   git checkout -b results/<experiment>             # e.g. results/e3_data_eff
   unzip -o results_export.zip -d .                 # extracts into results/runs/<experiment>/<run_name>/seed<k>/
   git add results/runs/<experiment>
   git commit -m "Add <experiment> results (#<issue>)"
   git push -u origin results/<experiment>
   ```
3. Open a PR with `Closes #<issue>` if it finishes the issue, and update the run's row in `docs/RUNS.md`
   (status `results merged (#PR)`, checkpoint location, commit). Check that every `eval_*.json` has a clean
   `git_commit` (no `-dirty` suffix): the runner notebooks always run a clean checkout.

## Files that must never be committed
Raw/processed data (`*.h5`, `*.npy`, `*.parquet`), checkpoints (`*.pt`), API keys/tokens, large notebook outputs.
`.gitignore` covers most of these. Check `git status` before committing.

## Communication
* Use the issue thread for anything about a task (decisions stay findable). Use the group chat for quick coordination.
* **Blocked?** Comment on the blocking issue and tag its owner the same day.
* Weekly 15-minute sync: done / next / blockers. Update your issue checklists before it.
