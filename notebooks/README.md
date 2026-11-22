# notebooks/

Notebooks are **thin wrappers**. Reusable logic goes into the `galaxy/` package, not into notebook cells.

| Notebook | Issue | Purpose |
|----------|-------|---------|
| `00_eda.ipynb` | #2 | Exploratory data analysis (committed with outputs) |
| `01_pipeline_check.ipynb` | #9 | Input-pipeline check and throughput benchmark on a GPU |
| `02_build_kaggle_dataset.ipynb` | #10 | Builds the team Kaggle dataset `galaxy10-decals-processed-v1` (run once) |
| `kaggle_runner.ipynb` | #10 | Clone repo → install → attach data → run `scripts/*.py` on a Kaggle GPU → export results |
| `colab_runner.ipynb` | #10 | Same for Colab, with data and checkpoints on Google Drive (so `--resume` works across sessions) |

**Running experiments:** see CONTRIBUTING → *Running experiments* and *Results workflow*. Both runners take a
`BRANCH` (branch or commit) and a list of `RUN_CMDS`, call `galaxy.utils.env.setup_env()` for the paths, and end
by zipping the new `results/runs/...` files and printing the git commands to commit them.

Except for `00_eda.ipynb`, strip outputs before committing (`nbstripout --install` once per clone).
