# notebooks/

Notebooks are **thin wrappers**. Reusable logic goes into the `galaxy/` package, not into notebook cells.

| Notebook | Issue | Purpose |
|----------|-------|---------|
| `00_eda.ipynb` | #2 | Exploratory data analysis (committed with outputs) |
| `kaggle_runner.ipynb` | #10 | Clone repo → install → link data → run `scripts/*.py` on a Kaggle GPU |
| `colab_runner.ipynb` | #10 | Same for Colab, with checkpoints on Google Drive |

Except for `00_eda.ipynb`, strip outputs before committing (`nbstripout --install` once per clone).
