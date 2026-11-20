---
id: 10
title: "[Infra] Kaggle/Colab runner notebooks, shared data hosting & run registry"
owner: ege
labels: [area:infra, type:feature, priority:high]
milestone: "M1 - Pipelines & Baselines"
week: W2
depends_on: [3, 4]
---
## Goal
Any member can launch any experiment on free GPUs in under 5 minutes, and we track who runs what.

## Context
Plan §7 (compute budget: about 65 GPU-h, which needs all 5 Kaggle accounts), NFR-4.

## Tasks
- [ ] Upload `data/processed/` (`images_256.npy`, `meta.csv`) to a **private Kaggle Dataset** (`galaxy10-decals-processed-v1`) and add all members. Put the same files in a shared Google Drive folder for Colab.
- [ ] `notebooks/kaggle_runner.ipynb`: cell 1 clones the repo at a branch/commit (parameter), cell 2 runs `pip install -e .[all]`, cell 3 symlinks the Kaggle dataset to `data/processed`, cell 4 sets `RUN_CMD` (one or more `scripts/train.py` / `evaluate.py` commands), cell 5 runs it, cell 6 copies `results/runs/...` and checkpoints to `/kaggle/working` and prints the `git add` instructions.
- [ ] `notebooks/colab_runner.ipynb`: same idea with Drive mount, checkpoints written to Drive, so `--resume` works across sessions.
- [ ] `galaxy/utils/env.py`: detect Kaggle/Colab/local and set `DATA_DIR`/`CKPT_DIR` (env vars override).
- [ ] `docs/RUNS.md`: run registry table (`experiment | run_name | seeds | owner | platform/account | status | checkpoint location | commit`). Put a sample row in it.
- [ ] Results workflow (documented in CONTRIBUTING): download `results/runs/<...>/` from the notebook output, commit it on a branch `results/<experiment>`, open a PR.
- [ ] Do a full dry run with `small_cnn` for 1 epoch on Kaggle **and** Colab.

## Deliverables
Two notebooks, `env.py`, `docs/RUNS.md`, CONTRIBUTING section.

## Acceptance criteria
- Another member (not Ege) runs the Kaggle notebook successfully following only the docs. Note who in this issue.

## Blocks
All training runs from M1 onward.
