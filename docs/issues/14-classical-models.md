---
id: 14
title: "[Classical] PCA pipeline + Logistic Regression / SVM-RBF / XGBoost with Optuna"
owner: yunus
labels: [area:classical, type:experiment, priority:high]
milestone: "M2 - Core Experiments"
week: W3
depends_on: [5, 7, 11]
---
## Goal
Tuned classical baselines, reported in the same format as the deep models.

## Context
Plan §3.3 (feature reduction, tuning, seeds), §4.4 (results schema), §4.5 (`baselines_classical`).

## Tasks
- [ ] `galaxy/features/pipeline.py`: `load_feature_matrix(split, feature_sets=("hog","color","morph"))` joins the parquet caches on `orig_index`. `build_preprocessor()` is a `ColumnTransformer`: HOG → StandardScaler → PCA(0.95), and color/morph → median imputer → StandardScaler.
- [ ] `scripts/train_classical.py --model {logreg,svm_rbf,xgboost} [--tune --n-trials 50] [--seed k] [--features ...] [--fraction f]`
  - [ ] LR: `C` (log scale), `class_weight ∈ {None, balanced}`, multinomial
  - [ ] SVM-RBF: `C`, `gamma` (log scale), `class_weight`, `probability=True` for the final fit (or calibrated) so probabilities can be saved
  - [ ] XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `reg_lambda`. Early stopping on val. `tree_method="hist"` (GPU if available).
  - [ ] The Optuna objective is **val** macro-F1. Save the study to `results/optuna/classical_<model>.db` and the best params to `configs/experiments/baselines_classical/<model>_best.yaml`.
  - [ ] Final fit on train with the best params, then `save_eval` for val and test (#5) with probabilities.
- [ ] Feature ablation (one table): HOG only / color only / morph only / all (XGBoost, best params).
- [ ] XGBoost: 3 seeds. LR/SVM: 1 run (deterministic, state it).

## Deliverables
`pipeline.py`, `scripts/train_classical.py`, best-param configs, result files under `results/runs/baselines_classical/`, ablation table in this issue.

## Acceptance criteria
- `python scripts/aggregate_results.py` (from #16) shows the classical rows next to the deep models.
- The feature ablation is reported, and PCA dimensionality / explained variance is noted.

## Blocks
#22, #24
