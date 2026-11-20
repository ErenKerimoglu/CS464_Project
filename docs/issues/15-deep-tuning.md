---
id: 15
title: "[Models] Optuna tuning harness + hyperparameter search for deep models"
owner: berkay
labels: [area:models, type:experiment, priority:critical]
milestone: "M2 - Core Experiments"
week: W3
depends_on: [4, 12]
---
## Goal
Fair, budget-aware tuning, so the model comparison is not an artifact of default hyperparameters.

## Context
Plan §3.2, §7 (budget: about 15 trials per model with a short schedule). Tuning uses **val only**.

## Tasks
- [ ] `scripts/tune.py --config <model cfg> --n-trials 15 --epochs 8 [--fraction 0.5] --study-name <name>`
  - [ ] Generic search space from config (`tune.space:` block), so Eftelya can reuse it for escnn (#13). Defaults: `lr` log-uniform [1e-5, 1e-3] (pretrained) or [1e-4, 3e-3] (scratch), `weight_decay` log-uniform [1e-3, 0.1], `label_smoothing` ∈ {0, 0.1}, `drop_path` [0, 0.2] (ConvNeXt/ViT).
  - [ ] `MedianPruner` with per-epoch `trial.report(val_macro_f1)`.
  - [ ] Storage `sqlite:///results/optuna/<study>.db` (committed, it is small), so studies can resume across Kaggle sessions.
  - [ ] Write the best params to `configs/experiments/model_family/<model>.yaml` (tuned config = base model config + overrides).
- [ ] Run studies for `small_cnn`, `resnet50`, `convnext_tiny`, `vit_small` (one seed). Spread them across Kaggle accounts and log them in `docs/RUNS.md`.
- [ ] Figures per model: optimization history + parameter importance (`optuna.visualization.matplotlib`) → `results/figures/tuning/`.

## Deliverables
`scripts/tune.py`, 4 studies, 4 tuned configs, figures.

## Acceptance criteria
- Each tuned config beats or equals the default config on val macro-F1 (show both numbers).
- Tuned configs are merged before #17, #18, #19, #20 and #21 start final runs (**end of week 3**).

## Blocks
#17, #18, #19, #20, #21
