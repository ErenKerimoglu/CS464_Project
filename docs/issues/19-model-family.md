---
id: 19
title: "[Models] Model-family comparison: final 3-seed runs + main results table"
owner: berkay
labels: [area:models, type:experiment, priority:critical]
milestone: "M2 - Core Experiments"
week: W4
depends_on: [15]
---
## Goal
The core result table: classical vs. scratch CNN vs. pretrained CNNs vs. ViT vs. equivariant CNN.

## Context
Plan §6 (Family: 4 models × 3 seeds, `rotflip`, tuned). These runs are **reused** by E1 (`rotflip` arm), E2 (full fine-tuning arm), E3 (100 % point), E4 (CE baseline) and E5/E6 (evaluation). **Quality and on-time delivery matter for everyone.**

## Tasks
- [ ] Configs `configs/experiments/model_family/{small_cnn,resnet50,convnext_tiny,vit_small}.yaml` (tuned, from #15).
- [ ] Runs: 4 models × seeds 0/1/2 = 12 runs. Spread them across accounts (coordinate in `docs/RUNS.md`).
- [ ] **Keep the `best.pt` checkpoints** in the shared Drive/Kaggle dataset `galaxy10-checkpoints` (other people's evaluation-only experiments need them). Document the location.
- [ ] Commit the results (`eval_val`, `eval_test`, preds, train_log).
- [ ] Training curves figure (val macro-F1 vs. epoch, mean ± std band, one line per model) → `results/figures/model_family/curves.png`.
- [ ] `results/tables/main_table.md` via #16, including classical (#14) and escnn (#13) rows. Columns: acc, macro-F1, #params, GPU time.
- [ ] Significance: the best model vs. the runner-up.
- [ ] Draft `report/sections/model_family.md` (setup + results + discussion of CNN vs ViT inductive bias).

## Deliverables
12 runs, checkpoints shared, curves figure, main table, report paragraph.

## Acceptance criteria
- Done by **end of week 4**.
- The main table has every model family with mean ± std, and the best model is named for E5/E6.

## Blocks
#20, #21, #22, #23, #25
