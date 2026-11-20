---
id: 20
title: "[E3] Data efficiency: learning curves at 10/25/50/100% of training data"
owner: ege
labels: [area:experiments, type:experiment, priority:high]
milestone: "M2 - Core Experiments"
week: W4
depends_on: [9, 13, 15, 19]
---
## Goal
Learn how performance scales with labeled data for a pretrained CNN, a scratch CNN and the equivariant CNN (RQ3). Do inductive biases (pretraining, equivariance) matter more when data is scarce?

## Context
Plan §4.1 (nested `fraction` subsets), §6 (E3: 3 models × 3 fractions × 3 seeds; 100 % reused from #19 and #13).
Val/test sets stay **full**. Only train is subsampled.

## Tasks
- [ ] Configs `configs/experiments/e3_data_eff/*.yaml` that set `data.fraction ∈ {0.10, 0.25, 0.50}`.
  Keep **the number of optimizer steps roughly comparable**: scale `max_epochs` up for small fractions (e.g. 10 % → ×4 epochs, capped) and document the rule. Early stopping still uses the full val set.
- [ ] Runs: `resnet50_f{010,025,050}`, `small_cnn_f…`, `escnn_d4_f…` × seeds 0/1/2 (27 small runs).
- [ ] Check the nesting: the run log prints the train-set size and class counts per fraction.
- [ ] Figure: macro-F1 (and accuracy) vs. training-set fraction on a log x-axis, mean ± std band, one line per model, plus classical XGBoost from #24 when available. Use `MODEL_COLORS`.
- [ ] Table: relative performance at 10 % vs. 100 % per model.
- [ ] Draft `report/sections/e3_data_efficiency.md`.

## Deliverables
27 runs, figure, table, report paragraph.

## Acceptance criteria
- All points have 3 seeds. The figure includes the 100 % points from #19/#13.
