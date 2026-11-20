---
id: 13
title: "[Equivariance] Train & tune the escnn model (3 seeds) for E1 and the family comparison"
owner: eftelya
labels: [area:orientation, type:experiment, priority:high]
milestone: "M1 - Pipelines & Baselines"
week: W2-W3
depends_on: [4, 8, 9]
---
## Goal
Final, tuned equivariant-CNN results. They are a key row in the main table and a key curve in the orientation figure.

## Context
Plan §3.2 (scratch recipe: 60 epochs, LR 1e-3), §6 run matrix (escnn: `rotflip` ×3, optional `none` ×3).

## Tasks
- [ ] Smoke run (1 epoch, 10 % data) on Kaggle. Record time/epoch and memory in `docs/RUNS.md`.
- [ ] Tune LR, weight decay and width multiplier with `scripts/tune.py` once #15 lands (about 12 trials, 15 epochs, MedianPruner). If #15 is late, use a small manual grid (3 LRs × 2 widths) and document it.
- [ ] Choose D4 vs C8 on validation macro-F1 (one seed each) and report the winner with the reason.
- [ ] Final runs: `model_family/escnn_d4` (or c8) with `rotflip`, seeds 0/1/2, saved in the §4.4 schema.
- [ ] Extra runs: the same model with policy `none` (seeds 0/1/2) under `e1_orientation/escnn_d4_none`. This tests whether an equivariant net still needs augmentation (interesting for RQ1).
- [ ] Commit the results via PR.
- [ ] Write a short paragraph for the report (`report/sections/escnn.md`): architecture, group, #params, training recipe.

## Deliverables
Configs in `configs/experiments/model_family/` and `e1_orientation/`, result files, report paragraph.

## Acceptance criteria
- 6 result folders (2 policies × 3 seeds) with `eval_test.json` and `eval_val.json`.
- Test accuracy is competitive with SmallCNN. If it is clearly worse, post the analysis and a fix attempt here.

## Blocks
#20, #23, #25
