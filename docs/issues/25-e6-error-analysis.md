---
id: 25
title: "[E6] Error analysis: confusion matrices, Grad-CAM (galaxy vs background stars), t-SNE"
owner: eftelya
labels: [area:experiments, type:experiment, priority:high]
milestone: "M3 - Analysis"
week: W5
depends_on: [11, 13, 19]
---
## Goal
Understand **why** models fail. Which classes get confused, what the models look at, and how the learned feature spaces differ (RQ4, RQ6).

## Context
Plan §6 (E6: best model from #19, SmallCNN, escnn). Uses saved predictions (`preds_*.npz`) and the shared checkpoints. No training.

## Tasks
- [ ] `galaxy/analysis/confusion.py`: row-normalized confusion matrices (mean over seeds) for the best model, SmallCNN, escnn and the best classical model. Also a "top confused pairs" table (e.g. Barred vs Unbarred Tight Spiral, Round Smooth vs In-between).
- [ ] `galaxy/analysis/gradcam.py` (using `pytorch-grad-cam`; ViT through `vit_reshape_transform` from #12):
  - [ ] Grid figure: for each class, 2 correct + 2 wrong test examples, image + CAM overlay.
  - [ ] **Quantitative on-galaxy score:** the fraction of CAM mass inside the central-galaxy mask (`segmentation.segment_central_source` from #11), compared with mass on *other* detected sources (background stars/companions). Report its distribution for correct vs. wrong predictions and per model.
  - [ ] *(Optional)* CAM of the same galaxy at 0°/45°/90° for ResNet-50 `none` vs `rotflip` vs escnn, a visual link to E1.
- [ ] `galaxy/analysis/tsne.py`: `forward_features` on the test set → t-SNE (perplexity 30, PCA-50 init), colored by class, for best model vs. SmallCNN (vs. escnn). Also a silhouette score per embedding.
- [ ] Look at 20 of the most confident errors and label the cause (label noise / ambiguous / artifact / companion), then summarize them in a table.
- [ ] Draft `report/sections/e6_error_analysis.md`.

## Deliverables
`confusion.py`, `gradcam.py`, `tsne.py`, figures in `results/figures/e6/`, tables, report paragraph.

## Acceptance criteria
- The on-galaxy score is reported for at least 3 models, with a statement on whether errors correlate with off-galaxy attention.
- The figures follow the shared plotting style (#16).
