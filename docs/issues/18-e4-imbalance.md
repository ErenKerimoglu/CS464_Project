---
id: 18
title: "[E4] Class imbalance: weighted cross-entropy, focal loss, oversampling"
owner: eftelya
labels: [area:experiments, type:experiment, priority:high]
milestone: "M2 - Core Experiments"
week: W3-W4
depends_on: [4, 15]
---
## Goal
Find out whether imbalance handling improves macro-F1 and recall on rare classes (Cigar Shaped Smooth: 334 images, Disturbed: 1081) without hurting accuracy.

## Context
Plan §1 RQ4, §4.3 (loss/sampler plug-in points in the Trainer), §6 (E4: ResNet-50, 3 methods × 3 seeds; plain CE reused from #19).

## Tasks
- [ ] `galaxy/training/losses.py` `build_loss(cfg, class_counts)`:
  - [ ] `ce` (with label smoothing, the default)
  - [ ] `weighted_ce`: weights ∝ 1/n_c^β, normalized to mean 1. β=1 by default (also try effective-number weights from Cui et al. 2019 if time allows).
  - [ ] `focal`: γ=2 (configurable), optional α = class weights. Numerically stable with AMP (compute in fp32).
- [ ] `galaxy/training/samplers.py` `build_sampler(cfg, labels)`: `shuffle` (default) and `oversample` (`WeightedRandomSampler`, weights 1/n_c, `num_samples = len(train)`, so each epoch has balanced classes in expectation).
- [ ] `tests/test_losses.py`: focal with γ=0 equals CE, weighted CE with uniform weights equals CE, and the oversampler gives a roughly uniform class histogram over 10 k draws.
- [ ] Runs (tuned ResNet-50 config, `rotflip`): `e4_imbalance/resnet50_wce`, `resnet50_focal`, `resnet50_oversample`, seeds 0/1/2.
- [ ] *(Optional)* post-hoc logit adjustment of the plain CE model (subtract τ·log prior). Evaluation only, no training cost.
- [ ] Table: method × {acc, macro-F1, balanced acc, recall of the 3 rarest classes}. Figure: per-class recall heatmap (methods × classes).
- [ ] Draft `report/sections/e4_imbalance.md`.

## Deliverables
`losses.py`, `samplers.py`, tests, 9 runs, table + heatmap, report paragraph.

## Acceptance criteria
- All 9 runs are committed. Comparisons with CE include the bootstrap CI from #16.
