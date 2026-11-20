---
id: 12
title: "[Models] Pretrained backbones: ResNet-50, ConvNeXt-Tiny, ViT-S/16 (timm)"
owner: berkay
labels: [area:models, type:feature, priority:critical]
milestone: "M1 - Pipelines & Baselines"
week: W2
depends_on: [6]
---
## Goal
The three ImageNet-pretrained model families, each usable for full fine-tuning, linear probing, from-scratch training and Grad-CAM.

## Context
Plan §3.2 (timm names, recipe), §4.2 (contract), §6 (run matrix).

## Tasks
- [ ] `galaxy/models/timm_models.py`: register `resnet50` (`resnet50.tv_in1k`), `convnext_tiny` (`convnext_tiny.fb_in1k`), `vit_small` (`vit_small_patch16_224.augreg_in21k_ft_in1k`).
  - [ ] `pretrained` flag (False gives a random init for E2 scratch runs).
  - [ ] Replace the classifier head with 10 classes. Optional dropout / drop-path rate from config.
  - [ ] `forward_features` = pooled pre-logit embedding (`timm` `forward_head(..., pre_logits=True)`).
  - [ ] `gradcam_target_layer`: ResNet `layer4[-1]`, ConvNeXt `stages[-1]`, ViT `blocks[-1].norm1`, plus a `vit_reshape_transform` helper for pytorch-grad-cam.
  - [ ] `backbone_parameters()` / `head_parameters()` for the linear probe (`freeze_backbone` in #4) and per-group LR.
- [ ] Configs: `configs/models/resnet50.yaml`, `convnext_tiny.yaml`, `vit_small.yaml` (input 224, LR 1e-4, 30 epochs, `rotflip`).
- [ ] Note: models use our per-band dataset stats, not ImageNet mean/std (plan §3.1). Write a sentence on this choice for the report.
- [ ] Smoke runs on Kaggle: 2 epochs each, 100 % data. Record time/epoch, peak memory and the largest batch size that fits a T4 with AMP in `docs/RUNS.md`. This feeds the compute budget.
- [ ] The generic model tests (#6) pass for all three with `pretrained=False` (no downloads in CI).

## Deliverables
`timm_models.py`, three configs, smoke-run table.

## Acceptance criteria
- After 2 epochs, val accuracy is far above chance (> 60 %) for all three.
- Time/epoch for each model is posted here.

## Blocks
#15, #17, #19, #21, #25
