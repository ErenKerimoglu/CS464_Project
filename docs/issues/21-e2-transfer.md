---
id: 21
title: "[E2] Transfer learning: from scratch vs linear probe vs full fine-tuning"
owner: berkay
labels: [area:experiments, type:experiment, priority:high]
milestone: "M3 - Analysis"
week: W4-W5
depends_on: [12, 15, 19]
---
## Goal
Quantify how much ImageNet pretraining helps on galaxy images, for a CNN and a ViT (RQ2).

## Context
Plan §6 (E2: ResNet-50 and ViT-S/16; scratch + linear new, full FT reused from #19). Trainer `freeze_backbone` from #4.

## Tasks
- [ ] Linear probe: `pretrained=True, freeze_backbone=True`, head-only LR 1e-3, 20 epochs, `rotflip`. Runs `e2_transfer/{resnet50,vit_small}_linear`, seeds 0/1/2.
  - [ ] *(Fast cross-check)* extract frozen `forward_features` once and fit sklearn LR. Linear-probe numbers should be similar.
- [ ] From scratch: `pretrained=False`, scratch recipe (LR 1e-3, 60 epochs, warm-up 5 epochs for ViT), `rotflip`. Runs `e2_transfer/{resnet50,vit_small}_scratch`, seeds 0/1/2. ViT from scratch on about 12 k images is expected to be weak, which is part of the finding.
- [ ] Table: model × {scratch, linear, full FT} → acc / macro-F1, mean ± std. Also report GPU time per mode.
- [ ] Figure: grouped bars with error bars (`plotting.bar_with_err`).
- [ ] Draft `report/sections/e2_transfer.md`. Discuss domain shift (natural images vs. astronomical g/r/z) and the CNN vs. ViT difference.

## Deliverables
12 runs, table, figure, report paragraph.

## Acceptance criteria
- All 6 cells × 3 seeds are present. Linear-probe and scratch runs used exactly the same split and preprocessing as full FT (check the stored configs).
