---
id: 6
title: "[Models] Model registry + SmallCNN trained from scratch"
owner: berkay
labels: [area:models, type:feature, priority:high]
milestone: "M0 - Foundations"
week: W1
depends_on: []
---
## Goal
A single `build_model(name, ...)` entry point and the from-scratch CNN baseline.

## Context
Plan §4.2 (model contract). Every model must expose `forward`, `forward_features`, `gradcam_target_layer` and `input_size`.

## Tasks
- [ ] `galaxy/models/registry.py`: `register_model` decorator, `build_model(name, num_classes=10, pretrained=True, **kwargs)`, `list_models()`. A clear error message for unknown names.
- [ ] `galaxy/models/small_cnn.py` `SmallCNN`:
  - [ ] 4 conv blocks (conv3×3-BN-ReLU ×2 + maxpool) with widths 32→64→128→256, global average pooling, dropout 0.3, linear head. About 1–2 M parameters.
  - [ ] Configurable `input_size` (default 128, set in `configs/models/small_cnn.yaml`).
  - [ ] `forward_features` returns the GAP embedding. `gradcam_target_layer` = last conv block.
- [ ] `configs/models/small_cnn.yaml`: model name, input size, scratch LR 1e-3, 60 epochs (plan §3.2).
- [ ] `galaxy/models/utils.py`: `count_params(model)`, `model_summary(model, input_size)`.
- [ ] `tests/test_models.py`: for every registered model, check output shape `[B,10]`, `forward_features` shape, and that a backward pass works (CPU, tiny batch). Parametrize it so models added in #8 and #12 are tested automatically. Skip models whose optional dependency is missing (`pytest.importorskip`).

## Deliverables
`registry.py`, `small_cnn.py`, `utils.py`, config, tests.

## Acceptance criteria
- `build_model("small_cnn")` works and the tests pass in CI.
- Once #4 and #9 are merged: a 1-epoch smoke run on 10 % of the data completes on Colab/Kaggle (link the run in `docs/RUNS.md`).

## Blocks
#12, #8 (registry), #19
