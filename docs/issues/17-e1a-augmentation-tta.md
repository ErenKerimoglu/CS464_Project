---
id: 17
title: "[E1a] Orientation: no-aug vs rotation/flip augmentation + 8-view test-time augmentation"
owner: eren
labels: [area:orientation, type:experiment, priority:critical]
milestone: "M2 - Core Experiments"
week: W3-W4
depends_on: [9, 15]
---
## Goal
Measure how much invariance we gain through **data** (augmentation) and **inference** (TTA), with everything else held fixed.

## Context
Plan §1 RQ1, §3.2 (policies), §3.4 (TTA definition), §6 (E1a: ResNet-50 and ConvNeXt-T, `none` ×3 seeds; `rotflip` runs reused from #19).

## Tasks
- [ ] `galaxy/evaluation/tta.py`: `d4_views(x) -> list[Tensor]` (8 views on the **cropped** batch: rot90 k=0..3 × optional h-flip), `predict_tta8(model, loader, preprocess)` averages softmax. Plug it into `scripts/evaluate.py --tta8`, and write `eval_test_tta8.json`.
- [ ] Training runs (tuned configs from #15), logged in `docs/RUNS.md`:
  - [ ] `e1_orientation/resnet50_none` seeds 0/1/2
  - [ ] `e1_orientation/convnext_tiny_none` seeds 0/1/2
  - [ ] *(optional, if budget allows)* `e1_orientation/resnet50_d4` seeds 0/1/2 (exact D4 aug vs continuous rotations)
- [ ] TTA-8 evaluation on **every** deep run that exists by end of week 4: family runs (#19), the `none` runs, escnn (#13). For escnn TTA should change nothing, which is a good sanity check.
- [ ] Table: rows = model, cols = {none, none+TTA, rotflip, rotflip+TTA, escnn}, values = test acc / macro-F1, mean ± std.
- [ ] Significance (#16): rotflip vs none, and rotflip+TTA vs rotflip, for ResNet-50.
- [ ] Draft the E1a results paragraph in `report/sections/e1_orientation.md`.

## Deliverables
`tta.py`, `e1_orientation` result files, table, report paragraph.

## Acceptance criteria
- TTA-8 results exist for every deep model in the main table.
- A test checks that `d4_views` produces 8 distinct views and that averaging over them gives the same prediction for any D4-rotated input.

## Blocks
#23
