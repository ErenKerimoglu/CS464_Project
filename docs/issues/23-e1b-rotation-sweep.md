---
id: 23
title: "[E1b] Rotation sweep (accuracy vs test rotation angle) + orientation synthesis (headline figure)"
owner: eren
labels: [area:orientation, type:experiment, priority:critical]
milestone: "M3 - Analysis"
week: W5
depends_on: [13, 17, 19, 24]
---
## Goal
The headline result of the project: how does accuracy change as test images are rotated, for every model family? And the final answer to RQ1.

## Context
Plan §1 (headline deliverable), §3.4 (24 angles; rotate the 256 image before cropping). Classical curves come from #24.

## Tasks
- [ ] `galaxy/evaluation/rotation.py`: `make_rotation_fn(angle)` (uses `rotate_batch` from #9) for the `predict()` hook. `scripts/evaluate.py --rotate DEG`, and `scripts/run_rotation_sweep.py --runs <glob> --angles 0:360:15` (also with `--tta8`).
- [ ] Sweep all deep runs: model_family (all models, 3 seeds), `e1_orientation/*_none`, escnn (both policies), and the TTA variants of ResNet-50.
- [ ] **Headline figure** `results/figures/e1/accuracy_vs_angle.pdf`: polar or Cartesian plot, x = angle, y = test accuracy, mean ± std band, one line per model/variant (consistent `MODEL_COLORS`), including the classical curves from #24. Mark multiples of 90°, where interpolation-free rotations happen.
- [ ] Summary metrics per model: mean accuracy over angles, worst-case accuracy, and the **orientation sensitivity** = std over angles. Put them in a table.
- [ ] Synthesis table answering RQ1: Δ macro-F1 from (a) rotflip augmentation, (b) TTA, (c) equivariant architecture, compared with (d) the gap between model families (e.g. SmallCNN → ConvNeXt-T, CNN → ViT).
- [ ] Write `report/sections/e1_orientation.md` (finish the E1a draft) with the conclusion of the main research question.

## Deliverables
`rotation.py`, sweep script, result files, headline figure, two tables, E1 report section.

## Acceptance criteria
- Every model family is in the figure, with 3 seeds where applicable.
- At 0° the sweep reproduces the standard test metrics exactly (asserted in a test or check).
