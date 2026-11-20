---
id: 8
title: "[Equivariance] escnn rotation-equivariant CNN architecture + equivariance tests"
owner: eftelya
labels: [area:models, area:orientation, type:feature, priority:high]
milestone: "M0 - Foundations"
week: W1
depends_on: []
---
## Goal
The equivariant arm of the main experiment (E1): a CNN whose prediction is invariant to rotations **by design**.

## Context
Plan §1 RQ1, §4.2. Reference: Pandya et al. 2023 (arXiv:2311.01500), escnn docs/tutorials.
The model must follow the registry contract from #6. You can write against the contract before #6 is merged.

## Tasks
- [ ] **Day 1–2:** check that `escnn` installs on **Colab and Kaggle** (and Python 3.10–3.12). Pin a working version in `pyproject.toml` (`[project.optional-dependencies].equivariant`) and write down any extra steps in the README. If it fails, comment here right away. Fallback: an exact C4 group.
- [ ] `galaxy/models/equivariant.py` `EquivariantCNN`:
  - [ ] Group space configurable: `flipRot2dOnR2(N=4)` (**D4**, default, registered as `escnn_d4`) and `rot2dOnR2(N=8)` (**C8**, `escnn_c8`).
  - [ ] Input: 3 trivial representations (g, r, z). A `MaskModule` for a circular receptive field.
  - [ ] About 5–6 `R2Conv` blocks with regular representations, `InnerBatchNorm`, `ReLU`, `PointwiseAvgPoolAntialiased` downsampling. Widths chosen so the **parameter count is in the SmallCNN–ResNet-50 range**; report it.
  - [ ] `GroupPooling` → invariant features → global average pool → MLP head.
  - [ ] Implement `forward_features`, `input_size` (default 129, odd so 90° rotations are exact on the pixel grid; resize the 180 crop to it) and `gradcam_target_layer` (or document why Grad-CAM uses a different layer).
- [ ] `configs/models/escnn_d4.yaml`, `configs/models/escnn_c8.yaml`.
- [ ] `tests/test_equivariance.py` (`importorskip("escnn")`):
  - [ ] logits(rot90(x)) ≈ logits(x) and logits(flip(x)) ≈ logits(x) (D4), atol about 1e-4, model in eval mode
  - [ ] report the invariance error at 45° (only approximately invariant; not asserted, but printed)
- [ ] Speed: forward + backward time for batch 32 on GPU, compared with SmallCNN. Post it in this issue.

## Deliverables
`equivariant.py`, two configs, the equivariance test, pinned dependency, README note.

## Acceptance criteria
- The invariance tests pass for exact group elements.
- `build_model("escnn_d4")` passes the generic model tests from #6.
- The parameter count and speed are reported.

## Blocks
#13, #20 (escnn learning curve), #23
