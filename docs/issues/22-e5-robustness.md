---
id: 22
title: "[E5] Robustness: accuracy under noise, blur and downsampling (weaker telescopes)"
owner: yunus
labels: [area:experiments, type:experiment, priority:high]
milestone: "M3 - Analysis"
week: W4-W5
depends_on: [5, 9, 14, 19]
---
## Goal
Measure how each model family degrades under realistic image degradations (RQ5). Evaluation only, no new training.

## Context
Plan §6 (E5). Corruptions apply to the raw [0,1] 256×256 image **before** crop and standardization, through the `transform_fn` hook in `predict()` (#5).

## Tasks
- [ ] `galaxy/evaluation/corruptions.py` (torch, batched, deterministic given a seed):
  - [ ] `gaussian_noise(x, sigma)`, sigma ∈ {0.02, 0.05, 0.10, 0.20}
  - [ ] `gaussian_blur(x, sigma_px)` (PSF broadening), sigma_px ∈ {1, 2, 3, 4}
  - [ ] `downsample(x, factor)` (area downsample, then bilinear upsample back to 256), factor ∈ {2, 4, 8}
  - [ ] Registry `CORRUPTIONS = {"noise": ..., "blur": ..., "downsample": ...}` with severity lists. A NumPy twin (or a torch→numpy wrapper) for the classical feature extractors.
- [ ] `scripts/evaluate.py --corruption NAME --severity S` writes `eval_test_<name>_s<S>.json`. A loop script `scripts/run_robustness.py --runs <glob>` evaluates all severities.
- [ ] Evaluate: every `model_family` run (all seeds), escnn, `e1_orientation/*_none` (does augmentation help robustness?), and the classical models. For classical models, re-extract features on corrupted test images with `extract_features.py --corruption`.
- [ ] Figure: accuracy vs. severity, one panel per corruption, one line per model (mean ± std). Table: clean acc, mean corrupted acc, relative drop.
- [ ] Figure `results/figures/e5/corruption_examples.png`: one galaxy at every severity.
- [ ] Draft `report/sections/e5_robustness.md`.

## Deliverables
`corruptions.py`, robustness scripts, result files, figures, table, report paragraph.

## Acceptance criteria
- Every model family appears in the figure. Severity 0 equals the clean test result (sanity check in a test).
