---
id: 24
title: "[Classical] Orientation sweep & learning curves for classical models"
owner: yunus
labels: [area:classical, area:orientation, type:experiment, priority:high]
milestone: "M3 - Analysis"
week: W4-W5
depends_on: [14]
---
## Goal
Add the classical family to the orientation study (E1) and the data-efficiency study (E3). Hypothesis: HOG features are orientation-sensitive, while morphology statistics (C, A, Gini, M20) are almost rotation-invariant by construction.

## Context
Plan §3.3. Rotation geometry must match the deep models: rotate the 256 image → crop 180 (`rotate_np` from #7, consistent with `rotate_batch` in #9).

## Tasks
- [ ] Rotation sweep: for angles 0:360:15, extract features on rotated **test** images (`extract_features.py --rotate DEG`; morphology only if runtime allows, otherwise every 45°) and evaluate the trained classical models (no retraining). Write `eval_test_rot<DDD>.json` in the shared schema.
- [ ] Feature-group sensitivity: repeat with XGBoost on HOG-only, morph-only and all. Which features cause orientation sensitivity?
- [ ] *(Interesting extra)* Train XGBoost/SVM with **rotation-augmented** training features (each train image + 3 random rotations) and sweep again. This is the classical counterpart of `rotflip`.
- [ ] Learning curves: train the tuned XGBoost and SVM with `--fraction {0.1,0.25,0.5,1.0}` (same nested subsets), 3 seeds for XGBoost. Hand the results to #20 for the joint figure.
- [ ] Short table + paragraph in `report/sections/classical.md` (classical method, ablation from #14, orientation & data-efficiency findings).

## Deliverables
Rotated-feature caches (not in git), result files, sensitivity table, report section.

## Acceptance criteria
- Classical curves are ready for the headline figure in #23 by **mid week 5**.
- Learning-curve results are in the shared schema under `e3_data_eff/xgboost_f*`, `svm_rbf_f*`.

## Blocks
#23
