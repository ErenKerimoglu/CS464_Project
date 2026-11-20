---
id: 11
title: "[Classical] Galaxy segmentation + statmorph morphology statistics"
owner: yunus
labels: [area:classical, type:feature, priority:high]
milestone: "M1 - Pipelines & Baselines"
week: W2
depends_on: [7]
---
## Goal
Physically meaningful, mostly rotation-invariant features (C, A, S, Gini, M20, …), plus a reusable galaxy segmentation.

## Context
Plan §3.3. Images are uint8 RGB renderings of g/r/z, not calibrated flux, so we subtract the background and use r-band (channel 1) intensities.
`segmentation.py` is **also used by #25** (Grad-CAM "on-galaxy" score). Keep its API clean.

## Tasks
- [ ] `galaxy/features/segmentation.py`
  - [ ] `estimate_background(img_band)`: sigma-clipped median/std from a border annulus
  - [ ] `segment_central_source(img_band) -> (segmap, central_label, all_sources_segmap)` using `photutils.segmentation.detect_sources` + `deblend_sources` (threshold about 1.5σ, npixels about 10). The central source is the one containing (or closest to) the image center.
  - [ ] `galaxy_mask(img) -> bool[H,W]` helper
- [ ] `galaxy/features/morphology.py`
  - [ ] `morph_features(img) -> dict` with `statmorph.source_morphology(image, segmap, gain=1.0)` on the background-subtracted r-band 180 crop. Keep: concentration, asymmetry, smoothness, gini, m20, gini_m20_bulge, gini_m20_merger, ellipticity_asymmetry, elongation_asymmetry, r20, r80, flag. Skip the expensive Sérsic fit if the installed statmorph version supports it (check `source_morphology` kwargs).
  - [ ] Robustness: try/except per galaxy, timeout per galaxy, NaN on failure, plus a `morph_failed` column.
- [ ] Extract for all images with `scripts/extract_features.py --features morph` (joblib parallel). Report the runtime and failure rate per class.
- [ ] Figure `results/figures/classical/morph_by_class.png`: Gini–M20 scatter colored by class, and C vs A.
- [ ] `tests/test_morphology.py`: a synthetic Sérsic-like blob gives finite C/Gini/M20, and rotating the blob by 90° changes C and Gini by < 5 % (rotation-invariance sanity check).

## Deliverables
`segmentation.py`, `morphology.py`, cached `morph_v1.parquet`, figure, tests.

## Acceptance criteria
- Failure rate < 5 % (or documented with the reason).
- The Gini–M20 plot shows visible class separation (e.g. smooth vs. spiral). Comment on it in the issue.

## Blocks
#14, #24, #25 (segmentation)
