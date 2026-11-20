---
id: 7
title: "[Classical] HOG + color histogram feature extraction with caching"
owner: yunus
labels: [area:classical, type:feature, priority:high]
milestone: "M0 - Foundations"
week: W1
depends_on: [3]
---
## Goal
Fast, cached hand-crafted image features for the classical models.

## Context
Plan §3.3. Input is the **same 180×180 center crop** the deep models use, as raw [0,1] pixels.
Until the processed cache exists (#3), develop on the raw `.h5` or on synthetic images.

## Tasks
- [ ] `galaxy/features/common.py`: `center_crop_np(img, 180)`, `to_float01`, and `rotate_np(img256, angle)` (rotate the 256 image, then crop). The orientation study (#24) needs the same geometry as `rotate_batch` in #9.
- [ ] `galaxy/features/hog_color.py`
  - [ ] `hog_features(img)`: grayscale (band mean) → resize 128 → `skimage.feature.hog(orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2))`. Record the feature dimension.
  - [ ] `color_features(img, mask=None)`: 32-bin histograms per band (normalized), plus mean/std/p10/p50/p90 of the g−r and r−z "color" images. Use the galaxy mask once #11 provides it, otherwise a central circle of radius 40 px.
- [ ] `galaxy/features/cache.py`: `extract_all(split, fn, n_jobs=-1) -> DataFrame` (joblib in chunks), saved to `data/features/<name>_v1.parquet` with an `orig_index` column. Re-running skips existing caches unless `--force`.
- [ ] `scripts/extract_features.py --features hog color [--rotate DEG] [--corruption ...]`. The rotate/corruption flags write separate caches, used later by #22 and #24.
- [ ] Timing: report extraction time for all 17.7 k images.
- [ ] `tests/test_features.py`: shapes, determinism, and HOG of a 90°-rotated image ≠ the original (sanity check that HOG is orientation-sensitive).

## Deliverables
`common.py`, `hog_color.py`, `cache.py`, `scripts/extract_features.py`, tests, cached parquet files (on Drive/Kaggle, **not** in git).

## Acceptance criteria
- Features exist for all train/val/test images of `split_v1`.
- A quick sanity model (LR on HOG + color, default params) reaches clearly above chance on val. Post the number in this issue.

## Blocks
#14, #22, #24
