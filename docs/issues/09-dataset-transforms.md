---
id: 9
title: "[Data] GalaxyDataset, GPU preprocessing & augmentation policies"
owner: eren
labels: [area:data, type:feature, priority:critical]
milestone: "M1 - Pipelines & Baselines"
week: W2
depends_on: [3]
---
## Goal
A fast, shared input pipeline. Geometry, normalization and augmentation are implemented **once**, so every comparison is fair.

## Context
Plan §3.1 (geometry: rotate 256 → crop 180 → resize), §3.2 (policies), §4.1 (exact contract).
Augmentation runs **on the GPU in batches**, so Kaggle's 2–4 CPU cores never limit training.

## Tasks
- [ ] `galaxy/data/dataset.py`
  - [ ] `load_split()`, `load_band_stats()`, `load_raw_images(indices)` (memmap of `images_256.npy`)
  - [ ] `GalaxyDataset(split, fraction=1.0, fraction_seed=0)` returns `(uint8 tensor [3,256,256], label, orig_index)`. `fraction` uses the nested `frac_rank` column from #3.
  - [ ] `build_dataloaders(cfg, sampler=None)`: pinned memory, persistent workers, `drop_last` on train, deterministic worker seeding
- [ ] `galaxy/data/transforms.py`
  - [ ] `rotate_batch(x, angles_deg)`: per-sample rotation with `affine_grid` + `grid_sample` (bilinear, zero padding). Exact `torch.rot90` path when every angle is a multiple of 90°.
  - [ ] `center_crop(x, size)`, `resize(x, size)` (antialiased bilinear)
  - [ ] `GalaxyPreprocess(nn.Module)(policy, out_size, crop=180, band_stats)`: uint8 → float [0,1] → (train only: augmentation) → crop → resize → per-band standardize.
  - [ ] Policies: `none`, `rotflip` (angle ~ U[0,360), h-flip and v-flip with p=0.5 each), `d4` (random rot90 k ∈ {0..3} × flip).
- [ ] Add `data:` and `preprocess:` sections to `configs/base.yaml`.
- [ ] `tests/test_transforms.py`: `rotate_batch` by 90° equals `torch.rot90`. Rotation by 360° ≈ identity. After rotating by 45° and cropping to 180, no zero-padded pixels remain. Output stats ≈ N(0,1) on synthetic data with known stats. `none` is deterministic.
- [ ] Throughput benchmark on Kaggle T4: images/s for dataloader + preprocess (target > 1500 img/s). Post the number here.
- [ ] Save `results/figures/eda/augmentation_examples.png`: one galaxy under 8 random `rotflip` draws.

## Deliverables
`dataset.py`, `transforms.py`, config sections, tests, benchmark comment.

## Acceptance criteria
- Merged by the **middle of week 2**. #4's smoke run works with the real data.
- The tests prove the geometry claims above.

## Blocks
#13, #17, #18, #19, #20, #21, #22, #23
