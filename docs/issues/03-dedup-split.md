---
id: 3
title: "[Data] Deduplicate by sky coordinates + fixed stratified 70/15/15 split (split_v1)"
owner: eren
labels: [area:data, type:feature, priority:critical]
milestone: "M0 - Foundations"
week: W1
depends_on: [2]
---
## Goal
Produce **the** split that every experiment uses, free of duplicate leakage. **This is the top priority of week 1.** Everyone is blocked on it.

## Context
Plan §3.1 (deduplication + split rows).

## Tasks
- [ ] `galaxy/data/dedup.py`
  - [ ] `find_duplicates(ra, dec, radius_arcsec=1.0)` using `astropy.coordinates.SkyCoord.search_around_sky`, returning duplicate groups (connected components).
  - [ ] Sensitivity table: number of groups at 0.5/1/2/5″.
  - [ ] Policy: in a same-label group keep the lowest index. If labels **conflict**, drop all copies. Return `kept_indices` and a `removed.csv` (`orig_index, group_id, reason`).
  - [ ] Visual check: save `results/figures/eda/duplicate_pairs.png` with about 12 random pairs side by side.
- [ ] `galaxy/data/split.py`
  - [ ] Stratified 70/15/15 with `train_test_split` twice, `random_state=464` (`constants.SPLIT_SEED`).
  - [ ] Write `data/splits/split_v1.csv` with columns `orig_index,label,ra,dec,redshift,split`.
  - [ ] Compute **train-only** per-band mean/std on the 180×180 center crop, in [0,1] scale, and write `data/splits/band_stats_v1.json` (`{"mean":[3], "std":[3], "crop":180}`).
  - [ ] Nested stratified fraction subsets for E3: add a column `frac_rank` (random rank within each class of the train split, seed 0), so `fraction=f` means "rank < f·n_class". This makes 10 % ⊂ 25 % ⊂ 50 % automatically.
- [ ] `scripts/prepare_data.py`: download → dedup → split → write the processed cache `data/processed/images_256.npy` (uint8, `np.lib.format.open_memmap`, rows ordered by `orig_index`) + `meta.csv`.
- [ ] `tests/test_split.py` (runs on synthetic data): stratification within ±1 sample per class, no index in two splits, no pair within 1″ spans two splits, reproducible with the same seed.

## Deliverables
`dedup.py`, `split.py`, `prepare_data.py`, **committed** `data/splits/split_v1.csv`, `data/splits/band_stats_v1.json`, `removed.csv`, tests.

## Acceptance criteria
- `split_v1.csv` is on `main` by the **end of week 1** and is never regenerated afterwards (later changes need a `v2` and a team decision).
- Class proportions per split match the full dataset within 0.5 percentage points (table in the PR description).
- The PR description lists the number of removed duplicates and conflicts.

## Blocks
#7, #9, #11, and through them every experiment.
