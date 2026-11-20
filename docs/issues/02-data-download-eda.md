---
id: 2
title: "[Data] Download Galaxy10 DECaLS + exploratory data analysis"
owner: eren
labels: [area:data, type:feature, priority:critical]
milestone: "M0 - Foundations"
week: W1
depends_on: []
---
## Goal
Get the dataset reproducibly and record the facts that later design decisions depend on.

## Context
Plan §3.1. Source: astroNN `Galaxy10_DECals.h5` (Zenodo record 10845026, about 2.5 GB).
Keys: `images (17736,256,256,3) uint8`, `ans`, `ra`, `dec`, `redshift`, `pxscale`.

## Tasks
- [ ] `galaxy/data/download.py`: `download_galaxy10(dest=DATA_DIR/"raw")`, resumable (HTTP Range), progress bar, skips the download if the file exists and the size/MD5 match. Store the expected MD5 in `galaxy/constants.py`.
- [ ] `load_h5(path) -> dict[str, np.ndarray]`. Images are loaded lazily (h5py dataset), the other keys eagerly.
- [ ] `notebooks/00_eda.ipynb` covering:
  - [ ] class counts (bar chart) and a check against the counts in `constants.CLASS_COUNTS`
  - [ ] sample grid: 8 random images per class
  - [ ] per-band pixel mean/std/histograms
  - [ ] **galaxy-extent check:** what fraction of total flux (background-subtracted r-band) lies inside the central 180×180 crop? Report the distribution and how many galaxies have < 90 % of their flux inside. This confirms or changes the 180 px crop in plan §3.1.
  - [ ] redshift distribution per class, `pxscale` sanity check
  - [ ] a first look at near-duplicates (pairwise RA/Dec separation histogram), as input for #3
- [ ] Save the key figures to `results/figures/eda/` (`class_counts.png`, `samples_grid.png`, `crop_flux_fraction.png`).

## Deliverables
`galaxy/data/download.py`, `notebooks/00_eda.ipynb` (committed **with** outputs), figures, and a short summary comment on this issue.

## Acceptance criteria
- A fresh clone + `python scripts/prepare_data.py --download-only` downloads and verifies the file.
- The EDA summary comment states the class counts, band stats, crop-coverage result and an estimated number of duplicates.
- If the 180 px crop loses too much flux for many galaxies, propose an alternative in the issue **before** #3 freezes the band stats.

## Blocks
#3, #7, #9, #11
