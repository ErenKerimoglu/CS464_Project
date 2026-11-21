"""Project-wide constants. Changing anything here needs a team decision (see docs/PROJECT_PLAN.md §3)."""
from __future__ import annotations

import os
from pathlib import Path

# ----------------------------------------------------------------------------- paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("GALAXY_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
SPLITS_DIR = REPO_ROOT / "data" / "splits"          # always inside the repo (committed)
CKPT_DIR = Path(os.environ.get("GALAXY_CKPT_DIR", REPO_ROOT / "checkpoints"))
RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
CONFIGS_DIR = REPO_ROOT / "configs"

# ----------------------------------------------------------------------------- dataset
# Zenodo record 10845026 (astroNN Galaxy10 DECaLS); size and MD5 from the Zenodo API.
DATASET_URL = "https://zenodo.org/api/records/10845026/files/Galaxy10_DECals.h5/content"
DATASET_FILENAME = "Galaxy10_DECals.h5"
DATASET_SIZE = 2_735_267_419
DATASET_MD5 = "c6b7b4db82b3a5d63d6a7e3e5249b51c"
# astroNN's own de-duplicated variant: used only as a cross-check of our dedup (#3), never for training.
NODUP_FILENAME = "Galaxy10_DECals_NoDuplicated.h5"
NODUP_URL = "https://zenodo.org/api/records/10845026/files/Galaxy10_DECals_NoDuplicated.h5/content"

IMAGE_SIZE = 256          # raw image side (pixels)
PIXEL_SCALE_ARCSEC = 0.262
CROP_SIZE = 180           # < 256/sqrt(2): rotation by any angle leaves no empty corners (plan §3.1)
PRETRAINED_INPUT_SIZE = 224
BANDS = ("g", "r", "z")

CLASS_NAMES = (
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
)
CLASS_SHORT = ("Dist", "Merg", "RndSm", "InBtw", "Cigar", "Barred", "UnbTight", "UnbLoose", "EdgeNoB", "EdgeB")
CLASS_COUNTS = (1081, 1853, 2645, 2027, 334, 2043, 1829, 2628, 1423, 1873)  # before dedup
NUM_CLASSES = len(CLASS_NAMES)

# ----------------------------------------------------------------------------- protocol
SPLIT_VERSION = "v1"
SPLIT_SEED = 464
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
DEDUP_RADIUS_ARCSEC = 1.0
SEEDS = (0, 1, 2)
DATA_FRACTIONS = (0.10, 0.25, 0.50, 1.00)
ROTATION_ANGLES = tuple(range(0, 360, 15))

SPLIT_FILE = SPLITS_DIR / f"split_{SPLIT_VERSION}.csv"
BAND_STATS_FILE = SPLITS_DIR / f"band_stats_{SPLIT_VERSION}.json"
REMOVED_FILE = SPLITS_DIR / f"removed_{SPLIT_VERSION}.csv"
DEDUP_REPORT_FILE = SPLITS_DIR / f"dedup_report_{SPLIT_VERSION}.json"
PROCESSED_IMAGES = PROCESSED_DIR / "images_256.npy"
PROCESSED_META = PROCESSED_DIR / "meta.csv"

assert len(CLASS_COUNTS) == NUM_CLASSES and sum(CLASS_COUNTS) == 17736
