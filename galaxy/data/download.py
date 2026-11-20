"""Download and read Galaxy10 DECaLS. Issue #2 (owner: Eren)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from galaxy.constants import DATASET_FILENAME, DATASET_URL, RAW_DIR


def download_galaxy10(dest: Path = RAW_DIR, url: str = DATASET_URL, force: bool = False) -> Path:
    """Download ``Galaxy10_DECals.h5`` (resumable). Verify the size/MD5 and return the local path.

    Skips the download when a verified file already exists, unless ``force``.
    """
    raise NotImplementedError("Issue #2")


def load_h5(path: Path = RAW_DIR / DATASET_FILENAME) -> dict[str, np.ndarray]:
    """Return ``{"images": h5py.Dataset (lazy), "ans", "ra", "dec", "redshift", "pxscale": np.ndarray}``."""
    raise NotImplementedError("Issue #2")
