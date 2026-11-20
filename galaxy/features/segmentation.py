"""Background estimation and central-galaxy segmentation (photutils). Issue #11 (owner: Yunus).

Also used by the Grad-CAM "on-galaxy" score in #25, so keep this API stable.
"""
from __future__ import annotations

import numpy as np


def estimate_background(img_band: np.ndarray) -> tuple[float, float]:
    """Sigma-clipped (median, std) of a border annulus."""
    raise NotImplementedError("Issue #11")


def segment_central_source(img_band: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """Return ``(segmap, central_label, all_sources_segmap)``."""
    raise NotImplementedError("Issue #11")


def galaxy_mask(img: np.ndarray) -> np.ndarray:
    """Boolean mask of the central galaxy (r-band segmentation)."""
    raise NotImplementedError("Issue #11")
