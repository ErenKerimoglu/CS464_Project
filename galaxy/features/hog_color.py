"""HOG and color features. Issue #7 (owner: Yunus). Spec: PROJECT_PLAN §3.3."""
from __future__ import annotations

import numpy as np


def hog_features(img: np.ndarray) -> np.ndarray:
    """Grayscale (band mean) → resize 128 → HOG(9 orientations, 16 px cells, 2×2 blocks)."""
    raise NotImplementedError("Issue #7")


def color_features(img: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Per-band 32-bin histograms + statistics of the g−r and r−z color images inside ``mask``."""
    raise NotImplementedError("Issue #7")
