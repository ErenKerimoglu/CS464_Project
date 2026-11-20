"""NumPy geometry helpers matching the GPU pipeline. Issue #7 (owner: Yunus)."""
from __future__ import annotations

import numpy as np


def to_float01(img_uint8: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Issue #7")


def center_crop_np(img: np.ndarray, size: int = 180) -> np.ndarray:
    """Center crop of an (H,W,C) or (H,W) image."""
    raise NotImplementedError("Issue #7")


def rotate_np(img256: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate the full 256 image (bilinear, zero fill), then crop.

    Must match ``galaxy.data.transforms.rotate_batch``.
    """
    raise NotImplementedError("Issue #7")
