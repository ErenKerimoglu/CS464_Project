"""HOG and color features. Issue #7 (owner: Yunus). Spec: PROJECT_PLAN §3.3."""
from __future__ import annotations

import numpy as np
from skimage.feature import hog
from skimage.transform import resize

from galaxy.features.common import to_float01


def hog_features(img: np.ndarray) -> np.ndarray:
    """Grayscale (band mean) → resize 128 → HOG(9 orientations, 16 px cells, 2×2 blocks)."""
    arr = to_float01(np.asarray(img))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("HOG expects an H×W×3 image")
    gray = arr.mean(axis=-1)
    gray = resize(gray, (128, 128), order=1, mode="reflect", anti_aliasing=True)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float32, copy=False)


def color_features(img: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Per-band 32-bin histograms + statistics of the g−r and r−z color images inside ``mask``."""
    arr = to_float01(np.asarray(img))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("color features expect an H×W×3 image")

    h, w = arr.shape[:2]
    if mask is None:
        yy, xx = np.mgrid[:h, :w]
        mask = (xx - (w - 1) / 2.0) ** 2 + (yy - (h - 1) / 2.0) ** 2 <= 40 ** 2
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (h, w):
        raise ValueError("mask shape must match the image")
    if not mask.any():
        raise ValueError("mask contains no pixels")

    features: dict[str, float] = {}
    band_names = ("g", "r", "z")
    for band, name in enumerate(band_names):
        counts, _ = np.histogram(arr[..., band][mask], bins=32, range=(0.0, 1.0))
        counts = counts.astype(np.float64)
        counts /= counts.sum()
        for i, count in enumerate(counts):
            features[f"{name}_hist_{i:02d}"] = float(count)

    for name, values in (("gr", arr[..., 0] - arr[..., 1]), ("rz", arr[..., 1] - arr[..., 2])):
        values = values[mask].astype(np.float64)
        percentiles = np.percentile(values, (10, 50, 90))
        features[f"{name}_mean"] = float(values.mean())
        features[f"{name}_std"] = float(values.std())
        features[f"{name}_p10"] = float(percentiles[0])
        features[f"{name}_p50"] = float(percentiles[1])
        features[f"{name}_p90"] = float(percentiles[2])
    return features
