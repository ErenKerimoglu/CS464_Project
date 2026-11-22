"""Background estimation and central-galaxy segmentation (photutils). Issue #11 (owner: Yunus).

Also used by the Grad-CAM "on-galaxy" score in #25, so keep this API stable.
"""
from __future__ import annotations

import warnings

import numpy as np


def _image_band(img_band: np.ndarray) -> np.ndarray:
    data = np.asarray(img_band, dtype=np.float64)
    if data.ndim != 2 or min(data.shape) < 2:
        raise ValueError("img_band must be a non-empty 2D image")
    if not np.isfinite(data).any():
        raise ValueError("img_band contains no finite pixels")
    return data


def _border_annulus_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    width = max(1, int(round(min(h, w) * 0.1)))
    mask = np.ones((h, w), dtype=bool)
    if h > 2 * width and w > 2 * width:
        mask[width:-width, width:-width] = False
    return mask


def estimate_background(img_band: np.ndarray) -> tuple[float, float]:
    """Return sigma-clipped median and standard deviation from the image border annulus."""
    from astropy.stats import sigma_clipped_stats

    data = _image_band(img_band)
    border = _border_annulus_mask(data.shape)
    _, median, std = sigma_clipped_stats(data, mask=~border | ~np.isfinite(data), sigma=3.0, maxiters=5)
    median, std = float(median), float(std)
    if not np.isfinite(median) or not np.isfinite(std):
        raise ValueError("could not estimate a finite background from the image border")
    return median, std


def segment_central_source(img_band: np.ndarray) -> tuple[np.ndarray, int, np.ndarray]:
    """Return the central-source-only map, its label, and the full deblended segmentation map."""
    from photutils.segmentation import deblend_sources, detect_sources
    from photutils.utils.exceptions import NoDetectionsWarning

    data = _image_band(img_band)
    background, background_std = estimate_background(data)
    signal = np.nan_to_num(data - background, nan=0.0, posinf=0.0, neginf=0.0)
    threshold = max(1.5 * background_std, np.finfo(np.float64).eps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NoDetectionsWarning)
        detected = detect_sources(signal, threshold, n_pixels=10)
    if detected is None:
        empty = np.zeros(data.shape, dtype=np.int32)
        return empty.copy(), 0, empty

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*progress_bar.*deprecated.*",
        )
        segmentation = deblend_sources(
            signal,
            detected,
            n_pixels=10,
            n_levels=32,
            contrast=0.001,
            progress_bar=False,
        )
    all_sources = np.asarray(segmentation.data, dtype=np.int32)
    labels = np.unique(all_sources)
    labels = labels[labels > 0]
    if labels.size == 0:
        empty = np.zeros(data.shape, dtype=np.int32)
        return empty.copy(), 0, empty

    center_y, center_x = (np.array(data.shape) - 1) / 2.0
    center_pixel = (int(round(center_y)), int(round(center_x)))
    central_label = int(all_sources[center_pixel])
    if central_label == 0:
        best_distance = np.inf
        for label in labels:
            yy, xx = np.nonzero(all_sources == label)
            distance = np.min((yy - center_y) ** 2 + (xx - center_x) ** 2)
            if distance < best_distance:
                central_label, best_distance = int(label), float(distance)

    central_source = np.where(all_sources == central_label, central_label, 0).astype(np.int32)
    return central_source, central_label, all_sources


def galaxy_mask(img: np.ndarray) -> np.ndarray:
    """Return a boolean central-galaxy mask from an H×W image or an RGB H×W×3 image."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        r_band = arr
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        r_band = arr[..., 1]
    elif arr.ndim == 3 and arr.shape[0] == 3:
        r_band = arr[1]
    else:
        raise ValueError("img must be an H×W r-band image or an RGB H×W×3 / 3×H×W image")
    segmap, central_label, _ = segment_central_source(r_band)
    return (segmap == central_label) if central_label else np.zeros(r_band.shape, dtype=bool)
