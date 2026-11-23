"""Synthetic tests for galaxy segmentation and statmorph features."""
from __future__ import annotations

import numpy as np

from galaxy.features.morphology import morph_features
from galaxy.features.segmentation import estimate_background, galaxy_mask, segment_central_source


def _sersic_blob() -> np.ndarray:
    yy, xx = np.mgrid[:180, :180]
    x, y = xx - 89.5, yy - 89.5
    radius = np.sqrt(x**2 + (y / 0.72) ** 2)
    profile = np.exp(-3.672 * ((radius / 12.0) ** (1 / 2.2) - 1))
    profile /= profile.max()
    return np.stack((profile, profile * 0.9, profile * 0.8), axis=-1).astype(np.float32)


def test_background_and_central_segmentation():
    image = _sersic_blob()
    median, std = estimate_background(image[..., 1])
    assert np.isfinite(median) and np.isfinite(std) and std >= 0

    segmap, central_label, all_sources = segment_central_source(image[..., 1])
    assert segmap.shape == all_sources.shape == (180, 180)
    assert central_label > 0
    assert np.array_equal(segmap > 0, all_sources == central_label)
    assert galaxy_mask(image).dtype == np.bool_
    assert galaxy_mask(image).shape == (180, 180)


def test_morph_features_finite_and_rotation_invariant():
    image = _sersic_blob()
    original = morph_features(image)
    rotated = morph_features(np.rot90(image))

    assert original["morph_failed"] == rotated["morph_failed"] == 0
    for key in ("concentration", "gini", "m20"):
        assert np.isfinite(original[key])
    for key in ("concentration", "gini"):
        scale = max(abs(original[key]), 1e-8)
        assert abs(original[key] - rotated[key]) / scale < 0.05


def test_morph_features_returns_nan_on_no_detectable_source():
    result = morph_features(np.zeros((180, 180, 3), dtype=np.float32))
    assert result["morph_failed"] == 1
    assert np.isnan(result["gini"])
