"""Tests for HOG and color feature extraction."""
from __future__ import annotations

import numpy as np

from galaxy.features.common import center_crop_np, rotate_np, to_float01
from galaxy.features.hog_color import color_features, hog_features


def test_common_geometry_and_float_conversion(synthetic_images):
    image = synthetic_images[0]
    assert to_float01(image).dtype == np.float32
    assert to_float01(image).min() >= 0 and to_float01(image).max() <= 1
    assert center_crop_np(image).shape == (180, 180, 3)
    assert rotate_np(image, 90).shape == (180, 180, 3)


def test_hog_shape_determinism_and_rotation(synthetic_images):
    image = synthetic_images[0]
    features = hog_features(image)
    assert features.shape == (1764,)
    assert np.array_equal(features, hog_features(image))
    assert not np.allclose(features, hog_features(rotate_np(image, 90)))


def test_color_shape_and_normalized_histograms(synthetic_images):
    features = color_features(synthetic_images[0])
    assert len(features) == 106
    for band in ("g", "r", "z"):
        hist = [features[f"{band}_hist_{i:02d}"] for i in range(32)]
        assert sum(hist) == np.float64(1.0) or np.isclose(sum(hist), 1.0)
    assert {"gr_mean", "gr_std", "gr_p10", "gr_p50", "gr_p90",
            "rz_mean", "rz_std", "rz_p10", "rz_p50", "rz_p90"} <= features.keys()
