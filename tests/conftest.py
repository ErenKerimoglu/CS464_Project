"""Synthetic fixtures so everyone can develop and test against the contracts before real data exists.

Use them in your tests: ``def test_x(synthetic_images, synthetic_meta): ...``. Never use real data in CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxy.constants import IMAGE_SIZE, NUM_CLASSES


def _blob(rng: np.random.Generator, size: int = IMAGE_SIZE) -> np.ndarray:
    """A noisy elliptical 'galaxy' at the image centre with a random orientation, uint8 (H,W,3)."""
    yy, xx = np.mgrid[:size, :size] - size / 2
    theta, q, r = rng.uniform(0, np.pi), rng.uniform(0.3, 1.0), rng.uniform(10, 30)
    xr, yr = xx * np.cos(theta) + yy * np.sin(theta), -xx * np.sin(theta) + yy * np.cos(theta)
    profile = np.exp(-np.sqrt(xr**2 + (yr / q) ** 2) / r)
    img = np.stack([profile * s for s in rng.uniform(0.6, 1.0, 3)], axis=-1) * 220
    img += rng.normal(8, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture(scope="session")
def synthetic_images(rng) -> np.ndarray:
    """(60, 256, 256, 3) uint8: 6 images per class."""
    return np.stack([_blob(rng) for _ in range(6 * NUM_CLASSES)])


@pytest.fixture(scope="session")
def synthetic_meta(synthetic_images, rng) -> pd.DataFrame:
    """Metadata shaped like data/processed/meta.csv (labels, sky coordinates) for the synthetic images."""
    n = len(synthetic_images)
    return pd.DataFrame({
        "orig_index": np.arange(n),
        "label": np.repeat(np.arange(NUM_CLASSES), n // NUM_CLASSES),
        "ra": rng.uniform(0, 360, n),
        "dec": rng.uniform(-30, 60, n),
        "redshift": rng.uniform(0.01, 0.2, n),
    })
