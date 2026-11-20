"""Dataset + dataloaders. Issue #9 (owner: Eren). Contract: PROJECT_PLAN §4.1.

The dataset yields **raw uint8** images (3×256×256). All geometry, augmentation and normalization happen on the GPU
in :class:`galaxy.data.transforms.GalaxyPreprocess`, so every model is preprocessed identically.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Split = Literal["train", "val", "test"]


def load_split() -> pd.DataFrame:
    """Read ``data/splits/split_v1.csv``."""
    raise NotImplementedError("Issue #9")


def load_band_stats() -> dict:
    """Read ``data/splits/band_stats_v1.json``."""
    raise NotImplementedError("Issue #9")


def load_raw_images(indices: np.ndarray | None = None) -> np.ndarray:
    """uint8 (N,256,256,3) from the memory-mapped ``data/processed/images_256.npy`` (rows = orig_index)."""
    raise NotImplementedError("Issue #9")


class GalaxyDataset:  # (torch.utils.data.Dataset)
    """``__getitem__(i) -> (uint8 Tensor[3,256,256], label: int, orig_index: int)``.

    Args:
        split: "train" | "val" | "test".
        fraction: train-set fraction (nested stratified subset via ``frac_rank``). Must be 1.0 for val/test.
    """

    def __init__(self, split: Split, fraction: float = 1.0):
        raise NotImplementedError("Issue #9")


def build_dataloaders(cfg: dict, sampler=None) -> dict:
    """Return ``{"train": DataLoader, "val": DataLoader, "test": DataLoader}`` configured from ``cfg["data"]``."""
    raise NotImplementedError("Issue #9")
