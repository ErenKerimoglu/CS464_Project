"""Dataset + dataloaders. Issue #9 (owner: Eren). Contract: PROJECT_PLAN §4.1.

The dataset yields **raw uint8** images (3×256×256). All geometry, augmentation and normalization happen on the GPU
in :class:`galaxy.data.transforms.GalaxyPreprocess`, so every model is preprocessed identically.

Images come from ``data/processed/images_256.npy`` (written by ``scripts/prepare_data.py``; rows = ``orig_index``).
Set ``GALAXY_DATA_DIR`` to point at it (e.g. a Kaggle input folder).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from galaxy.constants import BAND_STATS_FILE, PROCESSED_IMAGES, SPLIT_FILE
from galaxy.data.split import subset_mask
from galaxy.utils.seed import worker_init_fn

Split = Literal["train", "val", "test"]


@lru_cache(maxsize=1)
def _split_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_split(path: Path = SPLIT_FILE) -> pd.DataFrame:
    """The frozen ``split_v1.csv`` (a copy, safe to modify)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - it is committed in git; pull main.")
    return _split_cached(str(path)).copy()


def load_band_stats(path: Path = BAND_STATS_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_raw_images(indices: np.ndarray | None = None, path: Path = PROCESSED_IMAGES) -> np.ndarray:
    """uint8 (N,256,256,3). Memory-mapped (no RAM cost) when ``indices`` is None, otherwise an in-RAM copy."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found - run scripts/prepare_data.py or set GALAXY_DATA_DIR.")
    arr = np.load(path, mmap_mode="r")
    return arr if indices is None else np.asarray(arr[np.asarray(indices)])


class GalaxyDataset(Dataset):
    """``__getitem__(i) -> (uint8 Tensor[3,256,256], label: int, orig_index: int)``.

    Args:
        split: "train" | "val" | "test".
        fraction: train-set fraction for E3 (nested stratified subset via ``frac_rank``). Must be 1.0 for val/test.
        in_memory: load this split's images into RAM once (fast; train ≈ 2.4 GB). Otherwise the memmap is opened
            lazily in every worker (a memmap must not be pickled into worker processes: that would copy it).
        images: optional in-RAM array indexed by ``orig_index`` (used by tests and notebooks).
    """

    def __init__(self, split: Split, fraction: float = 1.0, in_memory: bool = False,
                 images: np.ndarray | None = None, split_df: pd.DataFrame | None = None,
                 images_path: Path = PROCESSED_IMAGES):
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split '{split}'")
        df = load_split() if split_df is None else split_df
        rows = df[subset_mask(df, split, fraction)]
        self.split, self.fraction = split, fraction
        self.orig_index = rows["orig_index"].to_numpy(np.int64)
        self.labels = rows["label"].to_numpy(np.int64)
        self._path = Path(images_path)
        self._images = None      # array indexed by position in this dataset (in-memory) or by orig_index (memmap)
        self._by_position = False
        if images is not None:
            self._images = images
        elif in_memory:
            self._images = load_raw_images(self.orig_index, self._path)
            self._by_position = True

    def __len__(self) -> int:
        return len(self.orig_index)

    def _array(self) -> np.ndarray:
        if self._images is None:  # lazily, once per worker process
            self._images = load_raw_images(None, self._path)
        return self._images

    def __getitem__(self, i: int):
        oi = int(self.orig_index[i])
        img = self._array()[i if self._by_position else oi]
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return x, int(self.labels[i]), oi

    def __getstate__(self):
        state = self.__dict__.copy()
        if not self._by_position and isinstance(self._images, np.memmap):
            state["_images"] = None  # reopen in the worker instead of copying the memmap
        return state

    def class_counts(self, num_classes: int = 10) -> np.ndarray:
        return np.bincount(self.labels, minlength=num_classes)


def build_dataloaders(cfg: dict, sampler=None, splits=("train", "val", "test"), seed: int | None = None,
                      datasets: dict | None = None) -> dict[str, DataLoader]:
    """``{"train": DataLoader, "val": ..., "test": ...}`` from ``cfg["data"]``.

    ``sampler`` (from ``galaxy.training.samplers``) replaces shuffling for train. ``datasets`` lets callers pass
    prebuilt datasets (tests). Yields ``(uint8 images, labels, orig_index)`` batches.
    """
    d = cfg.get("data", {})
    seed = cfg.get("seed", 0) if seed is None else seed
    workers = int(d.get("num_workers", 4))
    loaders = {}
    for split in splits:
        ds = (datasets or {}).get(split) or GalaxyDataset(
            split, fraction=d.get("fraction", 1.0) if split == "train" else 1.0, in_memory=d.get("in_memory", False))
        train = split == "train"
        g = torch.Generator().manual_seed(seed)
        loaders[split] = DataLoader(
            ds,
            batch_size=int(d.get("batch_size", 64)) * (1 if train else int(d.get("eval_batch_multiplier", 2))),
            shuffle=train and sampler is None,
            sampler=sampler if train else None,
            drop_last=train,
            num_workers=workers,
            pin_memory=bool(d.get("pin_memory", True)) and torch.cuda.is_available(),
            persistent_workers=workers > 0,
            prefetch_factor=4 if workers > 0 else None,
            worker_init_fn=worker_init_fn if workers > 0 else None,
            generator=g,
        )
    return loaders
