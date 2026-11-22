"""Parallel feature extraction with parquet caching. Issue #7 (owner: Yunus)."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from galaxy.constants import FEATURES_DIR


def load_raw_images(indices):
    """Load rows from the processed image array without importing the PyTorch dataset."""
    from galaxy.constants import PROCESSED_IMAGES

    indices = np.asarray(indices, dtype=np.int64)
    images = np.load(PROCESSED_IMAGES, mmap_mode="r")
    return np.asarray(images[indices])


def extract_all(name: str, fn: Callable, indices=None, n_jobs: int = -1, force: bool = False,
                image_transform: Callable | None = None) -> pd.DataFrame:
    """Apply ``fn`` to every image (optionally after ``image_transform``, e.g. rotation/corruption).

    Cache the result as ``data/features/<name>_v1.parquet`` with an ``orig_index`` column.
    """
    cache_path = FEATURES_DIR / f"{name}_v1.parquet"
    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
        if "orig_index" not in cached.columns:
            raise ValueError(f"feature cache is missing orig_index: {cache_path}")
        return cached

    if indices is None:
        from galaxy.constants import SPLIT_FILE
        indices = pd.read_csv(SPLIT_FILE)["orig_index"].to_numpy(dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    images = load_raw_images(indices)

    def process_one(image):
        transformed = image_transform(image) if image_transform is not None else image
        result = fn(transformed)
        if isinstance(result, dict):
            row = dict(result)
        elif isinstance(result, np.ndarray):
            row = {f"feature_{i:04d}": float(value) for i, value in enumerate(result.reshape(-1))}
        elif isinstance(result, (list, tuple)):
            row = {f"feature_{i:04d}": float(value) for i, value in enumerate(np.asarray(result).reshape(-1))}
        else:
            row = {"feature": float(result)}
        return row

    rows = Parallel(n_jobs=n_jobs, batch_size="auto")(
        delayed(process_one)(image) for image in images
    )
    frame = pd.DataFrame(rows)
    frame.insert(0, "orig_index", indices)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame
