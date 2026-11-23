"""Parallel feature extraction with parquet caching. Issue #7 (owner: Yunus)."""
from __future__ import annotations

from collections.abc import Callable
from logging import getLogger
from multiprocessing import TimeoutError

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from galaxy.constants import FEATURES_DIR

logger = getLogger(__name__)


def load_raw_images(indices):
    """Load rows from the processed image array without importing the PyTorch dataset."""
    from galaxy.constants import PROCESSED_IMAGES

    indices = np.asarray(indices, dtype=np.int64)
    images = np.load(PROCESSED_IMAGES, mmap_mode="r")
    return np.asarray(images[indices])


def extract_all(name: str, fn: Callable, indices=None, n_jobs: int = -1, force: bool = False,
                image_transform: Callable | None = None, timeout_seconds: float | None = None,
                timeout_result: dict | None = None) -> pd.DataFrame:
    """Apply ``fn`` to every image (optionally after ``image_transform``, e.g. rotation/corruption).

    Cache the result as ``data/features/<name>_v1.parquet`` with an ``orig_index`` column.
    When a timeout is configured, each image is retried in isolation after its parallel chunk times out.
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

    if timeout_seconds is None:
        rows = Parallel(n_jobs=n_jobs, batch_size="auto")(
            delayed(process_one)(image) for image in images
        )
    else:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        rows = []
        timeout_jobs = 2 if n_jobs == 1 else n_jobs
        chunk_size = 128
        for start in range(0, len(images), chunk_size):
            image_chunk = images[start:start + chunk_size]
            index_chunk = indices[start:start + chunk_size]
            try:
                chunk_rows = Parallel(n_jobs=timeout_jobs, batch_size=1, timeout=timeout_seconds)(
                    delayed(process_one)(image) for image in image_chunk
                )
                rows.extend(chunk_rows)
            except TimeoutError:
                logger.warning(
                    "Feature chunk timed out after %.1f seconds; retrying %d images individually",
                    timeout_seconds,
                    len(image_chunk),
                )
                for image, orig_index in zip(image_chunk, index_chunk, strict=True):
                    try:
                        row = Parallel(n_jobs=timeout_jobs, batch_size=1, timeout=timeout_seconds)(
                            [delayed(process_one)(image)]
                        )[0]
                    except TimeoutError:
                        logger.warning(
                            "Feature extraction timed out after %.1f seconds for orig_index=%d",
                            timeout_seconds,
                            orig_index,
                        )
                        if timeout_result is None:
                            raise
                        row = dict(timeout_result)
                    rows.append(row)

    frame = pd.DataFrame(rows)
    frame.insert(0, "orig_index", indices)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame
