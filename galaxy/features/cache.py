"""Parallel feature extraction with parquet caching. Issue #7 (owner: Yunus)."""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def extract_all(name: str, fn: Callable, indices=None, n_jobs: int = -1, force: bool = False,
                image_transform: Callable | None = None) -> pd.DataFrame:
    """Apply ``fn`` to every image (optionally after ``image_transform``, e.g. rotation/corruption).

    Cache the result as ``data/features/<name>_v1.parquet`` with an ``orig_index`` column.
    """
    raise NotImplementedError("Issue #7")
