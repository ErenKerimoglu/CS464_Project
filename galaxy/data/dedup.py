"""Duplicate removal by sky coordinates. Issue #3 (owner: Eren). Protocol: PROJECT_PLAN §3.1."""
from __future__ import annotations

import numpy as np
import pandas as pd

from galaxy.constants import DEDUP_RADIUS_ARCSEC


def find_duplicates(ra: np.ndarray, dec: np.ndarray, radius_arcsec: float = DEDUP_RADIUS_ARCSEC) -> list[list[int]]:
    """Groups (connected components) of indices whose positions lie within ``radius_arcsec`` of each other."""
    raise NotImplementedError("Issue #3")


def resolve_duplicates(groups: list[list[int]], labels: np.ndarray, n: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply the policy: same label → keep the lowest index; conflicting labels → drop all copies.

    Returns ``(kept_indices, removed)``, where ``removed`` has the columns ``orig_index, group_id, reason``.
    """
    raise NotImplementedError("Issue #3")
