"""Fixed stratified 70/15/15 split + train-only band statistics. Issue #3 (owner: Eren).

Output files (committed to git, never regenerated):
    data/splits/split_v1.csv        orig_index,label,ra,dec,redshift,split,frac_rank
    data/splits/band_stats_v1.json  {"mean": [g, r, z], "std": [g, r, z], "crop": 180}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from galaxy.constants import SPLIT_FRACTIONS, SPLIT_SEED


def make_split(meta: pd.DataFrame, fractions=SPLIT_FRACTIONS, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Add a ``split`` column (train/val/test), stratified by ``label``, and a nested-subset ``frac_rank`` column.

    ``frac_rank`` ∈ [0, 1) is a random rank within each class of the train split, so
    ``fraction=f`` ⇔ ``frac_rank < f``. This makes the subsets nested: 10 % ⊂ 25 % ⊂ 50 %.
    """
    raise NotImplementedError("Issue #3")


def compute_band_stats(images_uint8: np.ndarray, crop: int = 180) -> dict:
    """Per-band mean/std in [0,1] scale over the center crop of the **train** images."""
    raise NotImplementedError("Issue #3")
