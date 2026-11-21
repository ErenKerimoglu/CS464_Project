"""Fixed stratified 70/15/15 split + train-only band statistics. Issue #3 (owner: Eren).

Output files (committed to git, **never regenerated**; a change needs a new version and a team decision):
    data/splits/split_v1.csv        orig_index,label,ra,dec,redshift,split,frac_rank
    data/splits/band_stats_v1.json  {"mean": [g, r, z], "std": [g, r, z], "crop": 180, ...}
    data/splits/removed_v1.csv      images dropped by deduplication and why
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from galaxy.constants import BAND_STATS_FILE, CROP_SIZE, NUM_CLASSES, SPLIT_FILE, SPLIT_FRACTIONS, SPLIT_SEED

SPLIT_COLUMNS = ["orig_index", "label", "ra", "dec", "redshift", "split", "frac_rank"]


def make_split(meta: pd.DataFrame, fractions=SPLIT_FRACTIONS, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Return ``meta`` (already de-duplicated) plus a stratified ``split`` column and a ``frac_rank`` column.

    ``frac_rank`` ∈ [0, 1) is a random rank within each class of the **train** split (NaN for val/test), so
    ``fraction=f`` ⇔ ``frac_rank < f``. This keeps the subsets nested (10 % ⊂ 25 % ⊂ 50 %) and stratified.
    """
    train_frac, val_frac, test_frac = fractions
    assert abs(train_frac + val_frac + test_frac - 1) < 1e-9
    idx = meta.index.to_numpy()
    y = meta["label"].to_numpy()
    rest, test = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    train, val = train_test_split(rest, test_size=val_frac / (train_frac + val_frac),
                                  stratify=meta.loc[rest, "label"].to_numpy(), random_state=seed)
    out = meta.copy()
    out["split"] = ""
    out.loc[train, "split"] = "train"
    out.loc[val, "split"] = "val"
    out.loc[test, "split"] = "test"

    rng = np.random.default_rng(seed)
    out["frac_rank"] = np.nan
    for c in range(NUM_CLASSES):
        members = out.index[(out["split"] == "train") & (out["label"] == c)].to_numpy()
        if len(members):
            order = rng.permutation(len(members))
            out.loc[members, "frac_rank"] = order / len(members)
    out = out.sort_values("orig_index").reset_index(drop=True)
    return out[[c for c in SPLIT_COLUMNS if c in out.columns]]


def subset_mask(split_df: pd.DataFrame, split: str, fraction: float = 1.0) -> np.ndarray:
    """Boolean mask selecting ``split`` (and, for train, the nested ``fraction`` subset)."""
    mask = (split_df["split"] == split).to_numpy()
    if fraction < 1.0:
        if split != "train":
            raise ValueError("fraction < 1 is only defined for the train split")
        mask = mask & (split_df["frac_rank"] < fraction).to_numpy()
    return mask


def split_summary(split_df: pd.DataFrame) -> pd.DataFrame:
    """Per class: counts per split and the class share (%) in each split and overall."""
    counts = pd.crosstab(split_df["label"], split_df["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
    share = (counts / counts.sum() * 100).round(2).add_suffix(" %")
    overall = (split_df["label"].value_counts(normalize=True).sort_index() * 100).round(2).rename("all %")
    return pd.concat([counts, share, overall], axis=1)


def _crop_bounds(size: int, crop: int) -> slice:
    lo = (size - crop) // 2
    return slice(lo, lo + crop)


def compute_band_stats(image_chunks, crop: int = CROP_SIZE) -> dict:
    """Per-band mean/std in [0,1] scale over the center ``crop`` of the given images.

    ``image_chunks`` is an iterable of uint8 arrays (B,H,W,3); pass only **train** images. Accumulated in float64,
    so the result does not depend on chunking.
    """
    total = np.zeros(3)
    total_sq = np.zeros(3)
    n = 0
    for block in image_chunks:
        block = np.asarray(block)
        s = _crop_bounds(block.shape[1], crop)
        x = block[:, s, s, :].astype(np.float64) / 255.0
        total += x.sum(axis=(0, 1, 2))
        total_sq += (x * x).sum(axis=(0, 1, 2))
        n += x.shape[0] * x.shape[1] * x.shape[2]
    if n == 0:
        raise ValueError("No images given")
    mean = total / n
    std = np.sqrt(np.maximum(total_sq / n - mean**2, 0))
    return {"mean": mean.round(6).tolist(), "std": std.round(6).tolist(), "crop": crop, "scale": "0-1",
            "bands": ["g", "r", "z"], "n_pixels_per_band": int(n)}


def save_split(split_df: pd.DataFrame, path: Path = SPLIT_FILE, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists. The split is frozen; pass force=True only after a team decision.")
    path.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(path, index=False, float_format="%.8f")
    return path


def save_band_stats(stats: dict, path: Path = BAND_STATS_FILE, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists; pass force=True only after a team decision.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path
