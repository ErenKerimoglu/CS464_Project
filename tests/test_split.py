"""Tests for deduplication, split and band statistics (#3). Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxy.constants import NUM_CLASSES
from galaxy.data.dedup import (
    ARCSEC,
    angular_sep_arcsec,
    find_duplicates,
    find_pairs,
    nearest_neighbor_sep_arcsec,
    resolve_duplicates,
    sensitivity_table,
)
from galaxy.data.split import compute_band_stats, make_split, save_split, split_summary, subset_mask


@pytest.fixture
def catalog() -> pd.DataFrame:
    """2000 random galaxies + injected duplicates: 20 same-label twins, 5 conflicting twins, 1 triplet chain."""
    rng = np.random.default_rng(1)
    n = 2000
    df = pd.DataFrame({
        "ra": rng.uniform(0, 360, n),
        "dec": np.degrees(np.arcsin(rng.uniform(-0.5, 0.9, n))),
        "label": rng.integers(0, NUM_CLASSES, n),
        "redshift": rng.uniform(0.01, 0.2, n),
    })
    extra = []
    for k in range(20):  # same label, 0.3" away
        extra.append((df.ra[k] + 0.3 / 3600 / np.cos(np.radians(df.dec[k])), df.dec[k], df.label[k]))
    for k in range(20, 25):  # different label, 0.5" away
        extra.append((df.ra[k], df.dec[k] + 0.5 / 3600, (df.label[k] + 1) % NUM_CLASSES))
    # chain: 30 → +0.8" → +1.6" (the ends are 1.6" apart but linked through the middle)
    extra.append((df.ra[30], df.dec[30] + 0.8 / 3600, df.label[30]))
    extra.append((df.ra[30], df.dec[30] + 1.6 / 3600, df.label[30]))
    df = pd.concat([df, pd.DataFrame(extra, columns=["ra", "dec", "label"])], ignore_index=True)
    df["redshift"] = df["redshift"].fillna(0.1)
    df["orig_index"] = np.arange(len(df))
    return df


def test_angular_sep_known_values():
    assert angular_sep_arcsec(10, 20, 10, 20 + 1 / 3600) == pytest.approx(1.0, rel=1e-6)
    assert angular_sep_arcsec(0, 0, 90, 0) == pytest.approx(90 * 3600, rel=1e-9)
    assert angular_sep_arcsec(359.9999, 0, 0.0001, 0) == pytest.approx(0.72, rel=1e-4)  # RA wrap-around
    assert ARCSEC == pytest.approx(np.pi / 648000)


def test_find_pairs_matches_brute_force(catalog):
    sub = catalog.iloc[:300]
    sep = angular_sep_arcsec(sub.ra.values[:, None], sub.dec.values[:, None], sub.ra.values[None], sub.dec.values[None])
    i, j = np.where(np.triu(sep < 1.0, k=1))
    brute = set(zip(i.tolist(), j.tolist(), strict=True))
    assert {tuple(p) for p in find_pairs(sub.ra.values, sub.dec.values, 1.0).tolist()} == brute


def test_duplicate_groups_and_policy(catalog):
    groups = find_duplicates(catalog.ra.values, catalog.dec.values, 1.0)
    assert len(groups) == 26  # 20 twins + 5 conflicts + 1 chain
    assert sorted(len(g) for g in groups).count(3) == 1
    kept, removed = resolve_duplicates(groups, catalog.label.values, len(catalog))
    assert (removed.reason == "duplicate").sum() == 20 + 2           # twins drop 1 copy, chain drops 2
    assert (removed.reason == "label_conflict").sum() == 10          # both copies of 5 conflicting pairs
    assert len(kept) == len(catalog) - 32
    assert set(kept).isdisjoint(removed.orig_index)
    dup = removed[removed.reason == "duplicate"]
    assert (dup.kept_index < dup.orig_index).all()                   # the lowest index is kept
    assert nearest_neighbor_sep_arcsec(catalog.ra.values, catalog.dec.values)[0] == pytest.approx(0.3, rel=1e-3)


def test_sensitivity_monotonic(catalog):
    tab = sensitivity_table(catalog.ra.values, catalog.dec.values, catalog.label.values)
    assert tab.images_in_groups.is_monotonic_increasing


def test_split_stratified_disjoint_reproducible(catalog):
    groups = find_duplicates(catalog.ra.values, catalog.dec.values)
    kept, _ = resolve_duplicates(groups, catalog.label.values, len(catalog))
    meta = catalog.loc[kept]
    s1, s2 = make_split(meta), make_split(meta)
    pd.testing.assert_frame_equal(s1, s2)                            # reproducible
    assert len(s1) == len(meta) and s1.orig_index.is_unique and set(s1.split) == {"train", "val", "test"}
    frac = s1.split.value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.70, abs=0.01) and frac["test"] == pytest.approx(0.15, abs=0.01)
    summary = split_summary(s1)
    for s in ("train", "val", "test"):                                # stratified per class
        assert (summary[f"{s} %"] - summary["all %"]).abs().max() < 1.0
    # no remaining near-duplicate pair crosses splits
    pairs = find_pairs(s1.ra.values, s1.dec.values, 1.0)
    assert len(pairs) == 0


def test_frac_rank_nested_and_stratified(catalog):
    s = make_split(catalog)
    assert s.loc[s.split != "train", "frac_rank"].isna().all()
    m10, m25, m50 = (subset_mask(s, "train", f) for f in (0.10, 0.25, 0.50))
    assert (m10 <= m25).all() and (m25 <= m50).all()                 # nested
    train = s[s.split == "train"]
    got = s[m25].label.value_counts().sort_index()
    want = (train.label.value_counts().sort_index() * 0.25)
    assert (got - want).abs().max() <= 1                             # ~25 % of every class
    with pytest.raises(ValueError):
        subset_mask(s, "val", 0.5)


def test_band_stats_chunking_invariant(synthetic_images):
    imgs = synthetic_images
    whole = compute_band_stats([imgs])
    chunked = compute_band_stats(imgs[i:i + 7] for i in range(0, len(imgs), 7))
    assert np.allclose(whole["mean"], chunked["mean"]) and np.allclose(whole["std"], chunked["std"])
    lo = (256 - 180) // 2
    ref = imgs[:, lo:lo + 180, lo:lo + 180].reshape(-1, 3) / 255.0
    assert np.allclose(whole["mean"], ref.mean(0), atol=1e-6) and np.allclose(whole["std"], ref.std(0), atol=1e-6)


def test_split_file_is_frozen(tmp_path, catalog):
    path = tmp_path / "split.csv"
    s = make_split(catalog)
    save_split(s, path)
    with pytest.raises(FileExistsError):
        save_split(s, path)
