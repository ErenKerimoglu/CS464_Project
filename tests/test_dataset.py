"""Dataset / dataloader contract (#9). Synthetic data, plus a check against the real frozen split file."""
from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch

from galaxy.constants import SPLIT_FILE
from galaxy.data.dataset import GalaxyDataset, build_dataloaders, load_split


@pytest.fixture
def split_df(synthetic_meta):
    """6 images per class → 4 train / 1 val / 1 test each (too few for sklearn's stratified split)."""
    df = synthetic_meta.copy()
    pos = df.groupby("label").cumcount()
    df["split"] = np.select([pos < 4, pos == 4], ["train", "val"], "test")
    df["frac_rank"] = np.where(df.split == "train", pos / 4, np.nan)
    return df


@pytest.fixture
def npy(tmp_path, synthetic_images):
    path = tmp_path / "images_256.npy"
    np.save(path, synthetic_images)
    return path


def test_item_contract(split_df, synthetic_images):
    ds = GalaxyDataset("val", images=synthetic_images, split_df=split_df)
    x, y, oi = ds[0]
    assert x.shape == (3, 256, 256) and x.dtype == torch.uint8
    assert isinstance(y, int) and isinstance(oi, int)
    assert np.array_equal(x.permute(1, 2, 0).numpy(), synthetic_images[oi])
    assert y == int(split_df.set_index("orig_index").loc[oi, "label"])


def test_memmap_and_in_memory_give_same_items(split_df, npy):
    lazy = GalaxyDataset("train", split_df=split_df, images_path=npy)
    ram = GalaxyDataset("train", split_df=split_df, images_path=npy, in_memory=True)
    for i in (0, 5, len(lazy) - 1):
        assert torch.equal(lazy[i][0], ram[i][0]) and lazy[i][1:] == ram[i][1:]


def test_memmap_is_not_pickled(split_df, npy):
    ds = GalaxyDataset("train", split_df=split_df, images_path=npy)
    _ = ds[0]                                   # opens the memmap
    blob = pickle.dumps(ds)
    assert len(blob) < 100_000                  # path only, not the image array
    assert torch.equal(pickle.loads(blob)[0][0], ds[0][0])


def test_fraction_subsets(split_df, synthetic_images):
    full = GalaxyDataset("train", images=synthetic_images, split_df=split_df)
    half = GalaxyDataset("train", fraction=0.5, images=synthetic_images, split_df=split_df)
    assert set(half.orig_index) <= set(full.orig_index) and 0 < len(half) < len(full)
    with pytest.raises(ValueError):
        GalaxyDataset("test", fraction=0.5, images=synthetic_images, split_df=split_df)


def test_dataloaders(split_df, synthetic_images):
    ds = {s: GalaxyDataset(s, images=synthetic_images, split_df=split_df) for s in ("train", "val", "test")}
    cfg = {"data": {"batch_size": 8, "num_workers": 0}}
    loaders = build_dataloaders(cfg, datasets=ds)
    x, y, oi = next(iter(loaders["train"]))
    assert x.shape == (8, 3, 256, 256) and x.dtype == torch.uint8 and y.shape == (8,) and oi.shape == (8,)
    assert len(loaders["val"].dataset) == len(ds["val"]) and loaders["val"].batch_size == 16
    # same seed → same shuffling order
    a = next(iter(build_dataloaders(cfg, datasets=ds, seed=3)["train"]))[2]
    b = next(iter(build_dataloaders(cfg, datasets=ds, seed=3)["train"]))[2]
    assert torch.equal(a, b)


@pytest.mark.skipif(not SPLIT_FILE.exists(), reason="frozen split not present")
def test_real_split_file_sizes():
    df = load_split()
    sizes = df.split.value_counts().to_dict()
    assert sizes == {"train": 12332, "val": 2643, "test": 2643}
    assert len(GalaxyDataset("train", fraction=0.1, split_df=df, images=np.empty((0,)))) == pytest.approx(1233, abs=10)
