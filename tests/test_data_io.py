"""Tests for download/IO and EDA helpers (#2). Synthetic data only, no network."""
from __future__ import annotations

import hashlib

import h5py
import numpy as np
import pytest

from galaxy.data.download import download_file, iter_image_chunks, load_metadata, md5sum, open_images, verify_file
from galaxy.data.eda import class_count_table, crop_flux_fraction, pixel_stats


def test_download_file_and_verify(tmp_path):
    src = tmp_path / "src.bin"
    payload = np.random.default_rng(0).bytes(3_000_000)
    src.write_bytes(payload)
    dest = download_file(src.as_uri(), tmp_path / "out" / "file.bin", progress=False)
    assert dest.read_bytes() == payload and not (tmp_path / "out" / "file.bin.part").exists()
    md5 = hashlib.md5(payload).hexdigest()
    assert md5sum(dest) == md5
    assert verify_file(dest, size=len(payload), md5=md5)
    assert not verify_file(dest, size=len(payload) + 1, md5=md5)
    assert not verify_file(dest, size=len(payload), md5="0" * 32)


@pytest.fixture
def tiny_h5(tmp_path, synthetic_images, synthetic_meta):
    path = tmp_path / "g10.h5"
    with h5py.File(path, "w") as f:
        f["images"] = synthetic_images
        f["ans"] = synthetic_meta.label.values.astype(np.uint8)
        for k in ("ra", "dec", "redshift"):
            f[k] = synthetic_meta[k].values
        f["pxscale"] = np.full(len(synthetic_meta), 0.262)
    return path


def test_load_metadata_and_images(tiny_h5, synthetic_images, synthetic_meta):
    meta = load_metadata(tiny_h5)
    assert list(meta.columns) == ["orig_index", "label", "ra", "dec", "redshift", "pxscale"]
    assert (meta.label.values == synthetic_meta.label.values).all()
    with open_images(tiny_h5) as images:
        got = {int(i): b for sel, block in iter_image_chunks(images, np.array([50, 3, 4, 5, 20]), chunk=2)
               for i, b in zip(sel, block, strict=True)}
    assert sorted(got) == [3, 4, 5, 20, 50]
    assert all((got[i] == synthetic_images[i]).all() for i in got)


def test_crop_flux_fraction():
    imgs = np.zeros((2, 256, 256, 3), np.uint8)
    imgs[0, 120:136, 120:136] = 200          # compact central source → fully inside the crop
    imgs[1, 120:136, 120:136] = 200
    imgs[1, 5:21, 120:136] = 200             # equal-flux source outside the 180 crop
    frac = crop_flux_fraction(imgs)
    assert frac[0] == pytest.approx(1.0) and frac[1] == pytest.approx(0.5, abs=0.01)


def test_tables(synthetic_images, synthetic_meta):
    t = class_count_table(synthetic_meta.label.values)
    assert t["count"].sum() == len(synthetic_meta) and len(t) == 10
    assert list(pixel_stats(synthetic_images).index) == ["g", "r", "z"]
