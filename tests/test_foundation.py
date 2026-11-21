"""Tests for the shared, already-implemented foundation (constants, config, seeding, fixtures)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from galaxy import constants as C
from galaxy.utils import git_commit, seed_everything
from galaxy.utils.config import apply_overrides, deep_merge, load_config


def test_constants_consistent():
    assert C.NUM_CLASSES == 10 == len(C.CLASS_SHORT)
    assert sum(C.CLASS_COUNTS) == 17736
    assert abs(sum(C.SPLIT_FRACTIONS) - 1) < 1e-9
    assert C.CROP_SIZE < C.IMAGE_SIZE / np.sqrt(2)  # any rotation leaves no empty corners in the crop
    assert len(C.ROTATION_ANGLES) == 24


def test_deep_merge_and_overrides():
    merged = deep_merge({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"b": 5}})
    assert merged == {"a": {"b": 5, "c": 2}, "d": 1}
    cfg = apply_overrides(merged, ["a.c=1e-3", "e.f=true", "g=[1,2]"])
    assert cfg["a"]["c"] == pytest.approx(1e-3) and cfg["e"]["f"] is True and cfg["g"] == [1, 2]
    with pytest.raises(ValueError):
        apply_overrides({}, ["no_equals_sign"])


@pytest.mark.parametrize("name", ["small_cnn", "resnet50", "convnext_tiny", "vit_small", "escnn_d4"])
def test_model_configs_inherit_base(name):
    cfg = load_config(C.CONFIGS_DIR / "models" / f"{name}.yaml", overrides=["train.lr=0.5"])
    assert cfg["model"]["name"] == name
    assert cfg["preprocess"]["crop"] == C.CROP_SIZE          # inherited from base.yaml
    assert cfg["train"]["lr"] == 0.5                          # override wins


def test_seed_everything_reproducible():
    seed_everything(3)
    a = (random.random(), np.random.rand())
    seed_everything(3)
    assert a == (random.random(), np.random.rand())


def test_git_commit_returns_string():
    assert isinstance(git_commit(), str) and git_commit()


def test_synthetic_fixtures(synthetic_images, synthetic_meta):
    assert synthetic_images.shape == (60, 256, 256, 3) and synthetic_images.dtype == np.uint8
    assert synthetic_meta["label"].value_counts().nunique() == 1  # balanced


def test_processed_dir_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("GALAXY_PROCESSED_DIR", str(tmp_path / "x"))
    assert C._find_processed_dir() == tmp_path / "x"          # explicit override wins
    monkeypatch.delenv("GALAXY_PROCESSED_DIR")
    assert C._find_processed_dir().name == "processed"        # falls back to DATA_DIR/processed
