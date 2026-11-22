"""Tests for classical feature matrix assembly and preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from galaxy.features import pipeline


def _write_feature_cache(directory, name, frame):
    frame.to_parquet(directory / f"{name}_v1.parquet", index=False)


def test_load_feature_matrix_aligns_caches_and_applies_train_fraction(tmp_path, monkeypatch):
    split = pd.DataFrame({
        "orig_index": [30, 10, 20, 40],
        "label": [3, 1, 2, 4],
        "split": ["train", "train", "val", "test"],
        "frac_rank": [0.2, 0.8, np.nan, np.nan],
    })
    split_path = tmp_path / "split.csv"
    split.to_csv(split_path, index=False)
    monkeypatch.setattr(pipeline, "SPLIT_FILE", split_path)
    monkeypatch.setattr(pipeline, "FEATURES_DIR", tmp_path)
    cache_rows = [10, 40, 30, 20]
    _write_feature_cache(tmp_path, "hog", pd.DataFrame({
        "orig_index": cache_rows,
        "feature_0000": [100.0, 400.0, 300.0, 200.0],
    }))
    _write_feature_cache(tmp_path, "color", pd.DataFrame({
        "orig_index": cache_rows,
        "g_hist_00": [1.0, 4.0, 3.0, 2.0],
    }))

    X, y, indices = pipeline.load_feature_matrix("train", ("hog", "color"))
    assert indices.tolist() == [30, 10]
    assert y.tolist() == [3, 1]
    assert X["feature_0000"].tolist() == [300.0, 100.0]
    assert X["g_hist_00"].tolist() == [3.0, 1.0]

    X_fraction, y_fraction, fraction_indices = pipeline.load_feature_matrix(
        "train", ("hog", "color"), fraction=0.5,
    )
    assert fraction_indices.tolist() == [30]
    assert y_fraction.tolist() == [3]
    assert X_fraction["feature_0000"].tolist() == [300.0]


def test_load_feature_matrix_fails_when_cache_is_incomplete(tmp_path, monkeypatch):
    split_path = tmp_path / "split.csv"
    pd.DataFrame({
        "orig_index": [1, 2],
        "label": [0, 1],
        "split": ["train", "train"],
        "frac_rank": [0.0, 0.5],
    }).to_csv(split_path, index=False)
    monkeypatch.setattr(pipeline, "SPLIT_FILE", split_path)
    monkeypatch.setattr(pipeline, "FEATURES_DIR", tmp_path)
    _write_feature_cache(tmp_path, "hog", pd.DataFrame({
        "orig_index": [1],
        "feature_0000": [1.0],
    }))

    with pytest.raises(ValueError, match="missing 1 selected samples"):
        pipeline.load_feature_matrix("train", ("hog",))


def test_build_preprocessor_pca_and_imputes_morphology():
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        **{f"feature_{i:04d}": rng.normal(size=20) for i in range(5)},
        "g_hist_00": rng.normal(size=20),
        "concentration": np.r_[np.nan, rng.normal(size=19)],
        "morph_failed": np.zeros(20),
    })
    preprocessor = pipeline.build_preprocessor(pipeline.feature_columns_by_group(X.columns))
    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == len(X)
    assert np.isfinite(transformed).all()
    pca = preprocessor.named_transformers_["hog"].named_steps["pca"]
    assert 1 <= pca.n_components_ <= 5
    assert pca.explained_variance_ratio_.sum() >= 0.95
