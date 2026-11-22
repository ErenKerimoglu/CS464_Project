"""Tests for feature-cache timeout handling."""
from __future__ import annotations

import time

import numpy as np

from galaxy.features import cache


def test_extract_all_timeout_uses_failure_row(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "FEATURES_DIR", tmp_path)
    monkeypatch.setattr(
        cache,
        "load_raw_images",
        lambda indices: np.zeros((len(indices), 2, 2), dtype=np.uint8),
    )

    def slow_feature(_image):
        time.sleep(4)
        return {"value": 1.0, "morph_failed": 0.0}

    result = cache.extract_all(
        "timeout_probe",
        slow_feature,
        indices=np.array([42]),
        n_jobs=2,
        timeout_seconds=2,
        timeout_result={"value": np.nan, "morph_failed": 1.0},
    )
    assert result.loc[0, "orig_index"] == 42
    assert result.loc[0, "morph_failed"] == 1
    assert np.isnan(result.loc[0, "value"])
