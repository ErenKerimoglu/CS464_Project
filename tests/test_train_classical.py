"""Tests for classical estimator construction."""
from __future__ import annotations

from pathlib import Path
from runpy import run_path

import numpy as np

_train_classical = run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "train_classical.py")
)
DEFAULT_PARAMS = _train_classical["DEFAULT_PARAMS"]
_make_estimator = _train_classical["_make_estimator"]


def test_logreg_and_svm_produce_ten_class_probabilities():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(50, 5))
    y = np.repeat(np.arange(10), 5)

    for model_name in ("logreg", "svm_rbf"):
        estimator = _make_estimator(
            model_name,
            DEFAULT_PARAMS[model_name],
            seed=0,
            probabilities=True,
        )
        estimator.fit(X, y)
        probabilities = estimator.predict_proba(X[:4])

        assert probabilities.shape == (4, 10)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
