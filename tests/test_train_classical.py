"""Tests for classical estimator construction."""
from __future__ import annotations

import numpy as np

from scripts.train_classical import DEFAULT_PARAMS, _make_estimator


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
