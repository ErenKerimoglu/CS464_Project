"""Synthetic tests for E6 confusion-matrix analysis."""
from __future__ import annotations

import numpy as np
import pytest

from galaxy.analysis.confusion import mean_row_normalized_confusion, top_confused_pairs


def test_mean_row_normalized_confusion_averages_per_seed():
    predictions = [
        (np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])),
        (np.array([0, 0, 0, 1]), np.array([0, 0, 1, 0])),
    ]

    matrix = mean_row_normalized_confusion(predictions, num_classes=3)

    assert np.allclose(matrix, [
        [7 / 12, 5 / 12, 0],
        [1 / 2, 1 / 2, 0],
        [0, 0, 0],
    ])


def test_mean_row_normalized_confusion_accepts_probabilities():
    predictions = [(np.array([0, 1]), np.array([[0.8, 0.2], [0.3, 0.7]]))]

    matrix = mean_row_normalized_confusion(predictions, num_classes=2)

    assert matrix == pytest.approx(np.eye(2))


def test_mean_row_normalized_confusion_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="at least one seed"):
        mean_row_normalized_confusion([])
    with pytest.raises(ValueError, match="non-empty 1-D"):
        mean_row_normalized_confusion([(np.array([]), np.array([]))])
    with pytest.raises(ValueError, match="different sample counts"):
        mean_row_normalized_confusion([(np.array([0, 1]), np.array([0]))])


def test_top_confused_pairs_excludes_diagonal_and_breaks_ties_stably():
    matrix = np.array([
        [0.5, 0.4, 0.1],
        [0.3, 0.6, 0.1],
        [0.2, 0.7, 0.1],
    ])

    pairs = top_confused_pairs(matrix, limit=3, class_names=("A", "B", "C"))

    assert pairs == [
        {"true_class": "C", "predicted_class": "B", "rate": 0.7},
        {"true_class": "A", "predicted_class": "B", "rate": 0.4},
        {"true_class": "B", "predicted_class": "A", "rate": 0.3},
    ]


def test_top_confused_pairs_validates_matrix_and_limit():
    with pytest.raises(ValueError, match="shape"):
        top_confused_pairs(np.eye(2), class_names=("A",))
    with pytest.raises(ValueError, match="finite"):
        top_confused_pairs(np.array([[0.0, np.nan], [0.0, 1.0]]), class_names=("A", "B"))
    with pytest.raises(ValueError, match="non-negative"):
        top_confused_pairs(np.array([[0.0, -0.1], [0.0, 1.0]]), class_names=("A", "B"))
    with pytest.raises(ValueError, match="limit"):
        top_confused_pairs(np.eye(2), limit=-1, class_names=("A", "B"))
