"""Confusion-matrix helpers for the E6 error analysis."""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from galaxy.constants import CLASS_NAMES, NUM_CLASSES
from galaxy.evaluation.metrics import confusion_matrix, row_normalize, to_class_predictions


def mean_row_normalized_confusion(
    predictions: Iterable[tuple[np.ndarray, np.ndarray]],
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Average per-seed row-normalized confusion matrices.

    Each item is ``(y_true, y_pred_or_probs)`` for one seed. Averaging the
    normalized matrices gives each seed equal weight, even if sample counts
    differ. Classes absent from a seed have zero rows in that seed's matrix.
    """
    matrices = []
    for seed, (y_true, y_pred_or_probs) in enumerate(predictions):
        y_true = np.asarray(y_true)
        if y_true.ndim != 1 or y_true.size == 0:
            raise ValueError(f"seed {seed}: y_true must be a non-empty 1-D array")
        y_pred = to_class_predictions(y_pred_or_probs, num_classes)
        if len(y_true) != len(y_pred):
            raise ValueError(f"seed {seed}: y_true and predictions have different sample counts")
        matrices.append(row_normalize(confusion_matrix(y_true, y_pred, num_classes)))
    if not matrices:
        raise ValueError("at least one seed of predictions is required")
    return np.mean(matrices, axis=0)


def top_confused_pairs(
    mean_confusion: np.ndarray,
    limit: int = 10,
    class_names: Sequence[str] = CLASS_NAMES,
) -> list[dict[str, str | float]]:
    """Return the largest off-diagonal rates from a mean row-normalized matrix."""
    matrix = np.asarray(mean_confusion, dtype=np.float64)
    if matrix.shape != (len(class_names), len(class_names)):
        raise ValueError("mean_confusion shape must match the number of class names")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("mean_confusion must contain finite, non-negative values")
    if limit < 0:
        raise ValueError("limit must be non-negative")

    pairs = [
        (float(matrix[true_class, predicted_class]), true_class, predicted_class)
        for true_class in range(len(class_names))
        for predicted_class in range(len(class_names))
        if true_class != predicted_class and matrix[true_class, predicted_class] > 0
    ]
    pairs.sort(key=lambda pair: (-pair[0], pair[1], pair[2]))
    return [
        {
            "true_class": class_names[true_class],
            "predicted_class": class_names[predicted_class],
            "rate": rate,
        }
        for rate, true_class, predicted_class in pairs[:limit]
    ]
