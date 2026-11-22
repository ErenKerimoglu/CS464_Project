"""Classification metrics. Issue #5 (owner: Ege). Protocol: PROJECT_PLAN §3.4.

Everything is computed from one ``num_classes × num_classes`` confusion matrix, so the per-class arrays always have
``num_classes`` entries in class-index order (``galaxy.constants.CLASS_NAMES``), even when a class is missing from
a (small) subset. Averages follow ``sklearn.metrics`` exactly (``tests/test_metrics.py`` checks this):

* ``macro_f1``: mean F1 over the classes that occur in ``y_true`` **or** in the predictions (sklearn's default label
  set). A class that is absent from both does not pull the average down.
* ``balanced_accuracy``: mean recall over the classes that occur in ``y_true``.
* Undefined per-class values (0/0) are reported as 0 (sklearn's ``zero_division=0``).

On the real val/test sets every class is present, so all of these reduce to the textbook definitions.
"""
from __future__ import annotations

import numpy as np

from galaxy.constants import NUM_CLASSES

METRIC_KEYS = ("accuracy", "macro_f1", "balanced_accuracy", "per_class_recall", "per_class_precision",
               "per_class_f1", "confusion_matrix", "n_samples")


def to_class_predictions(y_pred_or_probs, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Class indices ``(N,)`` from either class indices ``(N,)`` or scores/probabilities ``(N, C)`` (argmax)."""
    y = np.asarray(y_pred_or_probs)
    if y.ndim == 2:
        if y.shape[1] != num_classes:
            raise ValueError(f"probabilities have {y.shape[1]} columns, expected {num_classes}")
        return y.argmax(axis=1).astype(np.int64)
    if y.ndim != 1:
        raise ValueError(f"expected class indices (N,) or probabilities (N,C), got shape {y.shape}")
    return y.astype(np.int64)


def confusion_matrix(y_true, y_pred, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """``cm[i, j]`` = number of samples with true class ``i`` predicted as ``j`` (int64, ``C × C``)."""
    y_true, y_pred = np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)
    for name, y in (("y_true", y_true), ("y_pred", y_pred)):
        if y.size and (y.min() < 0 or y.max() >= num_classes):
            raise ValueError(f"{name} contains labels outside [0, {num_classes})")
    return np.bincount(y_true * num_classes + y_pred, minlength=num_classes**2).reshape(num_classes, num_classes)


def row_normalize(cm) -> np.ndarray:
    """Row-normalized confusion matrix (each true-class row sums to 1; empty rows stay 0)."""
    cm = np.asarray(cm, dtype=np.float64)
    support = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, support, out=np.zeros_like(cm), where=support > 0)


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b > 0)


def compute_metrics(y_true, y_pred_or_probs, num_classes: int = NUM_CLASSES) -> dict:
    """accuracy, macro_f1, balanced_accuracy, per_class_{recall,precision,f1}, confusion_matrix, n_samples.

    Args:
        y_true: true class indices ``(N,)``.
        y_pred_or_probs: predicted class indices ``(N,)`` or probabilities/logits ``(N, num_classes)``.
    Returns:
        A JSON-serializable dict (plain Python floats/ints/lists) with the keys of :data:`METRIC_KEYS`.
    """
    y_true = np.asarray(y_true)
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1-D class indices, got shape {y_true.shape}")
    y_pred = to_class_predictions(y_pred_or_probs, num_classes)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} samples, predictions have {len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("cannot compute metrics on an empty set")

    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1)          # true count per class
    predicted = cm.sum(axis=0)        # predicted count per class
    recall = _safe_div(tp, support)
    precision = _safe_div(tp, predicted)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    present = (support > 0) | (predicted > 0)
    return {
        "accuracy": float(tp.sum() / len(y_true)),
        "macro_f1": float(f1[present].mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "per_class_recall": recall.tolist(),
        "per_class_precision": precision.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }
