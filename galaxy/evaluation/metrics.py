"""Classification metrics. Issue #5 (owner: Ege)."""
from __future__ import annotations

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred_or_probs: np.ndarray, num_classes: int = 10) -> dict:
    """accuracy, macro_f1, balanced_accuracy, per_class_{recall,precision,f1}, confusion_matrix, n_samples.

    Accepts class indices (N,) or probabilities (N,C). Must handle classes missing from ``y_true``.
    """
    raise NotImplementedError("Issue #5")
