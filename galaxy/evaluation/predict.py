"""Model inference with optional raw-image transform hook. Issue #5 (owner: Ege)."""
from __future__ import annotations


def predict(model, loader, preprocess, device: str = "cuda", transform_fn=None, tta=None):
    """Return ``(probs[N,10], y_true[N], orig_index[N])``.

    ``transform_fn(x_float01[B,3,256,256]) -> same shape`` is applied **before** ``preprocess`` (rotation sweep #23,
    corruptions #22). ``tta`` (#17) maps a preprocessed batch to a list of views whose softmax outputs are averaged.
    """
    raise NotImplementedError("Issue #5")
