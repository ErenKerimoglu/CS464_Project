"""8-view dihedral test-time augmentation. Issue #17 (owner: Eren).

TTA works on the **preprocessed** batch (after the 180 crop, resize and normalization). The 8 elements of the
dihedral group D4 (4 exact 90° rotations × optional horizontal flip) are lossless pixel permutations, so no
interpolation is involved. ``predict()`` averages the softmax over the views.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

N_VIEWS = 8


def d4_views(x: Tensor) -> list[Tensor]:
    """8 views of a preprocessed batch ``(B,C,H,W)``: rot90 k=0..3 × {identity, horizontal flip}.

    The set of views is the same (up to order) for any D4-transformed input, so the averaged prediction is exactly
    invariant to 90° rotations and flips of the crop.
    """
    if x.ndim != 4 or x.shape[-1] != x.shape[-2]:
        raise ValueError(f"d4_views expects a square batch (B,C,H,W), got {tuple(x.shape)}")
    views = []
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        views += [r, torch.flip(r, dims=(3,))]
    return views


def predict_tta8(model: nn.Module, loader, preprocess: nn.Module, device=None, transform_fn=None,
                 amp: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``predict()`` with the 8 D4 views averaged: ``(probs, y_true, orig_index)``."""
    from galaxy.evaluation.predict import predict

    return predict(model, loader, preprocess, device, transform_fn=transform_fn, tta=d4_views, amp=amp)


def evaluate_tta8(model: nn.Module, loader, preprocess: nn.Module, run_dir: Path, config: dict, split: str = "test",
                  device=None, extra: dict | None = None, amp: bool = False) -> dict:
    """Evaluate with TTA-8 and write ``eval_<split>_tta8.json`` (same as ``scripts/evaluate.py --tta8``)."""
    from galaxy.evaluation.predict import evaluate
    from galaxy.evaluation.results import eval_name_for

    extra = {**(extra or {}), "tta": "d4_8"}
    return evaluate(model, loader, preprocess, run_dir, eval_name_for(split, tta8=True), config, device=device,
                    tta=d4_views, extra=extra, amp=amp)
