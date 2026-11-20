"""Trainer. Issue #4 (owner: Ege). Contract: PROJECT_PLAN §4.3.

AMP, AdamW + warm-up/cosine, early stopping on val macro-F1, ``last.pt``/``best.pt`` checkpoints, exact resume,
``train_log.csv``, ``freeze_backbone`` (linear probe), and loss/sampler plug-ins from ``losses``/``samplers``.
"""
from __future__ import annotations


class Trainer:
    def __init__(self, cfg: dict, model, loaders: dict, preprocess, run_dir, ckpt_dir, device: str = "cuda"):
        raise NotImplementedError("Issue #4")

    def fit(self, resume: bool = False) -> dict:
        """Train until ``max_epochs`` or early stopping. Return a summary (best epoch, best val macro-F1, time)."""
        raise NotImplementedError("Issue #4")
