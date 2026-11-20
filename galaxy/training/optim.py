"""Optimizer and LR schedule. Issue #4 (owner: Ege)."""
from __future__ import annotations


def build_optimizer(model, cfg: dict):
    """AdamW. No weight decay on norm/bias parameters. Optional separate head LR (``train.head_lr``)."""
    raise NotImplementedError("Issue #4")


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    """Linear warm-up for ``train.warmup_epochs``, then per-step cosine decay to 0."""
    raise NotImplementedError("Issue #4")
