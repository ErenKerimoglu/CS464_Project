"""Optimizer and LR schedule. Issue #4 (owner: Ege). Recipe: PROJECT_PLAN §3.2.

* AdamW with ``train.weight_decay`` on weight matrices/kernels only. Biases and normalization parameters (every
  parameter with ``ndim <= 1``, which also covers escnn's batch norms) get no weight decay.
* ``train.head_lr`` (optional) gives the classifier head its own learning rate. The head is whatever the model returns
  from ``head_parameters()`` (all our models implement it). ``null`` → same LR as the backbone.
* Frozen parameters (``requires_grad=False``, e.g. linear probe) are left out of the optimizer.
* Schedule, stepped **per optimizer step**: linear warm-up over ``train.warmup_epochs`` epochs, then cosine decay to
  ``train.min_lr_ratio × lr`` (default 0) at ``train.max_epochs``.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def param_groups(model: nn.Module, cfg: dict) -> list[dict]:
    """AdamW parameter groups: {backbone, head} × {decay, no_decay}; empty groups are dropped."""
    t = cfg.get("train", {})
    lr = float(t.get("lr", 1e-4))
    head_lr = float(t["head_lr"]) if t.get("head_lr") is not None else lr
    wd = float(t.get("weight_decay", 0.0))
    head_ids = {id(p) for p in model.head_parameters()} if hasattr(model, "head_parameters") else set()

    groups: dict[tuple[str, bool], list] = {}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head_ids else "backbone"
        groups.setdefault((part, p.ndim > 1), []).append(p)
    out = []
    for (part, decay), params in sorted(groups.items()):
        out.append({"params": params, "lr": head_lr if part == "head" else lr,
                    "weight_decay": wd if decay else 0.0, "name": f"{part}_{'decay' if decay else 'no_decay'}"})
    if not out:
        raise ValueError("the model has no trainable parameters")
    return out


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """AdamW. No weight decay on norm/bias parameters. Optional separate head LR (``train.head_lr``)."""
    t = cfg.get("train", {})
    betas = tuple(t.get("betas", (0.9, 0.999)))
    return torch.optim.AdamW(param_groups(model, cfg), lr=float(t.get("lr", 1e-4)), betas=betas)


def lr_factor(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.0) -> float:
    """Multiplier of the base LR at optimizer step ``step`` (0-based): linear warm-up, then cosine."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, steps_per_epoch: int):
    """Linear warm-up for ``train.warmup_epochs``, then per-step cosine decay (to ``min_lr_ratio × lr``)."""
    t = cfg.get("train", {})
    total = int(t.get("max_epochs", 30)) * steps_per_epoch
    warmup = int(round(float(t.get("warmup_epochs", 0)) * steps_per_epoch))
    min_ratio = float(t.get("min_lr_ratio", 0.0))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, warmup, total, min_ratio))
