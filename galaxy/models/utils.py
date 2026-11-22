"""Model helpers. Issue #6 (owner: Berkay)."""
from __future__ import annotations

import torch
from torch import nn


def count_params(model: nn.Module, trainable_only: bool = False) -> int:
    """Number of parameters (optionally only those with ``requires_grad``)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


@torch.no_grad()
def model_summary(model: nn.Module, input_size: int | None = None, batch_size: int = 1) -> str:
    """Short text summary: parameters per top-level child, total / trainable counts and the output shape.

    ``input_size`` defaults to ``model.input_size``. The model is run once in eval mode on zeros and its
    training flag is restored afterwards.
    """
    size = input_size or getattr(model, "input_size", None)
    if size is None:
        raise ValueError("input_size not given and model has no `input_size` attribute")
    lines = [f"{type(model).__name__} (input {size}x{size})"]
    for name, child in model.named_children():
        lines.append(f"  {name:<16}{count_params(child):>12,}")
    was_training = model.training
    model.eval()
    try:
        device = next(model.parameters()).device
        out = model(torch.zeros(batch_size, 3, size, size, device=device))
    finally:
        model.train(was_training)
    lines.append(f"  total params    {count_params(model):>12,}")
    lines.append(f"  trainable       {count_params(model, trainable_only=True):>12,}")
    lines.append(f"  output shape    {tuple(out.shape)}")
    return "\n".join(lines)
