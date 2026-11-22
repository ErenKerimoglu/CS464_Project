"""Loss functions. Issue #18 (owner: Eftelya). The default (``ce``) must exist before #4 is merged."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

LOSSES = ("ce", "weighted_ce", "focal")


def _class_weights(class_counts, beta: float = 1.0, effective: bool = False, device=None) -> torch.Tensor:
    counts = torch.as_tensor(class_counts, dtype=torch.float32, device=device)
    if effective:
        weights = (1.0 - torch.pow(torch.tensor(0.99, device=device), counts)) / (1.0 - 0.99)
        weights = 1.0 / weights
    else:
        weights = torch.pow(counts.clamp_min(1.0), -beta)
    return weights * (weights.numel() / weights.sum())


class FocalLoss(nn.Module):
    """Stable focal loss with optional class weights."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probs = log_probs.exp()
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -torch.pow(1.0 - target_probs, self.gamma) * target_log_probs
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device).gather(0, targets)
        return loss.mean()


def build_loss(cfg: dict, class_counts):
    """``cfg["loss"]["name"]`` ∈ LOSSES. ``ce`` uses ``cfg["loss"]["label_smoothing"]``."""
    loss_cfg = cfg.get("loss", {})
    name = loss_cfg.get("name", "ce")
    if name not in LOSSES:
        raise ValueError(f"Unknown loss {name!r}; expected one of {LOSSES}")
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)))
    if name == "weighted_ce":
        beta = float(loss_cfg.get("beta", 1.0))
        effective = bool(loss_cfg.get("effective_number", False))
        weights = _class_weights(class_counts, beta=beta, effective=effective)
        return nn.CrossEntropyLoss(weight=weights, label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)))
    gamma = float(loss_cfg.get("gamma", 2.0))
    alpha = None
    if bool(loss_cfg.get("use_class_weights", True)):
        alpha = _class_weights(class_counts, beta=float(loss_cfg.get("beta", 1.0)))
    return FocalLoss(gamma=gamma, alpha=alpha)
