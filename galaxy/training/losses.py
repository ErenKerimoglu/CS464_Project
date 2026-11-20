"""Loss functions. Issue #18 (owner: Eftelya). The default (``ce``) must exist before #4 is merged."""
from __future__ import annotations

LOSSES = ("ce", "weighted_ce", "focal")


def build_loss(cfg: dict, class_counts):
    """``cfg["loss"]["name"]`` ∈ LOSSES. ``ce`` uses ``cfg["loss"]["label_smoothing"]``."""
    raise NotImplementedError("Issue #4 (ce) / #18 (weighted_ce, focal)")
