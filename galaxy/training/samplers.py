"""Training samplers. Issue #18 (owner: Eftelya). The default (``shuffle``) must exist before #4 is merged."""
from __future__ import annotations

SAMPLERS = ("shuffle", "oversample")


def build_sampler(cfg: dict, labels):
    """Return ``None`` for ``shuffle`` (DataLoader shuffles), or a class-balanced ``WeightedRandomSampler``."""
    raise NotImplementedError("Issue #4 (shuffle) / #18 (oversample)")
