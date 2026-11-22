"""Training samplers. Issue #18 (owner: Eftelya). The default (``shuffle``) must exist before #4 is merged."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

SAMPLERS = ("shuffle", "oversample")


def build_sampler(cfg: dict, labels):
    """Return ``None`` for ``shuffle`` (DataLoader shuffles), or a class-balanced ``WeightedRandomSampler``."""
    sampler_cfg = cfg.get("sampler", {})
    name = sampler_cfg.get("name", "shuffle")
    if name not in SAMPLERS:
        raise ValueError(f"Unknown sampler {name!r}; expected one of {SAMPLERS}")
    if name == "shuffle":
        return None

    labels_array = np.asarray(labels)
    classes, counts = np.unique(labels_array, return_counts=True)
    count_by_class = dict(zip(classes.tolist(), counts.tolist(), strict=True))
    sample_weights = torch.as_tensor(
        [1.0 / count_by_class[int(label)] for label in labels_array],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels_array),
        replacement=True,
        generator=torch.Generator().manual_seed(int(sampler_cfg.get("seed", 464))),
    )
