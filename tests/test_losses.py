"""Synthetic tests for class-imbalance losses and samplers (#18)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from galaxy.training.losses import FocalLoss, build_loss
from galaxy.training.samplers import build_sampler


def test_focal_gamma_zero_equals_cross_entropy():
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.2, 1.4, -0.1]], requires_grad=True)
    targets = torch.tensor([0, 1])
    focal = FocalLoss(gamma=0.0)(logits, targets)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    assert torch.allclose(focal, ce, atol=1e-6)


def test_weighted_ce_uniform_counts_equals_ce():
    logits = torch.randn(16, 4, requires_grad=True)
    targets = torch.randint(0, 4, (16,))
    weighted = build_loss({"loss": {"name": "weighted_ce", "beta": 1.0}}, [10, 10, 10, 10])
    ce = build_loss({"loss": {"name": "ce"}}, [10, 10, 10, 10])
    assert torch.allclose(weighted(logits, targets), ce(logits, targets), atol=1e-6)


def test_oversampler_is_roughly_balanced():
    labels = np.repeat(np.arange(3), [5, 20, 75])
    sampler = build_sampler({"sampler": {"name": "oversample", "seed": 464}}, labels)
    draws = np.asarray([labels[int(index)] for _ in range(100) for index in sampler])
    counts = np.bincount(draws, minlength=3)
    proportions = counts / counts.sum()
    assert np.abs(proportions - 1.0 / 3.0).max() < 0.03


@pytest.mark.parametrize("name", ["ce", "weighted_ce", "focal"])
def test_build_loss_accepts_name(name):
    loss = build_loss({"loss": {"name": name}}, [100, 50, 10])
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))
    assert torch.isfinite(loss(logits, targets))
