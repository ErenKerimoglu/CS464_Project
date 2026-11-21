"""Geometry and normalization guarantees of the shared GPU preprocessing (#9). CPU, synthetic data."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from galaxy.data.transforms import GalaxyPreprocess, center_crop, resize, rotate_batch

STATS = {"mean": [0.2, 0.19, 0.18], "std": [0.15, 0.14, 0.13], "crop": 180}


@pytest.fixture
def batch(synthetic_images) -> torch.Tensor:
    return torch.from_numpy(synthetic_images[:6]).permute(0, 3, 1, 2).float() / 255


def test_rot90_convention_matches_numpy(batch):
    out = rotate_batch(batch, 90.0)
    ref = torch.from_numpy(np.rot90(batch.numpy(), 1, axes=(2, 3)).copy())  # CCW as displayed
    assert torch.equal(out, ref)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_interpolated_path_agrees_with_exact_rot90(batch, k):
    exact = rotate_batch(batch, 90.0 * k, crop=180)
    interp = rotate_batch(batch, 90.0 * k, crop=180, exact_if_possible=False)
    assert torch.allclose(exact, interp, atol=1e-5)  # same direction and same pixel grid


def test_zero_and_full_turn_are_identity(batch):
    assert torch.allclose(rotate_batch(batch, 0.0, crop=180, exact_if_possible=False), center_crop(batch, 180),
                          atol=1e-5)
    assert torch.allclose(rotate_batch(batch, 360.0, crop=180, exact_if_possible=False), center_crop(batch, 180),
                          atol=1e-5)


def test_per_sample_angles(batch):
    angles = torch.tensor([0.0, 90.0, 180.0, 270.0, 0.0, 90.0])
    out = rotate_batch(batch, angles)
    for i, a in enumerate(angles.tolist()):
        assert torch.equal(out[i], torch.rot90(batch[i], int(a // 90), dims=(1, 2)))


@pytest.mark.parametrize("angle", [17.0, 45.0, 133.0, 225.0, 314.9])
def test_no_padding_inside_crop(angle):
    ones = torch.ones(2, 3, 256, 256)
    out = rotate_batch(ones, angle, crop=180, exact_if_possible=False)
    assert out.shape == (2, 3, 180, 180)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6)  # no zero-padded pixel reached the crop


def test_fused_crop_equals_rotate_then_crop(batch):
    fused = rotate_batch(batch, 33.0, crop=180, exact_if_possible=False)
    full = center_crop(rotate_batch(batch, 33.0, exact_if_possible=False), 180)
    assert torch.allclose(fused, full, atol=1e-5)


def test_rotation_moves_an_offcenter_dot_counterclockwise():
    x = torch.zeros(1, 1, 256, 256)
    x[0, 0, 128, 168] = 1.0  # right of the center (row 128, col 168)
    out = rotate_batch(x, 90.0, exact_if_possible=False)
    r, c = divmod(int(out[0, 0].argmax()), 256)
    assert abs(c - 128) <= 1 and r < 100  # CCW on screen: right → top


@pytest.mark.parametrize("policy", ["none", "rotflip", "d4"])
def test_preprocess_shapes_dtype_and_eval_determinism(synthetic_images, policy):
    pre = GalaxyPreprocess(policy, out_size=224, band_stats=STATS)
    x = torch.from_numpy(synthetic_images[:4]).permute(0, 3, 1, 2).contiguous()
    y = pre(x, train=True)
    assert y.shape == (4, 3, 224, 224) and y.dtype == torch.float32 and torch.isfinite(y).all()
    assert torch.equal(pre(x, train=False), pre(x, train=False))       # eval never augments
    assert torch.equal(pre(x, train=False), GalaxyPreprocess("none", 224, STATS)(x, train=False))


def test_preprocess_accepts_float01_and_uint8(synthetic_images):
    pre = GalaxyPreprocess("none", out_size=180, band_stats=STATS)
    x8 = torch.from_numpy(synthetic_images[:2]).permute(0, 3, 1, 2).contiguous()
    assert torch.allclose(pre(x8), pre(x8.float() / 255), atol=1e-6)


def test_standardization_uses_band_stats():
    rng = np.random.default_rng(0)
    mean, std = np.array(STATS["mean"]), np.array(STATS["std"])
    x = rng.normal(mean[None, :, None, None], std[None, :, None, None], size=(64, 3, 256, 256)).astype(np.float32)
    y = GalaxyPreprocess("none", out_size=180, band_stats=STATS)(torch.from_numpy(x))
    assert torch.allclose(y.mean((0, 2, 3)), torch.zeros(3), atol=0.01)
    assert torch.allclose(y.std((0, 2, 3)), torch.ones(3), atol=0.01)


def test_augmentation_actually_varies(synthetic_images):
    torch.manual_seed(0)
    pre = GalaxyPreprocess("rotflip", out_size=128, band_stats=STATS)
    x = torch.from_numpy(np.repeat(synthetic_images[:1], 8, 0)).permute(0, 3, 1, 2).contiguous()
    y = pre(x, train=True)
    assert len({round(float(t.sum()), 2) for t in (y - y[0]).abs()}) > 4  # 8 copies → different views


def test_resize_and_bad_inputs():
    assert resize(torch.rand(1, 3, 180, 180), 224).shape[-1] == 224
    with pytest.raises(ValueError):
        GalaxyPreprocess("bogus", 224, STATS)
    with pytest.raises(ValueError):
        GalaxyPreprocess("none", 224, {**STATS, "crop": 200})
