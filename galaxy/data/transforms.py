"""Batched GPU preprocessing and augmentation. Issue #9 (owner: Eren). Protocol: PROJECT_PLAN §3.1–3.2.

Geometry (identical for all models and policies): rotate the full 256×256 image → center-crop 180 → resize to the
model input size → per-band standardization with the frozen train statistics.

Policies (training only, ``train=True``):
* ``none``    : crop + resize only
* ``rotflip`` : angle ~ U[0°, 360°) and independent horizontal / vertical flips with p = 0.5 (the default)
* ``d4``      : a random element of the dihedral group D4 (rot90 · k, k ∈ {0..3}, × optional h-flip); exact, no
                interpolation

Conventions: positive angles rotate **counter-clockwise as displayed** (same as ``torch.rot90(x, 1, (2, 3))`` and
``np.rot90``). Rotation and cropping are done in a single ``grid_sample`` that only samples the crop region,
so a rotated crop costs the same as a plain one. 180 < 256/√2, so no padded pixel can enter the crop.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from galaxy.constants import CROP_SIZE

POLICIES = ("none", "rotflip", "d4")


def _as_angles(angles_deg, batch: int, device) -> Tensor:
    a = torch.as_tensor(angles_deg, dtype=torch.float32, device=device)
    return a.expand(batch) if a.ndim == 0 else a


def _rot90_per_sample(x: Tensor, k: Tensor) -> Tensor:
    """Rotate sample i by k[i]·90° (CCW) exactly. Only valid for square images."""
    k = k.to(torch.long) % 4
    if bool((k == k[0]).all()):
        return torch.rot90(x, int(k[0]), dims=(2, 3))
    out = torch.empty_like(x)
    for kk in range(4):
        m = k == kk
        if m.any():
            out[m] = torch.rot90(x[m], kk, dims=(2, 3))
    return out


def center_crop(x: Tensor, size: int) -> Tensor:
    h, w = x.shape[-2:]
    if size > min(h, w):
        raise ValueError(f"crop {size} larger than image {h}×{w}")
    top, left = (h - size) // 2, (w - size) // 2
    return x[..., top:top + size, left:left + size]


def resize(x: Tensor, size: int) -> Tensor:
    """Antialiased bilinear resize of a (B,C,H,W) float batch (no-op if already ``size``)."""
    if x.shape[-1] == size and x.shape[-2] == size:
        return x
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)


def rotate_batch(x: Tensor, angles_deg, crop: int | None = None, exact_if_possible: bool = True) -> Tensor:
    """Rotate every image of a float batch ``x`` (B,C,H,W, square) by its own angle (degrees, CCW).

    Args:
        angles_deg: scalar or (B,) tensor/sequence.
        crop: if given, return only the centered ``crop``×``crop`` region of the rotated image. This is done in the
            same ``grid_sample`` call, so it is cheaper than rotating and then cropping.
        exact_if_possible: use ``torch.rot90`` (no interpolation) when every angle is a multiple of 90°.
    Bilinear interpolation, zero padding outside the image (never reached inside a crop ≤ H/√2).
    """
    b, c, h, w = x.shape
    if h != w:
        raise ValueError("rotate_batch expects square images")
    angles = _as_angles(angles_deg, b, x.device)
    if exact_if_possible and bool((torch.remainder(angles, 90) == 0).all()):
        out = _rot90_per_sample(x, torch.round(angles / 90))
        return center_crop(out, crop) if crop else out

    size = crop or h
    rad = angles * (math.pi / 180)
    cos, sin = torch.cos(rad), torch.sin(rad)
    s = size / h  # output covers the central `size` pixels of the input (normalized coordinates)
    # sampling grid: input = s · R(-θ) · output   (y axis points down → this is a CCW rotation on screen)
    theta = torch.stack([torch.stack([cos * s, -sin * s, torch.zeros_like(cos)], 1),
                         torch.stack([sin * s, cos * s, torch.zeros_like(cos)], 1)], 1)
    grid = F.affine_grid(theta.to(x.dtype), [b, c, size, size], align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def random_flips(x: Tensor, p_h: float = 0.5, p_v: float = 0.5) -> Tensor:
    b = x.shape[0]
    if p_h > 0:
        m = (torch.rand(b, device=x.device) < p_h).view(b, 1, 1, 1)
        x = torch.where(m, x.flip(3), x)
    if p_v > 0:
        m = (torch.rand(b, device=x.device) < p_v).view(b, 1, 1, 1)
        x = torch.where(m, x.flip(2), x)
    return x


def to_float01(x: Tensor) -> Tensor:
    """uint8 [0,255] → float32 [0,1]. Float input is assumed to be [0,1] already and passed through."""
    return x.float().div_(255.0) if x.dtype == torch.uint8 else x.float()


class GalaxyPreprocess(nn.Module):
    """``forward(x[B,3,256,256] uint8 or float01, train) -> float[B,3,out,out]`` (standardized).

    Move it to the GPU together with the model (``preprocess.to(device)``) and feed it raw batches from the
    DataLoader. Augmentation is applied only when ``train=True``. ``transform_fn`` hooks (rotation sweep,
    corruptions) are applied by ``galaxy.evaluation.predict.predict`` to the float01 batch before this module.
    """

    def __init__(self, policy: str = "rotflip", out_size: int = 224, band_stats: dict | None = None,
                 crop: int = CROP_SIZE):
        super().__init__()
        if policy not in POLICIES:
            raise ValueError(f"unknown policy '{policy}', expected one of {POLICIES}")
        if band_stats is None:
            from galaxy.data.dataset import load_band_stats

            band_stats = load_band_stats()
        if band_stats.get("crop", crop) != crop:
            raise ValueError(f"band stats were computed on a {band_stats['crop']} crop, preprocessing uses {crop}")
        self.policy, self.out_size, self.crop = policy, int(out_size), int(crop)
        self.register_buffer("mean", torch.tensor(band_stats["mean"], dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(band_stats["std"], dtype=torch.float32).view(1, 3, 1, 1))

    @classmethod
    def from_config(cls, cfg: dict, band_stats: dict | None = None) -> GalaxyPreprocess:
        p = cfg.get("preprocess", {})
        return cls(policy=p.get("policy", "rotflip"), out_size=p.get("out_size", 224), band_stats=band_stats,
                   crop=p.get("crop", CROP_SIZE))

    def geometry(self, x: Tensor, train: bool) -> Tensor:
        """float01 (B,3,256,256) → augmented (if train) center crop (B,3,crop,crop)."""
        if train and self.policy == "rotflip":
            angles = torch.rand(x.shape[0], device=x.device) * 360.0
            return random_flips(rotate_batch(x, angles, crop=self.crop, exact_if_possible=False))
        if train and self.policy == "d4":
            k = torch.randint(0, 4, (x.shape[0],), device=x.device)
            return random_flips(center_crop(_rot90_per_sample(x, k), self.crop), p_h=0.5, p_v=0.0)
        return center_crop(x, self.crop)

    def normalize(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    @torch.no_grad()
    def forward(self, x: Tensor, train: bool = False) -> Tensor:
        x = to_float01(x)
        return self.normalize(resize(self.geometry(x, train), self.out_size)).contiguous()

    def extra_repr(self) -> str:
        return f"policy={self.policy}, crop={self.crop}, out_size={self.out_size}"
