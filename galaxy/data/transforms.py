"""Batched GPU preprocessing and augmentation. Issue #9 (owner: Eren). Protocol: PROJECT_PLAN §3.1–3.2.

Geometry (identical for all models and policies): rotate the full 256×256 image → center-crop 180 → resize to the
model input size → per-band standardization with train statistics.

Policies: ``none`` | ``rotflip`` (uniform angle in [0,360) + random h/v flips; the default)
| ``d4`` (random rot90 × flip).
"""
from __future__ import annotations

POLICIES = ("none", "rotflip", "d4")


def rotate_batch(x, angles_deg):
    """Rotate each image of ``x`` (B,C,H,W, float) by its own angle (degrees, counter-clockwise).

    Uses ``affine_grid``/``grid_sample`` (bilinear, zero padding). Uses the exact ``torch.rot90`` path when every
    angle is a multiple of 90°. Shared by augmentation (#9), TTA (#17), the rotation sweep (#23) and the classical
    orientation study (#24, NumPy twin in ``galaxy.features.common.rotate_np``).
    """
    raise NotImplementedError("Issue #9")


def center_crop(x, size: int):
    raise NotImplementedError("Issue #9")


def resize(x, size: int):
    """Antialiased bilinear resize of a (B,C,H,W) batch."""
    raise NotImplementedError("Issue #9")


class GalaxyPreprocess:  # (torch.nn.Module)
    """``forward(x_uint8[B,3,256,256], train: bool) -> float[B,3,out,out]`` (standardized).

    Augmentation is applied only when ``train=True``. ``transform_fn`` hooks (rotation, corruptions) are applied by
    ``galaxy.evaluation.predict.predict`` to the raw batch **before** this module.
    """

    def __init__(self, policy: str, out_size: int, band_stats: dict, crop: int = 180):
        raise NotImplementedError("Issue #9")
