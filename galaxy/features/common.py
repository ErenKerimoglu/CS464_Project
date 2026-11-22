"""NumPy geometry helpers matching the GPU pipeline. Issue #7 (owner: Yunus)."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def to_float01(img_uint8: np.ndarray) -> np.ndarray:
    """Convert uint8 pixels to float [0,1]. Float inputs are passed through."""
    img = np.asarray(img_uint8)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if np.issubdtype(img.dtype, np.floating):
        return img.astype(np.float32, copy=False)
    return img.astype(np.float32)


def center_crop_np(img: np.ndarray, size: int = 180) -> np.ndarray:
    """Center crop of an (H,W,C) or (H,W) image."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        h, w = arr.shape
        top, left = (h - size) // 2, (w - size) // 2
        if size > min(h, w):
            raise ValueError(f"crop {size} larger than image {h}×{w}")
        return arr[top:top + size, left:left + size]
    if arr.ndim == 3:
        if arr.shape[-1] in (1, 3, 4) and arr.shape[-2] >= size:
            h, w = arr.shape[:2]
            top, left = (h - size) // 2, (w - size) // 2
            if size > min(h, w):
                raise ValueError(f"crop {size} larger than image {h}×{w}")
            return arr[top:top + size, left:left + size]
        # Also accept CHW arrays.
        if arr.shape[0] in (1, 3, 4):
            c, h, w = arr.shape
            top, left = (h - size) // 2, (w - size) // 2
            if size > min(h, w):
                raise ValueError(f"crop {size} larger than image {h}×{w}")
            return arr[:, top:top + size, left:left + size]
    raise ValueError("center crop expects a 2D, HWC, or CHW image")


def rotate_np(img256: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate the full 256 image (bilinear, zero fill), then crop.

    Must match ``galaxy.data.transforms.rotate_batch``.
    """
    arr = np.asarray(img256)
    if arr.ndim != 2 and arr.ndim != 3:
        raise ValueError("rotate_np expects a 2D or 3D image")

    angle = float(angle_deg)
    if abs(angle) < 1e-12:
        rotated = arr.copy()
    elif abs(angle % 90) < 1e-12:
        k = int(round(angle / 90)) % 4
        if arr.ndim == 2:
            rotated = np.rot90(arr, k=k, axes=(0, 1))
        elif arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
            rotated = np.rot90(arr, k=k, axes=(0, 1))
        elif arr.shape[0] in (1, 3, 4):
            rotated = np.rot90(arr, k=k, axes=(1, 2))
        else:
            rotated = np.rot90(arr, k=k, axes=(0, 1))
    else:
        if arr.ndim == 2:
            rotated = ndimage.rotate(arr, angle, reshape=False, order=1, mode="constant", cval=0, prefilter=False)
        else:
            # ndimage rotates HWC/CHW consistently over the first two axes for HWC.
            if arr.shape[-1] in (1, 3, 4) and arr.shape[0] not in (1, 3, 4):
                rotated = ndimage.rotate(arr, angle, axes=(0, 1), reshape=False, order=1,
                                         mode="constant", cval=0, prefilter=False)
            elif arr.shape[0] in (1, 3, 4):
                rotated = np.stack([
                    ndimage.rotate(arr[c], angle, reshape=False, order=1, mode="constant", cval=0, prefilter=False)
                    for c in range(arr.shape[0])
                ], axis=0)
            else:
                rotated = ndimage.rotate(arr, angle, axes=(0, 1), reshape=False, order=1,
                                         mode="constant", cval=0, prefilter=False)
    return center_crop_np(rotated, 180)
