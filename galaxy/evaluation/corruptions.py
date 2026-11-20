"""Image corruptions simulating weaker telescopes. Issue #22 (owner: Yunus)."""
from __future__ import annotations

SEVERITIES = {
    "noise": (0.02, 0.05, 0.10, 0.20),   # Gaussian sigma in [0,1] pixel units
    "blur": (1, 2, 3, 4),                # Gaussian PSF sigma in pixels
    "downsample": (2, 4, 8),             # area downsample factor, then bilinear upsample back to 256
}


def make_corruption_fn(name: str, severity: int, seed: int = 0):
    """``transform_fn`` for ``predict()``. ``severity`` is a 1-based index into ``SEVERITIES[name]``; 0 = clean."""
    raise NotImplementedError("Issue #22")
