"""Test-time rotation for the orientation sweep. Issue #23 (owner: Eren)."""
from __future__ import annotations


def make_rotation_fn(angle_deg: float):
    """Return a ``transform_fn`` for ``predict()`` that rotates every raw 256 image by ``angle_deg``."""
    raise NotImplementedError("Issue #23")
