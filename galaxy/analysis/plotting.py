"""One plotting style for the whole report. Issue #16 (owner: Ege). Every figure must use this module."""
from __future__ import annotations

# Fixed colour per model family (fill in the final palette in #16; keep keys stable).
MODEL_COLORS: dict[str, str] = {}
CLASS_COLORS: dict[str, str] = {}


def set_style() -> None:
    raise NotImplementedError("Issue #16")


def save_fig(fig, path) -> None:
    """Save as PDF + PNG (300 dpi) under results/figures/."""
    raise NotImplementedError("Issue #16")
