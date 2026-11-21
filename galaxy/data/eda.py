"""Exploratory-analysis helpers used by notebooks/00_eda.ipynb. Issue #2 (owner: Eren)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from galaxy.constants import CLASS_NAMES, CLASS_SHORT, CROP_SIZE, NUM_CLASSES


def border_background(band: np.ndarray, width: int = 8) -> np.ndarray:
    """Median of the outer ``width``-pixel frame per image. ``band``: (B,H,W) → (B,)."""
    frame = np.concatenate([
        band[:, :width, :].reshape(len(band), -1), band[:, -width:, :].reshape(len(band), -1),
        band[:, width:-width, :width].reshape(len(band), -1), band[:, width:-width, -width:].reshape(len(band), -1),
    ], axis=1)
    return np.median(frame, axis=1)


def crop_flux_fraction(images: np.ndarray, crop: int = CROP_SIZE, band: int = 1) -> np.ndarray:
    """Fraction of background-subtracted flux (r-band by default) that lies inside the central ``crop``.

    ``images``: uint8 (B,256,256,3) → float (B,). A value near 1 means the galaxy fits in the crop.
    """
    x = images[..., band].astype(np.float32)
    x = np.clip(x - border_background(x)[:, None, None], 0, None)
    lo = (x.shape[1] - crop) // 2
    inside = x[:, lo:lo + crop, lo:lo + crop].sum(axis=(1, 2))
    total = x.sum(axis=(1, 2))
    return np.where(total > 0, inside / np.maximum(total, 1e-9), 1.0)


def pixel_stats(images: np.ndarray, crop: int | None = CROP_SIZE) -> pd.DataFrame:
    """Per-band mean/std/percentiles in [0,1] scale (optionally on the center crop)."""
    x = images
    if crop:
        lo = (x.shape[1] - crop) // 2
        x = x[:, lo:lo + crop, lo:lo + crop]
    x = x.reshape(-1, 3).astype(np.float32) / 255.0
    q = np.percentile(x, [1, 50, 99], axis=0)
    return pd.DataFrame({"mean": x.mean(0), "std": x.std(0), "p01": q[0], "p50": q[1], "p99": q[2]},
                        index=["g", "r", "z"]).round(4)


def class_count_table(labels: np.ndarray) -> pd.DataFrame:
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    return pd.DataFrame({"class": CLASS_NAMES, "short": CLASS_SHORT, "count": counts,
                         "share %": (counts / counts.sum() * 100).round(2)})


# ----------------------------------------------------------------------------- figures
def plot_class_counts(labels: np.ndarray, ax=None):
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(8, 3.5))[1]
    counts = np.bincount(labels, minlength=NUM_CLASSES)
    ax.bar(range(NUM_CLASSES), counts, color="#4C72B0")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(NUM_CLASSES), CLASS_SHORT, rotation=30)
    ax.set_ylabel("images")
    ax.set_title("Galaxy10 DECaLS class counts")
    return ax.figure


def plot_samples_grid(images, labels: np.ndarray, per_class: int = 8, seed: int = 0, crop: int | None = None):
    """``per_class`` random images per class, one row per class. ``images`` may be an h5 dataset."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(NUM_CLASSES, per_class, figsize=(per_class * 1.3, NUM_CLASSES * 1.35))
    for c in range(NUM_CLASSES):
        pick = np.sort(rng.choice(np.flatnonzero(labels == c), per_class, replace=False))
        imgs = np.asarray(images[pick])
        for j in range(per_class):
            img = imgs[j]
            if crop:
                lo = (img.shape[0] - crop) // 2
                img = img[lo:lo + crop, lo:lo + crop]
            ax = axes[c, j]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(CLASS_SHORT[c], fontsize=8)
    if crop:
        fig.suptitle(f"center {crop}×{crop} crop", fontsize=9)
    fig.tight_layout()
    return fig


def plot_pairs(images, pairs: list[tuple[int, int]], labels: np.ndarray, title: str = "Duplicate pairs"):
    """Show image pairs side by side, annotated with index and label."""
    import matplotlib.pyplot as plt

    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(3.2, 1.6 * n), squeeze=False)
    for r, (i, j) in enumerate(pairs):
        for col, k in enumerate((i, j)):
            ax = axes[r, col]
            ax.imshow(np.asarray(images[k]))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"#{k} · {CLASS_SHORT[labels[k]]}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    return fig
