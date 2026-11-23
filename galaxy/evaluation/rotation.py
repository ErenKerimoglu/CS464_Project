"""Test-time rotation for the orientation sweep (E1b). Issue #23 (owner: Eren).

Geometry is the same as in training (§3.1): the **full 256×256** image is rotated, then ``GalaxyPreprocess`` takes the
180 center crop, resizes and normalizes it. 180 < 256/√2, so no padded corner ever enters the crop. Multiples of 90°
use exact ``rot90`` (no interpolation), and 0° is the identity, so ``test_rot000`` equals the plain ``test`` result.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from torch import Tensor

from galaxy.constants import ROTATION_ANGLES
from galaxy.data.transforms import rotate_batch


def make_rotation_fn(angle_deg: float) -> Callable[[Tensor], Tensor]:
    """Return a ``transform_fn`` for ``predict()`` that rotates every raw 256 image by ``angle_deg``.

    Input and output: float [0,1] batches ``(B,3,256,256)``. Positive angles rotate counter-clockwise as displayed
    (the same convention as ``rotate_batch`` and ``np.rot90``).
    """
    angle = float(angle_deg)
    if angle % 360 == 0:
        def identity(x: Tensor) -> Tensor:
            return x

        return identity

    def rotate(x: Tensor) -> Tensor:
        return rotate_batch(x, angle, crop=None, exact_if_possible=True)

    return rotate


def parse_angles(spec: str | Sequence[float] | None) -> list[float]:
    """``"0:360:15"`` (start:stop:step, stop exclusive) or ``"0,45,90"`` → list of angles. ``None`` → the protocol's
    24 angles (``ROTATION_ANGLES``)."""
    if spec is None:
        return [float(a) for a in ROTATION_ANGLES]
    if not isinstance(spec, str):
        return [float(a) for a in spec]
    if ":" in spec:
        parts = [float(p) for p in spec.split(":")]
        if len(parts) != 3 or parts[2] <= 0:
            raise ValueError(f"angle range must be start:stop:step with step > 0, got {spec!r}")
        return [float(a) for a in np.arange(parts[0], parts[1], parts[2])]
    return [float(a) for a in spec.split(",") if a.strip()]


# ----------------------------------------------------------------------------- results
def sweep_table(evals: pd.DataFrame) -> pd.DataFrame:
    """Rotation-sweep rows of ``load_evals()`` → one row per (experiment, run_name, tta, angle) with the mean ± std
    over seeds of accuracy and macro-F1 (``n_seeds`` included)."""
    if evals.empty or "rotation_deg" not in evals:
        return pd.DataFrame(columns=["experiment", "run_name", "tta", "angle", "acc_mean", "acc_std", "f1_mean",
                                     "f1_std", "n_seeds"])
    df = evals[evals["rotation_deg"].notna() & (evals["split"] == "test")].copy()
    if "corruption" in df:
        df = df[df["corruption"].isna()]
    df["angle"] = df["rotation_deg"].astype(float) % 360
    df["tta"] = df["tta"].fillna("none") if "tta" in df else "none"
    out = (df.groupby(["experiment", "run_name", "tta", "angle"])
             .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"), f1_mean=("macro_f1", "mean"),
                  f1_std=("macro_f1", "std"), n_seeds=("seed", "nunique"))
             .reset_index())
    return out.fillna({"acc_std": 0.0, "f1_std": 0.0})


def summarize_sweep(table: pd.DataFrame) -> pd.DataFrame:
    """Per (experiment, run_name, tta): mean accuracy over angles, worst-case accuracy (and its angle), and the
    **orientation sensitivity** = std of the accuracy over angles. Also the mean macro-F1 over angles."""
    rows = []
    for (exp, run, tta), g in table.groupby(["experiment", "run_name", "tta"]):
        worst = g.loc[g["acc_mean"].idxmin()]
        rows.append({"experiment": exp, "run_name": run, "tta": tta, "n_angles": len(g),
                     "n_seeds": int(g["n_seeds"].min()), "acc_mean_over_angles": g["acc_mean"].mean(),
                     "acc_worst": worst["acc_mean"], "worst_angle": worst["angle"],
                     "orientation_sensitivity": g["acc_mean"].std(ddof=0),
                     "f1_mean_over_angles": g["f1_mean"].mean()})
    return pd.DataFrame(rows).sort_values("acc_mean_over_angles", ascending=False, ignore_index=True)


def plot_accuracy_vs_angle(table: pd.DataFrame, out_path: Path, metric: str = "acc", title: str | None = None):
    """Accuracy (or ``metric="f1"``) vs. test rotation angle, mean ± std band over seeds, one line per run/TTA
    variant. Dotted lines mark multiples of 90°, where rotations are exact. Writes ``.pdf`` and ``.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    for (_exp, run, tta), g in table.groupby(["experiment", "run_name", "tta"]):
        g = g.sort_values("angle")
        # close the curve at 360° so it reads as periodic
        ang = np.append(g["angle"].to_numpy(), 360.0)
        mean = np.append(g[f"{metric}_mean"].to_numpy(), g[f"{metric}_mean"].iloc[0])
        std = np.append(g[f"{metric}_std"].to_numpy(), g[f"{metric}_std"].iloc[0])
        label = run if tta == "none" else f"{run} + TTA-8"
        line, = ax.plot(ang, mean, marker="o", ms=3, lw=1.5, label=label, ls="-" if tta == "none" else "--")
        if (g["n_seeds"] > 1).any():
            ax.fill_between(ang, mean - std, mean + std, color=line.get_color(), alpha=0.15, lw=0)
    for a in (90, 180, 270):
        ax.axvline(a, color="0.6", ls=":", lw=0.8)
    ax.set(xlim=(0, 360), xticks=range(0, 361, 45), xlabel="test rotation angle (°)",
           ylabel="test accuracy" if metric == "acc" else "test macro-F1", title=title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(out_path.with_suffix(suffix), dpi=200)
    plt.close(fig)
    return out_path.with_suffix(".pdf")
