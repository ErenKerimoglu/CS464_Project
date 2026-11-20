"""Result files: the format every model writes. Issue #5 (owner: Ege). Schema: PROJECT_PLAN §4.4, results/README.md."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from galaxy.constants import RUNS_DIR


def run_dir(experiment: str, run_name: str, seed: int) -> Path:
    return RUNS_DIR / experiment / run_name / f"seed{seed}"


def save_eval(run_dir: Path, eval_name: str, metrics: dict, probs: np.ndarray, y_true: np.ndarray,
              orig_index: np.ndarray, config: dict, extra: dict | None = None) -> Path:
    """Write ``eval_<eval_name>.json`` + ``preds_<eval_name>.npz``. Works without torch (classical models use it)."""
    raise NotImplementedError("Issue #5")


def load_evals(pattern: str = "**/eval_*.json"):
    """One row per result file with flattened metrics → pandas DataFrame."""
    raise NotImplementedError("Issue #5")
