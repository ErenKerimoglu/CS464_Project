"""Result files: the format every model writes. Issue #5 (owner: Ege). Schema: PROJECT_PLAN §4.4, results/README.md.

This module deliberately does **not** import torch: classical models (#14) call :func:`save_eval` with numpy arrays.

Layout::

    results/runs/<experiment>/<run_name>/seed<k>/
        eval_<eval_name>.json    # metrics + full config + git commit
        preds_<eval_name>.npz    # probs [N,C] float16, y_true [N] int64, orig_index [N] int64
        train_log.csv            # deep models (#4)
        train_summary.json       # deep models (#4): train time, peak memory, best epoch
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from galaxy.constants import CLASS_SHORT, RUNS_DIR, SPLIT_VERSION

SPLITS = ("train", "val", "test")
# Keys every ``extra`` block contains (null when not applicable), so all result files have the same columns.
DEFAULT_EXTRA: dict[str, Any] = {"rotation_deg": None, "corruption": None, "severity": None,
                                 "n_params": None, "train_time_s": None, "peak_mem_gb": None}
TRAIN_SUMMARY_FILE = "train_summary.json"
_SEED_DIR = re.compile(r"^seed(\d+)$")


def run_dir(experiment: str, run_name: str, seed: int, root: Path = RUNS_DIR) -> Path:
    """``results/runs/<experiment>/<run_name>/seed<seed>`` (not created)."""
    return Path(root) / experiment / run_name / f"seed{int(seed)}"


def parse_run_dir(path: Path) -> dict:
    """``{"experiment", "run_name", "seed"}`` from a ``.../<experiment>/<run_name>/seed<k>`` path, else ``{}``."""
    path = Path(path)
    m = _SEED_DIR.match(path.name)
    if not m or len(path.parts) < 3:
        return {}
    return {"experiment": path.parent.parent.name, "run_name": path.parent.name, "seed": int(m.group(1))}


def eval_name_for(split: str, rotation_deg: float | None = None, corruption: str | None = None,
                  severity: int | None = None, tta8: bool = False) -> str:
    """Canonical eval names: ``test``, ``test_rot045``, ``test_blur_s2``, ``test_tta8``, ``test_rot045_tta8``."""
    parts = [split]
    if rotation_deg is not None:
        deg = float(rotation_deg) % 360
        parts.append(f"rot{int(deg):03d}" if deg.is_integer() else f"rot{deg:05.1f}".replace(".", "p"))
    if corruption is not None:
        if severity is None:
            raise ValueError("a corruption needs a severity")
        parts.append(f"{corruption}_s{int(severity)}")
    if tta8:
        parts.append("tta8")
    return "_".join(parts)


def split_from_eval_name(eval_name: str) -> str:
    """``test_rot045`` → ``test``; names that do not start with a split (e.g. ``sdss``) map to themselves."""
    head = eval_name.split("_", 1)[0]
    return head if head in SPLITS else eval_name


def model_name_from_config(config: dict) -> str | None:
    """Deep configs have ``model: {name: ...}``; classical configs (#14) have ``model: "xgboost"``."""
    model = (config or {}).get("model")
    if isinstance(model, dict):
        return model.get("name")
    return model if isinstance(model, str) else None


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays, Paths, tuples and sets into JSON types. NaN/inf become null."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Path):
        return obj.as_posix()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    return str(obj)


def _atomic_write(path: Path, write_fn) -> None:
    """Write via a temp file + rename, so an interrupted session never leaves a half-written result file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_eval(run_dir: Path, eval_name: str, metrics: dict, probs: np.ndarray, y_true: np.ndarray,
              orig_index: np.ndarray, config: dict, extra: dict | None = None, split: str | None = None) -> Path:
    """Write ``eval_<eval_name>.json`` + ``preds_<eval_name>.npz`` into ``run_dir`` and return the JSON path.

    Works without torch (classical models use it). ``experiment``/``run_name``/``seed`` are read from the
    ``.../<experiment>/<run_name>/seed<k>`` folder name, falling back to the same keys in ``config``. ``split``
    defaults to the prefix of ``eval_name`` (``test_rot045`` → ``test``).
    """
    run_dir = Path(run_dir)
    probs, y_true, orig_index = np.asarray(probs), np.asarray(y_true), np.asarray(orig_index)
    if probs.ndim != 2 or not (len(probs) == len(y_true) == len(orig_index)):
        raise ValueError(f"shape mismatch: probs {probs.shape}, y_true {y_true.shape}, orig_index {orig_index.shape}")
    if int(metrics.get("n_samples", len(y_true))) != len(y_true):
        raise ValueError(f"metrics.n_samples={metrics['n_samples']} but {len(y_true)} predictions were given")
    config = config or {}
    ids = {k: config.get(k) for k in ("experiment", "run_name", "seed")}
    ids.update(parse_run_dir(run_dir))

    record = {
        "experiment": ids["experiment"],
        "run_name": ids["run_name"],
        "seed": ids["seed"],
        "model": model_name_from_config(config),
        "eval_name": eval_name,
        "split": split or split_from_eval_name(eval_name),
        "split_version": SPLIT_VERSION,
        "git_commit": _git_commit(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "metrics": metrics,
        "extra": {**DEFAULT_EXTRA, **(extra or {})},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / f"eval_{eval_name}.json"
    npz_path = run_dir / f"preds_{eval_name}.npz"

    def write_npz(tmp: str) -> None:
        with open(tmp, "wb") as f:  # a file object stops numpy from appending ".npz" to the temp name
            np.savez_compressed(f, probs=probs.astype(np.float16), y_true=y_true.astype(np.int64),
                                orig_index=orig_index.astype(np.int64))

    _atomic_write(npz_path, write_npz)
    text = json.dumps(to_jsonable(record), indent=2, allow_nan=False) + "\n"
    _atomic_write(json_path, lambda tmp: Path(tmp).write_text(text, encoding="utf-8"))
    return json_path


def _git_commit() -> str:
    from galaxy.utils import git_commit  # local import: keeps this module importable in minimal environments

    return git_commit()


def load_eval(path: Path) -> dict:
    """One ``eval_*.json`` as a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_preds(run_dir: Path, eval_name: str) -> dict[str, np.ndarray]:
    """``{"probs" (float32), "y_true", "orig_index"}`` from ``preds_<eval_name>.npz``."""
    with np.load(Path(run_dir) / f"preds_{eval_name}.npz") as z:
        return {"probs": z["probs"].astype(np.float32), "y_true": z["y_true"], "orig_index": z["orig_index"]}


def load_train_summary(run_dir: Path) -> dict:
    """Contents of ``train_summary.json`` written by the Trainer (#4), or ``{}`` if the run has none."""
    path = Path(run_dir) / TRAIN_SUMMARY_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def flatten_eval(record: dict) -> dict:
    """One flat row: identifiers, scalar metrics, per-class metrics as ``recall_<Class>`` etc., ``extra`` keys."""
    metrics = record.get("metrics", {})
    row = {k: record.get(k) for k in ("experiment", "run_name", "seed", "model", "eval_name", "split",
                                      "split_version", "git_commit", "timestamp")}
    for key in ("accuracy", "macro_f1", "balanced_accuracy", "n_samples"):
        row[key] = metrics.get(key)
    for key, prefix in (("per_class_recall", "recall"), ("per_class_precision", "precision"),
                        ("per_class_f1", "f1")):
        values = metrics.get(key) or []
        names = CLASS_SHORT if len(values) == len(CLASS_SHORT) else [str(i) for i in range(len(values))]
        row.update({f"{prefix}_{name}": v for name, v in zip(names, values, strict=True)})
    row["confusion_matrix"] = metrics.get("confusion_matrix")
    for key, value in (record.get("extra") or {}).items():
        row[key if key not in row else f"extra_{key}"] = value
    row["config"] = record.get("config")
    return row


def load_evals(pattern: str = "**/eval_*.json", root: Path = RUNS_DIR):
    """All result files under ``root`` matching ``pattern`` → pandas DataFrame, one row per file.

    Columns: see :func:`flatten_eval`, plus ``path``. Rows are sorted by experiment, run_name, eval_name, seed.
    """
    import pandas as pd

    rows = []
    for path in sorted(Path(root).glob(pattern)):
        row = flatten_eval(load_eval(path))
        row["path"] = path.as_posix()
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["experiment", "run_name", "seed", "model", "eval_name", "split", "accuracy",
                                     "macro_f1", "balanced_accuracy", "n_samples", "path"])
    df = pd.DataFrame(rows)
    return df.sort_values(["experiment", "run_name", "eval_name", "seed"], na_position="last",
                          key=lambda s: s.astype(str) if s.name != "seed" else s).reset_index(drop=True)
