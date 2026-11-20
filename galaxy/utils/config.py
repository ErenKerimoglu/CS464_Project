"""YAML configuration with inheritance and command-line overrides.

A config file may contain ``defaults: [path, ...]`` (relative to ``configs/``, or to the file itself); those files
are loaded first and deep-merged in order, then the file's own keys win. Overrides use dotted keys::

    cfg = load_config("configs/experiments/e1_orientation/resnet50_none.yaml",
                      overrides=["train.lr=3e-4", "data.fraction=0.25", "preprocess.policy=none"])
    cfg["train"]["lr"]  # -> 0.0003

Values in overrides are parsed with YAML, so ``1e-3`` is a float, ``true`` is a bool and ``[1,2]`` is a list.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from galaxy.constants import CONFIGS_DIR


def deep_merge(base: dict, other: dict) -> dict:
    """Recursively merge ``other`` into a copy of ``base`` (``other`` wins)."""
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(ref: str, current: Path) -> Path:
    for cand in (current.parent / ref, CONFIGS_DIR / ref, Path(ref)):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"Config '{ref}' referenced from {current} not found")


def _load_file(path: Path, _stack: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in _stack:
        raise ValueError(f"Circular config defaults: {' -> '.join(map(str, _stack + (path,)))}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    merged: dict = {}
    for ref in raw.pop("defaults", []) or []:
        merged = deep_merge(merged, _load_file(_resolve(ref, path), _stack + (path,)))
    return deep_merge(merged, raw)


def set_by_dotted_key(cfg: dict, key: str, value: Any) -> None:
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise KeyError(f"Cannot set '{key}': '{p}' is not a mapping")
    node[leaf] = value


def parse_value(raw: str) -> Any:
    """YAML-parse a CLI value; also accept ``1e-3``-style floats, which YAML 1.1 would leave as strings."""
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got '{item}'")
        key, raw = item.split("=", 1)
        set_by_dotted_key(cfg, key.strip(), parse_value(raw))
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    """Load a YAML config (with ``defaults`` inheritance) and apply dotted ``key=value`` overrides."""
    cfg = _load_file(Path(path))
    cfg = apply_overrides(cfg, overrides or [])
    cfg.setdefault("_meta", {})["config_path"] = str(path)
    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
