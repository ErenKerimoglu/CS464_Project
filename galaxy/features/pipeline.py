"""Feature matrix assembly and sklearn preprocessing for classical models."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from galaxy.constants import FEATURES_DIR, SPLIT_FILE
from galaxy.data.split import subset_mask
from galaxy.features.morphology import MORPH_KEYS

FEATURE_SET_NAMES = ("hog", "color", "morph")
_MORPH_COLUMNS = frozenset((*MORPH_KEYS, "morph_failed"))


def _feature_group(column: str) -> str:
    if column.startswith("feature_") and column.removeprefix("feature_").isdigit():
        return "hog"
    if column.startswith(("g_hist_", "r_hist_", "z_hist_", "gr_", "rz_")):
        return "color"
    if column in _MORPH_COLUMNS:
        return "morph"
    raise ValueError(f"Unrecognized feature column: {column}")


def load_feature_matrix(
    split: str,
    feature_sets: Sequence[str] = FEATURE_SET_NAMES,
    suffix: str = "",
    fraction: float = 1.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Join cached features for a split and return ``(X, y, orig_index)`` in split-file order.

    ``suffix`` is appended to each cache name, e.g. ``suffix="_rot45"`` reads
    ``hog_rot45_v1.parquet``. A fractional subset is allowed only for training.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be 'train', 'val' or 'test', got {split!r}")
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    feature_sets = tuple(feature_sets)
    if not feature_sets:
        raise ValueError("feature_sets must contain at least one of 'hog', 'color', or 'morph'")
    if len(set(feature_sets)) != len(feature_sets):
        raise ValueError("feature_sets cannot contain duplicates")
    unknown = set(feature_sets).difference(FEATURE_SET_NAMES)
    if unknown:
        raise ValueError(f"Unknown feature sets: {sorted(unknown)}")
    if suffix and ("/" in suffix or "\\" in suffix):
        raise ValueError("suffix cannot contain path separators")
    cache_suffix = f"_{suffix.strip('_')}" if suffix.strip("_") else ""

    split_df = pd.read_csv(SPLIT_FILE)
    required_split_columns = {"orig_index", "label", "split"}
    if not required_split_columns.issubset(split_df.columns):
        raise ValueError(f"{SPLIT_FILE} must contain {sorted(required_split_columns)}")
    if split_df["orig_index"].duplicated().any():
        raise ValueError(f"{SPLIT_FILE} contains duplicate orig_index values")
    selected = split_df.loc[subset_mask(split_df, split, fraction), ["orig_index", "label"]]
    if selected.empty:
        raise ValueError(f"No samples selected for split={split!r}, fraction={fraction}")
    orig_index = selected["orig_index"].to_numpy(dtype=np.int64)
    y = selected["label"].to_numpy(dtype=np.int64)

    feature_frames = []
    seen_columns: set[str] = set()
    for feature_set in feature_sets:
        cache_path = FEATURES_DIR / f"{feature_set}{cache_suffix}_v1.parquet"
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"Feature cache not found: {cache_path}. "
                f"Create it with scripts/extract_features.py --features {feature_set}."
            )
        frame = pd.read_parquet(cache_path)
        if "orig_index" not in frame.columns:
            raise ValueError(f"Feature cache is missing orig_index: {cache_path}")
        if frame.columns.duplicated().any():
            raise ValueError(f"Feature cache contains duplicate column names: {cache_path}")
        if frame["orig_index"].duplicated().any():
            raise ValueError(f"Feature cache contains duplicate orig_index values: {cache_path}")

        columns = [column for column in frame.columns if column != "orig_index"]
        if not columns:
            raise ValueError(f"Feature cache contains no features: {cache_path}")
        overlap = seen_columns.intersection(columns)
        if overlap:
            raise ValueError(f"Feature caches contain duplicate columns: {sorted(overlap)}")
        for column in columns:
            _feature_group(column)
        seen_columns.update(columns)

        indexed = frame.set_index("orig_index")
        missing = pd.Index(orig_index).difference(indexed.index)
        if not missing.empty:
            preview = missing[:5].tolist()
            raise ValueError(
                f"Feature cache {cache_path} is missing {len(missing)} selected samples; "
                f"first orig_index values: {preview}"
            )
        feature_frames.append(indexed.loc[orig_index, columns].reset_index(drop=True))

    X = pd.concat(feature_frames, axis=1)
    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="raise")

    groups: dict[str, list[str]] = {name: [] for name in FEATURE_SET_NAMES}
    for column in X.columns:
        groups[_feature_group(column)].append(column)
    for group in ("hog", "color"):
        if groups[group] and not np.isfinite(X[groups[group]].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{group} features contain non-finite values")
    if groups["morph"]:
        X[groups["morph"]] = X[groups["morph"]].replace([np.inf, -np.inf], np.nan)

    return X, y, orig_index


def feature_columns_by_group(columns: Sequence[str]) -> dict[str, list[str]]:
    """Group feature column names for ``build_preprocessor``."""
    groups: dict[str, list[str]] = {name: [] for name in FEATURE_SET_NAMES}
    for column in columns:
        group = _feature_group(column)
        if column in groups[group]:
            raise ValueError(f"Duplicate feature column: {column}")
        groups[group].append(column)
    return groups


def build_preprocessor(columns_by_group: Mapping[str, Sequence[str]]) -> ColumnTransformer:
    """Scale HOG before PCA(0.95); impute and scale color/morphology columns."""
    unknown_groups = set(columns_by_group).difference(FEATURE_SET_NAMES)
    if unknown_groups:
        raise ValueError(f"Unknown feature groups: {sorted(unknown_groups)}")

    groups = {name: list(columns_by_group.get(name, ())) for name in FEATURE_SET_NAMES}
    all_columns = [column for columns in groups.values() for column in columns]
    if not all_columns:
        raise ValueError("At least one feature column is required")
    if len(set(all_columns)) != len(all_columns):
        raise ValueError("Feature columns must belong to exactly one group")
    for group, columns in groups.items():
        mismatched = [column for column in columns if _feature_group(column) != group]
        if mismatched:
            raise ValueError(f"Columns do not belong to group {group!r}: {mismatched}")

    transformers = []
    if groups["hog"]:
        transformers.append((
            "hog",
            Pipeline((
                ("scale", StandardScaler()),
                ("pca", PCA(n_components=0.95, svd_solver="full")),
            )),
            groups["hog"],
        ))

    numeric_columns = groups["color"] + groups["morph"]
    if numeric_columns:
        transformers.append((
            "numeric",
            Pipeline((
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            )),
            numeric_columns,
        ))

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
