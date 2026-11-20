"""Feature matrix assembly + sklearn preprocessing (PCA on HOG). Issue #14 (owner: Yunus)."""
from __future__ import annotations


def load_feature_matrix(split: str, feature_sets=("hog", "color", "morph"), suffix: str = ""):
    """Join the cached parquet files on ``orig_index``; return ``(X: DataFrame, y: ndarray, orig_index: ndarray)``."""
    raise NotImplementedError("Issue #14")


def build_preprocessor(columns_by_group: dict[str, list[str]]):
    """ColumnTransformer: HOG → StandardScaler → PCA(0.95); color/morph → median imputer → StandardScaler."""
    raise NotImplementedError("Issue #14")
