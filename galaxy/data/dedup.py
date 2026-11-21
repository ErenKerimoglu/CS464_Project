"""Duplicate removal by sky coordinates. Issue #3 (owner: Eren). Protocol: PROJECT_PLAN §3.1.

Positions are converted to unit vectors, and neighbours are found with a KD-tree on the chord distance
``2·sin(θ/2)``. That is exactly equivalent to an angular-separation search such as astropy's ``search_around_sky``,
but needs only scipy and takes well under a second for the whole dataset.

Policy: a group of positions within ``radius`` of each other (connected components) is one physical galaxy.
* every copy has the same label → keep the lowest ``orig_index``, drop the rest (reason ``duplicate``)
* the labels disagree → drop **all** copies (reason ``label_conflict``): we cannot know which label is right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from galaxy.constants import DEDUP_RADIUS_ARCSEC

ARCSEC = np.pi / (180 * 3600)


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _chord(arcsec: float) -> float:
    return 2 * np.sin(arcsec * ARCSEC / 2)


def angular_sep_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    """Great-circle separation (Vincenty formula, stable at small and large angles)."""
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    dra = ra2 - ra1
    num = np.hypot(np.cos(dec2) * np.sin(dra),
                   np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra))
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    return np.arctan2(num, den) / ARCSEC


def find_pairs(ra: np.ndarray, dec: np.ndarray, radius_arcsec: float = DEDUP_RADIUS_ARCSEC) -> np.ndarray:
    """All index pairs (i<j) within ``radius_arcsec``, shape (P, 2)."""
    tree = cKDTree(_unit_vectors(ra, dec))
    pairs = tree.query_pairs(_chord(radius_arcsec), output_type="ndarray")
    return pairs if len(pairs) else np.empty((0, 2), dtype=np.int64)


def find_duplicates(ra: np.ndarray, dec: np.ndarray, radius_arcsec: float = DEDUP_RADIUS_ARCSEC) -> list[list[int]]:
    """Groups (connected components, size ≥ 2) of indices whose positions chain within ``radius_arcsec``."""
    n = len(ra)
    pairs = find_pairs(ra, dec, radius_arcsec)
    if not len(pairs):
        return []
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, comp = connected_components(graph, directed=False)
    in_pair = np.unique(pairs)
    groups: dict[int, list[int]] = {}
    for i in in_pair:
        groups.setdefault(int(comp[i]), []).append(int(i))
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def nearest_neighbor_sep_arcsec(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    """Separation of every galaxy to its nearest other galaxy (for the EDA histogram)."""
    tree = cKDTree(_unit_vectors(ra, dec))
    dist, _ = tree.query(_unit_vectors(ra, dec), k=2)
    return 2 * np.arcsin(np.clip(dist[:, 1] / 2, 0, 1)) / ARCSEC


def resolve_duplicates(groups: list[list[int]], labels: np.ndarray, n: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply the policy (module docstring). Returns ``(kept_indices, removed)``.

    ``removed`` has one row per dropped image: ``orig_index, group_id, reason, label, kept_index``
    (``kept_index`` = the copy that was kept, or -1 for label conflicts).
    """
    labels = np.asarray(labels)
    keep = np.ones(n, dtype=bool)
    rows = []
    for gid, group in enumerate(groups):
        group = sorted(group)
        if len(set(labels[group].tolist())) == 1:
            for i in group[1:]:
                keep[i] = False
                rows.append((i, gid, "duplicate", int(labels[i]), group[0]))
        else:
            for i in group:
                keep[i] = False
                rows.append((i, gid, "label_conflict", int(labels[i]), -1))
    removed = pd.DataFrame(rows, columns=["orig_index", "group_id", "reason", "label", "kept_index"])
    return np.flatnonzero(keep), removed


def sensitivity_table(ra: np.ndarray, dec: np.ndarray, labels: np.ndarray,
                      radii_arcsec=(0.5, 1.0, 2.0, 5.0)) -> pd.DataFrame:
    """How the dedup outcome depends on the matching radius (reported in the PR / report)."""
    rows = []
    for r in radii_arcsec:
        groups = find_duplicates(ra, dec, r)
        _, removed = resolve_duplicates(groups, labels, len(ra))
        rows.append({
            "radius_arcsec": r,
            "groups": len(groups),
            "images_in_groups": sum(len(g) for g in groups),
            "removed_duplicate": int((removed.reason == "duplicate").sum()),
            "removed_conflict": int((removed.reason == "label_conflict").sum()),
            "conflict_groups": int(removed.loc[removed.reason == "label_conflict", "group_id"].nunique()),
        })
    return pd.DataFrame(rows)
