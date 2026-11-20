"""statmorph morphology statistics. Issue #11 (owner: Yunus). Spec: PROJECT_PLAN §3.3."""
from __future__ import annotations

import numpy as np

MORPH_KEYS = (
    "concentration", "asymmetry", "smoothness", "gini", "m20",
    "gini_m20_bulge", "gini_m20_merger", "ellipticity_asymmetry", "elongation_asymmetry",
    "r20", "r80", "flag",
)


def morph_features(img: np.ndarray) -> dict[str, float]:
    """statmorph on the background-subtracted r-band crop. Returns NaNs + ``morph_failed=1`` on failure."""
    raise NotImplementedError("Issue #11")
