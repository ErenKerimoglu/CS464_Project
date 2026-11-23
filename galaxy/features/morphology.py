"""statmorph morphology statistics. Issue #11 (owner: Yunus). Spec: PROJECT_PLAN §3.3."""
from __future__ import annotations

import logging

import numpy as np

from galaxy.features.common import center_crop_np, to_float01
from galaxy.features.segmentation import estimate_background, segment_central_source

MORPH_KEYS = (
    "concentration", "asymmetry", "smoothness", "gini", "m20",
    "gini_m20_bulge", "gini_m20_merger", "ellipticity_asymmetry", "elongation_asymmetry",
    "r20", "r80", "flag",
)

logger = logging.getLogger(__name__)


def _failed_features() -> dict[str, float]:
    return {**dict.fromkeys(MORPH_KEYS, np.nan), "morph_failed": 1.0}


def morph_features(img: np.ndarray) -> dict[str, float]:
    """Measure central-source morphology on the background-subtracted 180×180 r-band crop.

    Per-image measurement errors are logged and represented by NaNs with ``morph_failed=1``.
    """
    import statmorph

    arr = to_float01(np.asarray(img))
    if arr.ndim == 2:
        rgb_crop = None
        r_band = center_crop_np(arr, min(180, *arr.shape))
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        rgb_crop = center_crop_np(arr, min(180, *arr.shape[:2]))
        r_band = rgb_crop[..., 1]
    elif arr.ndim == 3 and arr.shape[0] == 3:
        chw_crop = center_crop_np(arr, min(180, *arr.shape[1:]))
        r_band = chw_crop[1]
    else:
        raise ValueError("img must be an H×W r-band image or an RGB H×W×3 / 3×H×W image")

    try:
        background, _ = estimate_background(r_band)
        background_subtracted = np.nan_to_num(r_band - background, nan=0.0, posinf=0.0, neginf=0.0)
        segmap, central_label, _ = segment_central_source(r_band)
        if central_label == 0:
            logger.warning("Morphology failed: no central source detected")
            return _failed_features()

        sources = statmorph.source_morphology(background_subtracted, segmap, gain=1.0)
        source = next((item for item in sources if int(item.label) == central_label), None)
        if source is None:
            logger.warning("Morphology failed: statmorph returned no source for label %d", central_label)
            return _failed_features()

        features = {}
        for key in MORPH_KEYS:
            value = getattr(source, key)
            features[key] = float(value) if value is not None and np.isfinite(value) else np.nan
        features["morph_failed"] = 0.0
        return features
    except ImportError:
        raise
    except Exception:
        logger.exception("Morphology measurement failed for one image")
        return _failed_features()
