"""Extract and cache hand-crafted features."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from galaxy.constants import SPLIT_FILE  # noqa: E402
from galaxy.features.cache import extract_all  # noqa: E402
from galaxy.features.common import center_crop_np, rotate_np, to_float01  # noqa: E402
from galaxy.features.hog_color import color_features, hog_features  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and cache hand-crafted galaxy features")
    parser.add_argument("--features", nargs="+", choices=("hog", "color", "morph"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--rotate", type=float, default=0.0, metavar="DEG")
    parser.add_argument("--corruption", choices=("none", "noise", "blur", "downsample"), default="none")
    parser.add_argument("--severity", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def _cache_name(feature: str, rotate_deg: float, corruption: str, severity: int) -> str:
    name = feature
    if rotate_deg != 0:
        name += f"_rot{int(round(rotate_deg)) % 360:03d}"
    if corruption != "none":
        name += f"_{corruption}_s{severity}"
    return name


def _prepare_image(image: np.ndarray) -> np.ndarray:
    return center_crop_np(to_float01(image), 180)


def main() -> None:
    args = build_parser().parse_args()
    if "morph" in args.features:
        raise SystemExit("morph extraction belongs to issue #11 and is not implemented yet")
    if args.corruption != "none":
        raise SystemExit("corruption extraction is implemented with issue #22; use no corruption for issue #7")

    split_df = pd.read_csv(SPLIT_FILE)
    if args.split == "all":
        indices = split_df["orig_index"].to_numpy(np.int64)
    else:
        indices = split_df.loc[split_df["split"] == args.split, "orig_index"].to_numpy(np.int64)

    for feature in args.features:
        fn = hog_features if feature == "hog" else color_features
        cache_name = _cache_name(feature, args.rotate, args.corruption, args.severity)

        def transform(image: np.ndarray) -> np.ndarray:
            raw = rotate_np(image, args.rotate) if args.rotate != 0 else _prepare_image(image)
            return raw

        start = time.perf_counter()
        frame = extract_all(
            cache_name,
            fn,
            indices=indices,
            n_jobs=args.n_jobs,
            force=args.force,
            image_transform=transform,
        )
        elapsed = time.perf_counter() - start
        print(f"{cache_name}: {len(frame)} rows in {elapsed:.2f}s -> data/features/{cache_name}_v1.parquet")


if __name__ == "__main__":
    main()
