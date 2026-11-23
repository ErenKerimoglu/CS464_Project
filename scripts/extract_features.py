"""Extract and cache hand-crafted features."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from galaxy.constants import CLASS_SHORT, FIGURES_DIR, SPLIT_FILE  # noqa: E402
from galaxy.features.cache import extract_all  # noqa: E402
from galaxy.features.common import center_crop_np, rotate_np, to_float01  # noqa: E402
from galaxy.features.hog_color import color_features, hog_features  # noqa: E402
from galaxy.features.morphology import MORPH_KEYS, morph_features  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract and cache hand-crafted galaxy features")
    parser.add_argument("--features", nargs="+", choices=("hog", "color", "morph"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--rotate", type=float, default=0.0, metavar="DEG")
    parser.add_argument("--corruption", choices=("none", "noise", "blur", "downsample"), default="none")
    parser.add_argument("--severity", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--morph-timeout", type=float, default=60.0, metavar="SECONDS")
    return parser


def _cache_name(
    feature: str,
    rotate_deg: float,
    corruption: str,
    severity: int,
    split: str = "all",
) -> str:
    name = feature
    if split != "all":
        name += f"_{split}"
    if rotate_deg != 0:
        name += f"_rot{int(round(rotate_deg)) % 360:03d}"
    if corruption != "none":
        name += f"_{corruption}_s{severity}"
    return name


def _prepare_image(image: np.ndarray) -> np.ndarray:
    return center_crop_np(to_float01(image), 180)


def _report_morphology(frame: pd.DataFrame, elapsed: float) -> None:
    split_df = pd.read_csv(SPLIT_FILE)[["orig_index", "label"]]
    results = split_df.merge(
        frame[["orig_index", "morph_failed", "gini", "m20", "concentration", "asymmetry"]],
        on="orig_index",
        how="inner",
    )
    results["morph_failed"] = results["morph_failed"].fillna(1.0)
    print(f"morph: {len(frame)} images in {elapsed:.2f}s")
    print("Morphology failure rate by class:")
    for label, short_name in enumerate(CLASS_SHORT):
        subset = results.loc[results["label"] == label, "morph_failed"]
        if subset.empty:
            print(f"  {short_name}: n/a (0/0)")
        else:
            print(f"  {short_name}: {float(subset.mean()):.2%} ({int(subset.sum())}/{len(subset)})")

    import matplotlib.pyplot as plt

    valid = results.loc[(results["morph_failed"] == 0) & results["gini"].notna() & results["m20"].notna()]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = plt.get_cmap("tab10")
    for label, short_name in enumerate(CLASS_SHORT):
        class_rows = valid[valid["label"] == label]
        if class_rows.empty:
            continue
        axes[0].scatter(
            class_rows["gini"],
            class_rows["m20"],
            s=8,
            alpha=0.45,
            color=colors(label),
            label=short_name,
            rasterized=True,
        )
        class_rows = class_rows[class_rows["concentration"].notna() & class_rows["asymmetry"].notna()]
        axes[1].scatter(
            class_rows["concentration"],
            class_rows["asymmetry"],
            s=8,
            alpha=0.45,
            color=colors(label),
            label=short_name,
            rasterized=True,
        )

    axes[0].set(xlabel="Gini", ylabel="M20", title="Gini–M20 by class")
    axes[1].set(xlabel="Concentration", ylabel="Asymmetry", title="Concentration vs asymmetry")
    axes[0].legend(title="Class", fontsize=7, ncol=2)
    axes[1].legend(title="Class", fontsize=7, ncol=2)
    output = FIGURES_DIR / "classical" / "morph_by_class.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved morphology figure: {output}")


def main() -> None:
    args = build_parser().parse_args()
    if args.corruption != "none":
        raise SystemExit("corruption extraction is implemented with issue #22; use no corruption for issue #7")
    if "morph" in args.features and args.rotate != 0:
        raise SystemExit("morphology rotation sweeps are not supported by this extractor")

    split_df = pd.read_csv(SPLIT_FILE)
    if args.split == "all":
        indices = split_df["orig_index"].to_numpy(np.int64)
    else:
        indices = split_df.loc[split_df["split"] == args.split, "orig_index"].to_numpy(np.int64)

    for feature in args.features:
        fn = {"hog": hog_features, "color": color_features, "morph": morph_features}[feature]
        cache_name = _cache_name(feature, args.rotate, args.corruption, args.severity, args.split)

        def transform(image: np.ndarray, feature_name: str = feature) -> np.ndarray:
            if feature_name == "morph":
                return image
            return rotate_np(image, args.rotate) if args.rotate != 0 else _prepare_image(image)

        start = time.perf_counter()
        frame = extract_all(
            cache_name,
            fn,
            indices=indices,
            n_jobs=args.n_jobs,
            force=args.force,
            image_transform=transform,
            timeout_seconds=args.morph_timeout if feature == "morph" else None,
            timeout_result={**dict.fromkeys(MORPH_KEYS, np.nan), "morph_failed": 1.0}
            if feature == "morph" else None,
        )
        elapsed = time.perf_counter() - start
        if feature == "morph":
            _report_morphology(frame, elapsed)
        else:
            print(
                f"{cache_name}: {len(frame)} rows in {elapsed:.2f}s "
                f"-> data/features/{cache_name}_v1.parquet"
            )


if __name__ == "__main__":
    main()
