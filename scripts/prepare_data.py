"""Download → deduplicate → split → band stats → processed image cache. Issues #2, #3 (owner: Eren).

    python scripts/prepare_data.py                    # everything (needs ~9 GB free: 2.7 GB h5 + 3.5 GB cache)
    python scripts/prepare_data.py --download-only    # just download + verify the h5 file
    python scripts/prepare_data.py --no-cache         # split files only, skip writing images_256.npy

Set GALAXY_DATA_DIR to put raw/processed data outside the repo (recommended on Colab/Kaggle and OneDrive).
The split files in data/splits/ are frozen: they are only written if they do not exist yet (or with --force-split).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from galaxy.constants import (
    BAND_STATS_FILE,
    DEDUP_RADIUS_ARCSEC,
    DEDUP_REPORT_FILE,
    NUM_CLASSES,
    PROCESSED_IMAGES,
    PROCESSED_META,
    RAW_DIR,
    REMOVED_FILE,
    SPLIT_FILE,
)
from galaxy.data.dedup import find_duplicates, resolve_duplicates, sensitivity_table
from galaxy.data.download import download_galaxy10, iter_image_chunks, load_metadata, open_images
from galaxy.data.split import compute_band_stats, make_split, save_band_stats, save_split, split_summary


def write_cache(h5_path: Path, out: Path = PROCESSED_IMAGES, chunk: int = 512) -> Path:
    """Copy all images (rows = orig_index) into an uncompressed, memory-mappable .npy file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npy")
    with open_images(h5_path) as images:
        arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=images.shape)
        for sel, block in iter_image_chunks(images, chunk=chunk):
            arr[sel[0]:sel[-1] + 1] = block
            print(f"\r[cache] {sel[-1] + 1}/{len(images)}", end="", flush=True)
        arr.flush()
        del arr
    tmp.replace(out)
    print(f"\n[cache] wrote {out} ({out.stat().st_size / 1e9:.2f} GB)")
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--no-cache", action="store_true", help="do not write data/processed/images_256.npy")
    p.add_argument("--skip-md5", action="store_true", help="verify size only (faster re-runs)")
    p.add_argument("--radius", type=float, default=DEDUP_RADIUS_ARCSEC, help="dedup radius in arcsec")
    p.add_argument("--force-split", action="store_true", help="overwrite the frozen split files (team decision!)")
    p.add_argument("--h5", type=Path, default=None,
                   help="use an existing Galaxy10_DECals.h5 (e.g. a read-only Kaggle input) instead of downloading")
    a = p.parse_args(argv)

    t0 = time.time()
    if a.h5 is not None:
        h5_path = a.h5
        print(f"[download] using {h5_path} (not re-verified)")
    else:
        h5_path = download_galaxy10(RAW_DIR, check_md5=not a.skip_md5)
    if a.download_only:
        return

    meta = load_metadata(h5_path)
    print(f"[meta] {len(meta)} galaxies, class counts {np.bincount(meta.label, minlength=NUM_CLASSES).tolist()}")

    # ---- deduplication
    ra, dec, y = meta.ra.to_numpy(), meta.dec.to_numpy(), meta.label.to_numpy()
    sens = sensitivity_table(ra, dec, y)
    print("[dedup] sensitivity to the matching radius:\n" + sens.to_string(index=False))
    groups = find_duplicates(ra, dec, a.radius)
    kept, removed = resolve_duplicates(groups, y, len(meta))
    print(f"[dedup] radius {a.radius}\": {len(groups)} groups, removed {len(removed)} images "
          f"({(removed.reason == 'duplicate').sum()} duplicates, {(removed.reason == 'label_conflict').sum()} "
          f"from label conflicts) → {len(kept)} kept")

    # ---- split
    split_df = make_split(meta.loc[kept])
    summary = split_summary(split_df)
    print("[split]\n" + summary.to_string())
    max_dev = max((summary[f"{s} %"] - summary["all %"]).abs().max() for s in ("train", "val", "test"))
    print(f"[split] max class-share deviation from overall: {max_dev:.2f} pp")

    if SPLIT_FILE.exists() and not a.force_split:
        existing = pd.read_csv(SPLIT_FILE)
        same = existing[["orig_index", "split"]].equals(split_df[["orig_index", "split"]].reset_index(drop=True))
        print(f"[split] {SPLIT_FILE.name} already exists (frozen) - regenerated split identical: {same}")
        if not same:
            print("[split] WARNING: this environment regenerates a different split (library version?). "
                  "The committed file is the source of truth and is used below. Please report this in issue #3.",
                  file=sys.stderr)
        split_df = existing
    else:
        save_split(split_df, force=a.force_split)
        REMOVED_FILE.parent.mkdir(parents=True, exist_ok=True)
        removed.to_csv(REMOVED_FILE, index=False)
        DEDUP_REPORT_FILE.write_text(json.dumps({
            "radius_arcsec": a.radius, "n_original": len(meta), "n_kept": int(len(kept)),
            "n_groups": len(groups), "n_removed": int(len(removed)),
            "sensitivity": sens.to_dict(orient="records"),
            "split_counts": split_df.split.value_counts().to_dict(),
            "max_class_share_deviation_pp": round(float(max_dev), 3),
        }, indent=2), encoding="utf-8")
        print(f"[split] wrote {SPLIT_FILE}, {REMOVED_FILE}, {DEDUP_REPORT_FILE}")

    # ---- processed cache (also speeds up the band-stats pass)
    if not a.no_cache and not PROCESSED_IMAGES.exists():
        write_cache(h5_path)
    PROCESSED_META.parent.mkdir(parents=True, exist_ok=True)
    split_df.to_csv(PROCESSED_META, index=False)

    # ---- band statistics (train only)
    train_idx = split_df.loc[split_df.split == "train", "orig_index"].to_numpy()
    if PROCESSED_IMAGES.exists():
        images = np.load(PROCESSED_IMAGES, mmap_mode="r")
        stats = compute_band_stats(block for _, block in iter_image_chunks(images, train_idx))
    else:
        with open_images(h5_path) as images:
            stats = compute_band_stats(block for _, block in iter_image_chunks(images, train_idx))
    print(f"[stats] train band mean {stats['mean']} std {stats['std']}")
    if BAND_STATS_FILE.exists() and not a.force_split:
        old = json.loads(BAND_STATS_FILE.read_text(encoding="utf-8"))
        same = np.allclose(old["mean"], stats["mean"], atol=1e-5) and np.allclose(old["std"], stats["std"], atol=1e-5)
        print(f"[stats] {BAND_STATS_FILE.name} already exists (frozen) - identical: {same}")
    else:
        save_band_stats(stats, force=a.force_split)
        print(f"[stats] wrote {BAND_STATS_FILE}")
    print(f"[done] {time.time() - t0:.0f}s. Commit data/splits/*. Raw/processed data stay out of git.")


if __name__ == "__main__":
    main()
