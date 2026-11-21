"""Download and read Galaxy10 DECaLS. Issue #2 (owner: Eren).

The h5 file has the keys ``images`` (N,256,256,3 uint8), ``ans`` (labels), ``ra``, ``dec``, ``redshift``, ``pxscale``.
"""
from __future__ import annotations

import hashlib
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from galaxy.constants import DATASET_FILENAME, DATASET_MD5, DATASET_SIZE, DATASET_URL, RAW_DIR

_CHUNK = 1 << 20  # 1 MiB


def md5sum(path: Path, chunk: int = 16 * _CHUNK) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify_file(path: Path, size: int | None = DATASET_SIZE, md5: str | None = DATASET_MD5,
                check_md5: bool = True) -> bool:
    """True if ``path`` exists and matches the expected size (and MD5, if ``check_md5``)."""
    path = Path(path)
    if not path.exists() or (size is not None and path.stat().st_size != size):
        return False
    return not (check_md5 and md5) or md5sum(path) == md5


def download_file(url: str, dest: Path, expected_size: int | None = None, progress: bool = True) -> Path:
    """Stream ``url`` to ``dest``. Resumes a partial ``dest.part`` with an HTTP Range request when possible."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    if expected_size is not None and start > expected_size:
        part.unlink()
        start = 0

    req = urllib.request.Request(url, headers={"User-Agent": "cs464-galaxy-downloader"})
    if start:
        req.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(req, timeout=60) as resp:
        if start and resp.status != 206:  # server ignored the Range header: start over
            start = 0
        total = expected_size or (start + int(resp.headers.get("Content-Length", 0))) or None
        bar = None
        if progress:
            from tqdm.auto import tqdm

            bar = tqdm(total=total, initial=start, unit="B", unit_scale=True, desc=dest.name)
        with open(part, "ab" if start else "wb") as f:
            while block := resp.read(_CHUNK):
                f.write(block)
                if bar:
                    bar.update(len(block))
        if bar:
            bar.close()
    part.replace(dest)
    return dest


def download_galaxy10(dest: Path = RAW_DIR, url: str = DATASET_URL, force: bool = False,
                      check_md5: bool = True) -> Path:
    """Download ``Galaxy10_DECals.h5`` into ``dest`` and verify its size and MD5. Returns the file path.

    Skips the download if a verified file already exists, unless ``force``.
    """
    path = Path(dest) / DATASET_FILENAME
    if not force and verify_file(path, check_md5=check_md5):
        print(f"[download] {path} already present and verified")
        return path
    if force and path.exists():
        path.unlink()
    for attempt in range(1, 4):
        try:
            download_file(url, path, expected_size=DATASET_SIZE)
            break
        except OSError as e:  # network hiccup: resume from the .part file
            print(f"[download] attempt {attempt} failed ({e}); resuming")
    else:
        raise RuntimeError("Download failed 3 times; run again to resume")
    if not verify_file(path, check_md5=check_md5):
        bad = path.with_name(path.name + ".corrupt")
        path.replace(bad)
        raise RuntimeError(f"Downloaded file failed verification (size/MD5); moved to {bad}. Run again.")
    print(f"[download] {path} verified (size {DATASET_SIZE:,} B, md5 {DATASET_MD5})")
    return path


def load_metadata(path: Path = RAW_DIR / DATASET_FILENAME) -> pd.DataFrame:
    """Everything except the images, one row per galaxy: orig_index, label, ra, dec, redshift, pxscale."""
    import h5py

    with h5py.File(path, "r") as f:
        meta = pd.DataFrame({
            "orig_index": np.arange(f["ans"].shape[0]),
            "label": np.asarray(f["ans"]).astype(np.int64),
            "ra": np.asarray(f["ra"], dtype=np.float64),
            "dec": np.asarray(f["dec"], dtype=np.float64),
            "redshift": np.asarray(f["redshift"], dtype=np.float64),
            "pxscale": np.asarray(f["pxscale"], dtype=np.float64),
        })
    return meta


@contextmanager
def open_images(path: Path = RAW_DIR / DATASET_FILENAME):
    """``with open_images() as images: images[i]`` → lazy h5py dataset of shape (N,256,256,3) uint8."""
    import h5py

    f = h5py.File(path, "r")
    try:
        yield f["images"]
    finally:
        f.close()


def iter_image_chunks(images, indices: np.ndarray | None = None,
                      chunk: int = 512) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(indices_chunk, images_chunk)`` reading in sorted order (fast for chunked/compressed h5 files)."""
    idx = np.arange(len(images)) if indices is None else np.sort(np.asarray(indices))
    for s in range(0, len(idx), chunk):
        sel = idx[s:s + chunk]
        if len(sel) and sel[-1] - sel[0] == len(sel) - 1:      # contiguous: one slice read
            block = np.asarray(images[sel[0]:sel[-1] + 1])
        else:
            block = np.asarray(images[sel])                    # h5py fancy indexing needs increasing indices
        yield sel, block
