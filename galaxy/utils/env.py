"""Where am I running, and where do data and checkpoints live? Issue #10 (owner: Ege).

The runner notebooks call :func:`setup_env` once, **before** launching ``scripts/*.py`` as subprocesses. It sets the
``GALAXY_*`` environment variables that :mod:`galaxy.constants` reads at import time, so every script started from
the notebook uses the right folders. Variables that are already set are never overwritten (env vars override).

=========  ===========================================  ============================================================
platform   images (``GALAXY_PROCESSED_DIR``)             checkpoints (``GALAXY_CKPT_DIR``)
=========  ===========================================  ============================================================
kaggle     attached dataset ``/kaggle/input/**/processed``  ``/kaggle/working/checkpoints`` (saved with the output)
colab      Drive copy → copied to local ``/content/data``   ``<drive_dir>/checkpoints`` (survives the session)
local      ``data/processed`` (``prepare_data.py``)          ``checkpoints/`` in the repo
=========  ===========================================  ============================================================

``<drive_dir>`` defaults to ``/content/drive/MyDrive/CS464_galaxy`` (override: ``GALAXY_DRIVE_DIR``).
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Literal

Platform = Literal["kaggle", "colab", "local"]

KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")
COLAB_CONTENT = Path("/content")
DEFAULT_DRIVE_DIR = COLAB_CONTENT / "drive" / "MyDrive" / "CS464_galaxy"
IMAGES_FILE = "images_256.npy"
ENV_KEYS = ("GALAXY_DATA_DIR", "GALAXY_PROCESSED_DIR", "GALAXY_CKPT_DIR")


def detect_platform() -> Platform:
    """``kaggle`` / ``colab`` / ``local`` (``GALAXY_PLATFORM`` forces a value)."""
    forced = os.environ.get("GALAXY_PLATFORM")
    if forced:
        if forced not in ("kaggle", "colab", "local"):
            raise ValueError(f"GALAXY_PLATFORM must be kaggle, colab or local, got {forced!r}")
        return forced  # type: ignore[return-value]
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or KAGGLE_WORKING.is_dir():
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or importlib.util.find_spec("google.colab") is not None:
        return "colab"
    return "local"


def find_kaggle_processed(input_dir: Path | None = None) -> Path | None:
    """``processed/`` folder of an attached dataset that contains ``images_256.npy`` (any nesting depth ≤ 4)."""
    input_dir = Path(input_dir or KAGGLE_INPUT)
    if not input_dir.is_dir():
        return None
    for pattern in ("*/processed", "*/*/processed", "*/*/*/processed", "*/*/*/*/processed"):
        hits = sorted(p for p in input_dir.glob(pattern) if (p / IMAGES_FILE).exists())
        if hits:
            return hits[0]
    return None


def _copy_if_missing(src_dir: Path, dst_dir: Path, verbose: bool) -> None:
    """Copy ``images_256.npy`` + ``meta.csv`` from Drive to local disk (memmap reads over Drive are very slow)."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in (IMAGES_FILE, "meta.csv"):
        src, dst = src_dir / name, dst_dir / name
        if src.exists() and (not dst.exists() or dst.stat().st_size != src.stat().st_size):
            if verbose:
                print(f"copying {src} → {dst} ({src.stat().st_size / 1e9:.1f} GB) ...", flush=True)
            tmp = dst.with_name(dst.name + ".part")
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)


def platform_paths(platform: Platform, drive_dir: Path | None = None, copy_data_to_local: bool = True,
                   verbose: bool = True) -> dict[str, str]:
    """The ``GALAXY_*`` values for ``platform`` (empty for ``local``: the defaults in ``galaxy.constants`` apply)."""
    if platform == "kaggle":
        paths = {"GALAXY_DATA_DIR": str(KAGGLE_WORKING / "data"),
                 "GALAXY_CKPT_DIR": str(KAGGLE_WORKING / "checkpoints")}
        processed = find_kaggle_processed()
        if processed is not None:
            paths["GALAXY_PROCESSED_DIR"] = str(processed)
        return paths
    if platform == "colab":
        drive = Path(drive_dir or os.environ.get("GALAXY_DRIVE_DIR") or DEFAULT_DRIVE_DIR)
        local_data = COLAB_CONTENT / "data"
        paths = {"GALAXY_DATA_DIR": str(local_data), "GALAXY_CKPT_DIR": str(drive / "checkpoints")}
        local_processed, drive_processed = local_data / "processed", drive / "processed"
        if (local_processed / IMAGES_FILE).exists() and not (drive_processed / IMAGES_FILE).exists():
            paths["GALAXY_PROCESSED_DIR"] = str(local_processed)        # e.g. downloaded with the Kaggle API
        elif (drive_processed / IMAGES_FILE).exists():
            if copy_data_to_local:
                _copy_if_missing(drive_processed, local_processed, verbose)
                paths["GALAXY_PROCESSED_DIR"] = str(local_processed)
            else:
                paths["GALAXY_PROCESSED_DIR"] = str(drive_processed)
        return paths
    return {}


def setup_env(platform: Platform | None = None, drive_dir: str | Path | None = None,
              copy_data_to_local: bool = True, verbose: bool = True) -> dict[str, str]:
    """Detect the platform and export ``GALAXY_DATA_DIR`` / ``GALAXY_PROCESSED_DIR`` / ``GALAXY_CKPT_DIR``.

    Already-set variables win. Returns ``{"platform", "data_dir", "processed_dir", "ckpt_dir"}`` as they will be seen
    by scripts started **after** this call (subprocesses). Modules already imported in the calling process keep the
    values they read at import time.
    """
    platform = platform or detect_platform()
    for key, value in platform_paths(platform, Path(drive_dir) if drive_dir else None, copy_data_to_local,
                                     verbose).items():
        os.environ.setdefault(key, value)
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = Path(os.environ.get("GALAXY_DATA_DIR", repo_root / "data"))
    resolved = {
        "platform": platform,
        "data_dir": str(data_dir),
        "processed_dir": os.environ.get("GALAXY_PROCESSED_DIR", str(data_dir / "processed")),
        "ckpt_dir": os.environ.get("GALAXY_CKPT_DIR", str(repo_root / "checkpoints")),
    }
    if verbose:
        print(describe_env(resolved))
    return resolved


def describe_env(resolved: dict[str, str]) -> str:
    """Human-readable summary: platform, GPU, paths, and whether the image cache was found."""
    lines = [f"platform       {resolved['platform']}"]
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU only)"
        lines.append(f"torch          {torch.__version__} · GPU: {gpu}")
    except ImportError:
        lines.append("torch          not installed")
    images = Path(resolved["processed_dir"]) / IMAGES_FILE
    found = f"found ({images.stat().st_size / 1e9:.1f} GB)" if images.exists() else "MISSING"
    lines += [f"data_dir       {resolved['data_dir']}",
              f"processed_dir  {resolved['processed_dir']}  [{IMAGES_FILE}: {found}]",
              f"ckpt_dir       {resolved['ckpt_dir']}"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- runner helpers (notebooks)
def run_commands(cmds: list[str] | str, cwd: str | Path) -> None:
    """Run shell commands one after another from ``cwd``, streaming their output into the notebook.

    Stops at the first failing command (``subprocess.CalledProcessError``). Child processes inherit the ``GALAXY_*``
    variables exported by :func:`setup_env`.
    """
    import subprocess
    import sys

    for cmd in [cmds] if isinstance(cmds, str) else cmds:
        print(f"\n$ {cmd}", flush=True)
        proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
        if proc.wait() != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)


def export_results(repo_dir: str | Path, dest: str | Path, newer_than: float, zip_it: bool = True) -> list[Path]:
    """Copy every file under ``<repo>/results/{runs,tables,figures}`` modified after ``newer_than`` (a
    ``time.time()`` taken before the runs started) to ``<dest>/results/...`` and optionally zip ``dest`` to
    ``<dest>.zip``. Tables and figures cover scripts that write there, e.g. the rotation sweep (#23).

    Returns the exported run folders (``<experiment>/<run_name>/seed<k>``, relative to ``results/runs``).
    """
    repo_dir, dest = Path(repo_dir), Path(dest)
    runs = repo_dir / "results" / "runs"
    run_dirs: set[Path] = set()
    n_files = 0
    for sub in ("runs", "tables", "figures"):
        folder = repo_dir / "results" / sub
        for f in sorted(folder.rglob("*")) if folder.is_dir() else []:
            if f.is_file() and f.name != ".gitkeep" and f.stat().st_mtime >= newer_than:
                rel = f.relative_to(repo_dir)
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest / rel)
                n_files += 1
                if sub == "runs":
                    run_dirs.add(f.parent.relative_to(runs))
    if zip_it and n_files:
        shutil.make_archive(str(dest), "zip", root_dir=dest)
    return sorted(run_dirs)


def results_instructions(run_dirs: list[Path], archive: str | Path, commit: str) -> str:
    """The git steps for committing exported results (CONTRIBUTING → Results workflow)."""
    if not run_dirs:
        return "No new result files were written in this session."
    experiments = sorted({p.parts[0] for p in run_dirs})
    branch = f"results/{experiments[0]}" if len(experiments) == 1 else "results/<experiment>"
    lines = [f"Exported {len(run_dirs)} run folder(s) (code commit {commit}):"]
    lines += [f"  results/runs/{p.as_posix()}" for p in run_dirs]
    lines += [
        "",
        f"1. Download {archive} (Kaggle: Output tab of the saved version · Colab: it is on your Drive).",
        "2. In your local clone:",
        "     git checkout main && git pull",
        f"     git checkout -b {branch}",
        f"     unzip -o {Path(archive).name} -d .      # extracts into results/runs/...",
        "     git add " + " ".join(f"results/runs/{e}" for e in experiments),
        '     git commit -m "Add <experiment> results (#<issue>)"',
        f"     git push -u origin {branch}",
        "3. Open a PR, and update the run's row in docs/RUNS.md (status, checkpoint location, commit).",
    ]
    return "\n".join(lines)
