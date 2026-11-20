"""Shared utilities: seeding, configuration, git provenance."""
from __future__ import annotations

import subprocess
from functools import lru_cache

from galaxy.constants import REPO_ROOT
from galaxy.utils.config import load_config, save_config
from galaxy.utils.seed import seed_everything

__all__ = ["git_commit", "load_config", "save_config", "seed_everything"]


@lru_cache(maxsize=1)
def git_commit() -> str:
    """Short hash of HEAD, with a ``-dirty`` suffix if there are uncommitted changes; ``unknown`` outside git."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
