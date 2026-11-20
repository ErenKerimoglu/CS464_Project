"""Shared helper so placeholder scripts fail with a pointer to the issue that implements them."""
import sys


def not_implemented(script: str, issue: int) -> None:
    sys.exit(f"{script} is not implemented yet. See GitHub issue #{issue} and docs/issues/.")
