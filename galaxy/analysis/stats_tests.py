"""Significance tests for model comparisons. Issue #16 (owner: Ege)."""
from __future__ import annotations


def mcnemar(pred_a, pred_b, y_true) -> dict:
    """McNemar's test (exact binomial when there are few discordant pairs) → {"b01", "b10", "p_value"}."""
    raise NotImplementedError("Issue #16")


def paired_bootstrap(probs_a, probs_b, y_true, metric: str = "macro_f1", n: int = 2000, seed: int = 0) -> dict:
    """→ {"delta", "ci_low", "ci_high"} for metric(a) − metric(b)."""
    raise NotImplementedError("Issue #16")
