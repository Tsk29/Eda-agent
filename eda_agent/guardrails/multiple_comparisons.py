"""Benjamini-Hochberg multiple-comparison correction.

Deterministic, pure. Every p-value produced during a run must pass through
this function before any finding is reported as significant. The LLM never
sees or controls this step.
"""

from __future__ import annotations

from dataclasses import dataclass

from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class MultipleComparisonsResult:
    """Raw and BH-corrected p-values, in the same order as the input."""

    raw_p_values: list[float]
    corrected_p_values: list[float]
    rejected: list[bool]


def apply_benjamini_hochberg(
    p_values: list[float], alpha: float = 0.05
) -> MultipleComparisonsResult:
    """Apply Benjamini-Hochberg FDR correction across all p-values in a run.

    Args:
        p_values: Raw p-values, one per statistical test performed this run.
            Order is preserved in the result.
        alpha: Family-wise false discovery rate to control.

    Returns:
        A MultipleComparisonsResult holding the raw p-values, the BH-corrected
        p-values (same order), and a per-value reject-null flag at `alpha`.
    """
    if not p_values:
        return MultipleComparisonsResult(raw_p_values=[], corrected_p_values=[], rejected=[])

    for p in p_values:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value out of range [0, 1]: {p}")

    rejected, corrected, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")

    return MultipleComparisonsResult(
        raw_p_values=list(p_values),
        corrected_p_values=[float(v) for v in corrected],
        rejected=[bool(v) for v in rejected],
    )
