"""Pure (Streamlit-free) helper logic for the app layer.

Kept separate from app/main.py so it can be unit tested without a
Streamlit runtime, a database, or any other side effect.
"""

from __future__ import annotations

from typing import Any

_SEVERITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}


def compute_rejection_rate(claims: list[dict[str, Any]]) -> float:
    """Fraction of claims with passed=False.

    Returns 0.0 when there are no claims, rather than dividing by zero.
    """
    if not claims:
        return 0.0
    rejected = sum(1 for claim in claims if not claim["passed"])
    return rejected / len(claims)


def sort_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort findings by severity (high -> medium -> low), then by
    p_value_corrected ascending. Findings with p_value_corrected=None sort
    last within their severity bucket.
    """

    def sort_key(finding: dict[str, Any]) -> tuple[int, float]:
        severity_rank = _SEVERITY_ORDER.get(finding["severity"], len(_SEVERITY_ORDER))
        p_value = finding.get("p_value_corrected")
        p_rank = p_value if p_value is not None else float("inf")
        return (severity_rank, p_rank)

    return sorted(findings, key=sort_key)
