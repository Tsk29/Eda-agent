"""Unit tests for the pure (non-Streamlit) helper logic in app/logic.py.

These test only plain functions -- no Streamlit server, no DB, no rendering.
"""

from __future__ import annotations

from app.logic import compute_rejection_rate, sort_findings


class TestComputeRejectionRate:
    def test_mix_of_pass_and_fail(self) -> None:
        claims = [
            {"passed": True},
            {"passed": False},
            {"passed": False},
            {"passed": True},
        ]
        assert compute_rejection_rate(claims) == 0.5

    def test_all_passed(self) -> None:
        claims = [{"passed": True}, {"passed": True}]
        assert compute_rejection_rate(claims) == 0.0

    def test_all_failed(self) -> None:
        claims = [{"passed": False}, {"passed": False}, {"passed": False}]
        assert compute_rejection_rate(claims) == 1.0

    def test_zero_claims_does_not_divide_by_zero(self) -> None:
        assert compute_rejection_rate([]) == 0.0


class TestSortFindings:
    def test_sorts_by_severity_high_to_low(self) -> None:
        findings = [
            {"severity": "low", "p_value_corrected": 0.01},
            {"severity": "high", "p_value_corrected": 0.5},
            {"severity": "medium", "p_value_corrected": 0.2},
        ]
        result = sort_findings(findings)
        assert [f["severity"] for f in result] == ["high", "medium", "low"]

    def test_ties_within_severity_sorted_by_p_value_ascending(self) -> None:
        findings = [
            {"severity": "high", "p_value_corrected": 0.3},
            {"severity": "high", "p_value_corrected": 0.01},
            {"severity": "high", "p_value_corrected": 0.1},
        ]
        result = sort_findings(findings)
        assert [f["p_value_corrected"] for f in result] == [0.01, 0.1, 0.3]

    def test_mixed_severities_with_tied_p_values(self) -> None:
        findings = [
            {"severity": "medium", "p_value_corrected": 0.05},
            {"severity": "high", "p_value_corrected": 0.05},
            {"severity": "low", "p_value_corrected": 0.05},
            {"severity": "high", "p_value_corrected": 0.01},
        ]
        result = sort_findings(findings)
        assert [(f["severity"], f["p_value_corrected"]) for f in result] == [
            ("high", 0.01),
            ("high", 0.05),
            ("medium", 0.05),
            ("low", 0.05),
        ]

    def test_none_p_value_corrected_sorts_last_within_severity(self) -> None:
        findings = [
            {"severity": "high", "p_value_corrected": None},
            {"severity": "high", "p_value_corrected": 0.02},
        ]
        result = sort_findings(findings)
        assert [f["p_value_corrected"] for f in result] == [0.02, None]

    def test_empty_list(self) -> None:
        assert sort_findings([]) == []
