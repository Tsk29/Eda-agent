from __future__ import annotations

from benchmark.scoring import finding_matches_defect, score
from eda_agent.schemas import Claim, Finding, VerifiedClaim


def _finding(kind: str, columns: list[str]) -> Finding:
    claim = Claim(text="dummy claim", sql="SELECT 1", stated_values={"x": 1.0})
    verified = VerifiedClaim(
        claim=claim, recomputed_values={"x": 1.0}, passed=True, max_relative_error=0.0
    )
    return Finding(
        kind=kind,
        severity="medium",
        columns=columns,
        effect_size=None,
        p_value=None,
        p_value_corrected=None,
        claim=verified,
    )


def test_finding_matches_defect_uses_subset_not_exact_match():
    defect = {"kind": "leakage", "columns": ["leaked_feature"]}
    exact = _finding("leakage", ["leaked_feature"])
    superset = _finding("leakage", ["leaked_feature", "target"])
    wrong_kind = _finding("outlier", ["leaked_feature"])
    wrong_columns = _finding("leakage", ["other_column"])

    assert finding_matches_defect(defect, exact)
    assert finding_matches_defect(defect, superset)
    assert not finding_matches_defect(defect, wrong_kind)
    assert not finding_matches_defect(defect, wrong_columns)


def test_score_fully_detected_no_false_positives():
    planted = [
        {"kind": "outlier", "columns": ["age"]},
        {"kind": "sentinel", "columns": ["income"]},
    ]
    findings_on_planted = [
        _finding("outlier", ["age"]),
        _finding("sentinel", ["income"]),
    ]
    result = score(planted, findings_on_planted, findings_on_clean=[])

    assert result["recall"] == 1.0
    assert result["false_positive_rate"] == 0.0


def test_score_partially_detected():
    planted = [
        {"kind": "outlier", "columns": ["age"]},
        {"kind": "sentinel", "columns": ["income"]},
    ]
    # only the outlier defect is found; sentinel is missed entirely
    findings_on_planted = [_finding("outlier", ["age"])]
    result = score(planted, findings_on_planted, findings_on_clean=[])

    assert result["recall"] == 0.5
    assert result["false_positive_rate"] == 0.0


def test_score_false_positive_present_in_clean_baseline_not_counted():
    planted = [{"kind": "outlier", "columns": ["age"]}]
    baseline_noise = _finding("correlation", ["region", "income"])
    findings_on_planted = [
        _finding("outlier", ["age"]),
        baseline_noise,
    ]
    findings_on_clean = [baseline_noise]  # same kind+columns fires without any defect

    result = score(planted, findings_on_planted, findings_on_clean)

    assert result["recall"] == 1.0
    # 1 finding out of 2 doesn't match a defect, but it matches clean-baseline
    # noise exactly, so it must not be counted as a false positive.
    assert result["false_positive_rate"] == 0.0


def test_score_false_positive_not_in_clean_baseline_is_counted():
    planted = [{"kind": "outlier", "columns": ["age"]}]
    spurious = _finding("correlation", ["region", "income"])
    findings_on_planted = [
        _finding("outlier", ["age"]),
        spurious,
    ]
    findings_on_clean: list[Finding] = []  # spurious finding never fires on clean data

    result = score(planted, findings_on_planted, findings_on_clean)

    assert result["recall"] == 1.0
    assert result["false_positive_rate"] == 0.5  # 1 false positive out of 2 findings


def test_score_zero_planted_defects_recall_is_one():
    result = score([], findings_on_planted=[], findings_on_clean=[])
    assert result["recall"] == 1.0
    assert result["false_positive_rate"] == 0.0


def test_score_zero_findings_on_planted_false_positive_rate_is_zero():
    planted = [{"kind": "outlier", "columns": ["age"]}]
    result = score(planted, findings_on_planted=[], findings_on_clean=[])
    assert result["recall"] == 0.0
    assert result["false_positive_rate"] == 0.0
