"""Scoring: recall and false-positive rate for a benchmark run.

Two different notions of "the same finding" are used here, deliberately:

1. **Detection match** (``finding_matches_defect``): a finding detects a
   planted defect if ``finding.kind == defect["kind"]`` and the defect's
   declared columns are a **subset** of the finding's reported columns. We
   use subset match rather than exact-set match because a finding is allowed
   to be more descriptive than the minimal ground truth — e.g. a
   ``subgroup_reversal`` defect declares ``[x, y, group]`` but a leakage
   defect might reasonably be reported alongside the target column even if
   the ground truth only requires the leaking column. Subset match rewards
   a finding for naming at least the right columns without penalizing it for
   naming more.

2. **Baseline-equivalence match** (``_finding_signature``): to decide whether
   a finding on the defected dataset is genuine baseline noise (and therefore
   not a false positive), we require an **exact** match on ``(kind, columns)``
   against findings produced on the clean dataset. This is intentionally
   stricter than the subset rule above: baseline noise should look literally
   identical to what would have been reported anyway, not just "related".
"""

from __future__ import annotations

from eda_agent.schemas import Finding


def finding_matches_defect(defect: dict, finding: Finding) -> bool:
    """True if ``finding`` counts as detecting ``planted defect``.

    Match rule: same ``kind``, and ``defect["columns"]`` is a subset of
    ``finding.columns`` (see module docstring for rationale).
    """
    if finding.kind != defect["kind"]:
        return False
    return set(defect["columns"]).issubset(set(finding.columns))


def _finding_signature(finding: Finding) -> tuple[str, tuple[str, ...]]:
    """Exact-match key used only to detect baseline noise (see module docstring)."""
    return (finding.kind, tuple(sorted(finding.columns)))


def score(
    planted_defects: list[dict],
    findings_on_planted: list[Finding],
    findings_on_clean: list[Finding],
) -> dict:
    """Score one benchmark run.

    Args:
        planted_defects: ground-truth dicts, each ``{"kind": ..., "columns": [...]}``.
        findings_on_planted: findings produced when running against the
            defected dataset.
        findings_on_clean: findings produced when running against the clean
            (un-defected) baseline dataset, used to filter out findings that
            would have fired anyway.

    Returns:
        ``{"recall": float, "false_positive_rate": float}``.

    Edge cases:
        - Zero planted defects: recall is defined as ``1.0`` (vacuously
          true — there was nothing to miss).
        - Zero findings on the defected dataset: false_positive_rate is
          defined as ``0.0`` (there is nothing to be a false positive).
    """
    detected = sum(
        1
        for defect in planted_defects
        if any(finding_matches_defect(defect, f) for f in findings_on_planted)
    )
    recall = detected / len(planted_defects) if planted_defects else 1.0

    clean_signatures = {_finding_signature(f) for f in findings_on_clean}
    false_positives = 0
    for finding in findings_on_planted:
        if any(finding_matches_defect(d, finding) for d in planted_defects):
            continue  # it detects a real defect, not a false positive
        if _finding_signature(finding) in clean_signatures:
            continue  # baseline noise, present even without defects
        false_positives += 1

    false_positive_rate = false_positives / len(findings_on_planted) if findings_on_planted else 0.0

    return {"recall": recall, "false_positive_rate": false_positive_rate}
