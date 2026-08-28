"""Target-leakage detection.

Flags any single feature with near-perfect predictive power on a declared
target: AUC > 0.95 for a binary target, or |Pearson correlation| > 0.98 for
either target type. Pure functions, no sklearn dependency (not installed in
this environment) — AUC is computed from the Mann-Whitney U / rank-sum
identity via `scipy.stats.rankdata`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import stats

from eda_agent.log import get_logger

logger = get_logger(__name__)

AUC_THRESHOLD = 0.95
CORRELATION_THRESHOLD = 0.98

TargetType = Literal["binary", "continuous"]


@dataclass(frozen=True)
class LeakageResult:
    """Result of a single feature-vs-target leakage check."""

    auc: float | None
    correlation: float | None
    is_leaky: bool
    reason: str | None


def compute_auc(feature: Sequence[float], target: Sequence[int]) -> float:
    """Compute ROC AUC of `feature` as a binary classifier for `target`.

    Uses the Mann-Whitney U identity:
        AUC = (sum_of_ranks(positive_class) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)
    which is equivalent to sklearn's `roc_auc_score` but needs only scipy.
    Ties in `feature` are broken with average ranks (scipy's default), which
    matches scipy's/sklearn's tie handling for AUC.

    `target` must contain exactly two distinct values (encoded as 0/1, or any
    two values — the larger one is treated as the positive class).

    An AUC below 0.5 means the feature is a near-perfect *inverse* predictor
    and is just as leaky as one above 0.5; callers wanting a leakage flag
    should compare `max(auc, 1 - auc)` against the threshold.
    """
    feature_arr = np.asarray(feature, dtype=float)
    target_arr = np.asarray(target)

    if feature_arr.shape != target_arr.shape:
        raise ValueError("feature and target must be the same length")

    labels = np.unique(target_arr)
    if labels.size != 2:
        raise ValueError(f"target must be binary (found {labels.size} distinct values)")

    positive_label = labels.max()
    is_positive = target_arr == positive_label
    n_pos = int(is_positive.sum())
    n_neg = int((~is_positive).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("target must contain at least one of each class")

    ranks = stats.rankdata(feature_arr)
    sum_ranks_pos = float(ranks[is_positive].sum())
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def detect_leakage(
    feature: Sequence[float],
    target: Sequence[float],
    target_type: TargetType,
) -> LeakageResult:
    """Check a single feature for leakage against a declared target.

    Args:
        feature: Numeric feature values.
        target: Target values. For `target_type="binary"` these must take
            exactly two distinct values; for `"continuous"` any numeric
            values are accepted.
        target_type: Declared type of the target column.

    Returns:
        A LeakageResult with whichever statistics apply (AUC for binary
        targets, Pearson correlation always) and an `is_leaky` flag that is
        True if AUC's discriminative power exceeds 0.95 or |correlation|
        exceeds 0.98.
    """
    feature_arr = np.asarray(feature, dtype=float)
    target_arr = np.asarray(target, dtype=float)

    if feature_arr.shape != target_arr.shape:
        raise ValueError("feature and target must be the same length")
    if feature_arr.size < 2:
        raise ValueError("need at least two observations")

    reasons: list[str] = []
    auc: float | None = None
    correlation: float | None = None

    if target_type == "binary":
        auc = compute_auc(feature_arr, target_arr.astype(int))
        discriminative_power = max(auc, 1.0 - auc)
        if discriminative_power > AUC_THRESHOLD:
            reasons.append(f"AUC {discriminative_power:.4f} exceeds threshold {AUC_THRESHOLD}")

    if np.std(feature_arr) > 0 and np.std(target_arr) > 0:
        correlation = float(stats.pearsonr(feature_arr, target_arr)[0])
        if abs(correlation) > CORRELATION_THRESHOLD:
            reasons.append(
                f"|correlation| {abs(correlation):.4f} exceeds threshold {CORRELATION_THRESHOLD}"
            )

    is_leaky = bool(reasons)
    if is_leaky:
        logger.warning("Leakage detected: %s", "; ".join(reasons))

    return LeakageResult(
        auc=auc,
        correlation=correlation,
        is_leaky=is_leaky,
        reason="; ".join(reasons) or None,
    )
