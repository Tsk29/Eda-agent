"""Subgroup-reversal (Simpson's paradox) detection.

For a reported aggregate relationship between two numeric variables, tests
whether the sign of that relationship flips within any major subgroup.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from eda_agent.log import get_logger

logger = get_logger(__name__)

MIN_GROUP_SIZE = 2


@dataclass(frozen=True)
class SubgroupReversalResult:
    """Overall vs. per-subgroup relationship signs."""

    overall_sign: int
    overall_correlation: float
    subgroup_signs: dict[str, int]
    subgroup_correlations: dict[str, float]
    reversed_subgroups: list[str]
    is_simpsons_paradox_risk: bool


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def detect_subgroup_reversal(
    x: Sequence[float],
    y: Sequence[float],
    groups: Sequence[str],
) -> SubgroupReversalResult:
    """Check whether the sign of the x-y relationship flips within subgroups.

    The overall and per-subgroup relationships are each summarized by the
    sign of the Pearson correlation between `x` and `y`. Subgroups smaller
    than `MIN_GROUP_SIZE` are skipped (too few points to assign a sign) and
    a subgroup whose correlation is exactly zero is treated as neutral, not
    a reversal.

    Args:
        x: Numeric values of the first variable in the reported relationship.
        y: Numeric values of the second variable in the reported relationship.
        groups: Subgroup label for each observation, same length as x and y.

    Returns:
        A SubgroupReversalResult listing every subgroup's sign and flagging
        `is_simpsons_paradox_risk` if any subgroup disagrees in sign with the
        overall relationship.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    group_arr = np.asarray(groups)

    if not (len(x_arr) == len(y_arr) == len(group_arr)):
        raise ValueError("x, y, and groups must be the same length")
    if len(x_arr) < 2:
        raise ValueError("need at least two observations")

    overall_correlation = float(stats.pearsonr(x_arr, y_arr)[0])
    overall_sign = _sign(overall_correlation)

    subgroup_signs: dict[str, int] = {}
    subgroup_correlations: dict[str, float] = {}
    reversed_subgroups: list[str] = []

    for group_value in sorted(set(group_arr.tolist()), key=str):
        mask = group_arr == group_value
        if int(mask.sum()) < MIN_GROUP_SIZE:
            continue
        group_x, group_y = x_arr[mask], y_arr[mask]
        if np.std(group_x) == 0 or np.std(group_y) == 0:
            continue
        group_correlation = float(stats.pearsonr(group_x, group_y)[0])
        group_sign = _sign(group_correlation)
        key = str(group_value)
        subgroup_signs[key] = group_sign
        subgroup_correlations[key] = group_correlation
        if group_sign != 0 and overall_sign != 0 and group_sign != overall_sign:
            reversed_subgroups.append(key)

    is_risk = bool(reversed_subgroups)
    if is_risk:
        logger.warning("Simpson's paradox risk in subgroups: %s", reversed_subgroups)

    return SubgroupReversalResult(
        overall_sign=overall_sign,
        overall_correlation=overall_correlation,
        subgroup_signs=subgroup_signs,
        subgroup_correlations=subgroup_correlations,
        reversed_subgroups=reversed_subgroups,
        is_simpsons_paradox_risk=is_risk,
    )
