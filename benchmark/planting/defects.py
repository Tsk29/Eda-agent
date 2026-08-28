"""Defect planting: one function per benchmark defect type.

Each function takes a clean ``pd.DataFrame`` and returns
``(modified_df, ground_truth)`` where ``ground_truth`` is
``{"kind": <Finding.kind literal>, "columns": [...]}``.

Every function asserts, on the DataFrame it is about to return, that the
defect it claims to have planted is actually present. A planting function
that silently no-ops is worse than useless for a benchmark: it would make the
ground truth a lie.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from eda_agent.log import get_logger

logger = get_logger(__name__)


def plant_outlier_cluster(
    df: pd.DataFrame,
    column: str,
    *,
    n_outliers: int = 20,
    magnitude: float = 10.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Push a cluster of rows in ``column`` far from the rest of the distribution.

    ``magnitude`` is expressed in standard deviations from the column mean.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    n_outliers = min(n_outliers, n)
    if n_outliers == 0:
        raise ValueError("cannot plant outliers into an empty DataFrame")

    idx = rng.choice(out.index.to_numpy(), size=n_outliers, replace=False)
    col_mean = out[column].mean()
    col_std = out[column].std()
    outlier_value = col_mean + magnitude * col_std
    out.loc[idx, column] = outlier_value

    z_scores = (out[column] - col_mean).abs() / col_std
    n_far = int((z_scores > magnitude - 1).sum())
    assert n_far >= n_outliers, (
        f"expected at least {n_outliers} extreme values in {column!r}, found {n_far}"
    )

    ground_truth = {"kind": "outlier", "columns": [column]}
    return out, ground_truth


def plant_correlated_missingness(
    df: pd.DataFrame,
    column: str,
    target_column: str,
    *,
    high_rate: float = 0.7,
    low_rate: float = 0.02,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Null out ``column`` at a rate that depends on ``target_column``.

    Rows where ``target_column`` equals its max value (the "positive" class
    for a binary 0/1 target) get nulled at ``high_rate``; all other rows get
    nulled at ``low_rate``. This assumes ``target_column`` is binary or at
    least has a meaningful max-value subgroup.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)

    is_high = (out[target_column] == out[target_column].max()).to_numpy()
    draw = rng.random(n)
    missing_mask = np.where(is_high, draw < high_rate, draw < low_rate)
    out.loc[missing_mask, column] = np.nan

    null_rate_high = out.loc[is_high, column].isna().mean()
    null_rate_low = out.loc[~is_high, column].isna().mean()
    assert null_rate_high > null_rate_low, (
        f"missingness in {column!r} is not correlated with {target_column!r}: "
        f"high-group rate {null_rate_high}, low-group rate {null_rate_low}"
    )

    ground_truth = {"kind": "missingness", "columns": [column, target_column]}
    return out, ground_truth


def plant_simpsons_paradox(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    group_column: str,
    *,
    within_group_slope: float = -1.0,
    group_offset_step: float = 5.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Overwrite ``x_column``/``y_column`` so the aggregate x-y relationship's
    sign is the opposite of the within-group relationship's sign.

    Classic construction: within each group of ``group_column``, x and y are
    negatively related (slope ``within_group_slope`` < 0 by default). Groups
    are laid out at increasing x/y offsets, so between-group variation
    dominates and the pooled (aggregate) correlation comes out positive.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)

    codes = out[group_column].astype("category").cat.codes.to_numpy().astype(float)
    n_groups = len(np.unique(codes))
    if n_groups < 2:
        raise ValueError(f"{group_column!r} needs at least 2 distinct groups")

    local_x = rng.uniform(0.0, 1.0, n)
    x = codes + local_x
    y = within_group_slope * local_x + group_offset_step * codes + rng.normal(0.0, noise_std, n)

    out[x_column] = x
    out[y_column] = y

    aggregate_corr = out[x_column].corr(out[y_column])
    within_group_corrs = out.groupby(group_column, observed=True).apply(
        lambda g: g[x_column].corr(g[y_column])
    )
    within_group_sign = np.sign(within_group_corrs.dropna()).unique()
    assert len(within_group_sign) == 1, "within-group correlation sign is not consistent"
    assert np.sign(aggregate_corr) != within_group_sign[0], (
        f"no sign reversal: aggregate corr {aggregate_corr}, "
        f"within-group signs {within_group_corrs.to_dict()}"
    )

    ground_truth = {
        "kind": "subgroup_reversal",
        "columns": [x_column, y_column, group_column],
    }
    return out, ground_truth


def plant_leaked_column(
    df: pd.DataFrame,
    target_column: str,
    *,
    leak_column: str = "leaked_feature",
    noise_std: float = 0.01,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add a new column that is a near-perfect predictor of ``target_column``.

    Implemented as the target plus a small amount of Gaussian noise, which
    keeps the leak column continuous (as a real leaked feature often is)
    while driving |correlation| with the target above the 0.98 guardrail
    threshold from CLAUDE.md.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    target = out[target_column].astype(float)
    noise = rng.normal(0.0, noise_std, len(out))
    out[leak_column] = target + noise

    corr = out[leak_column].corr(target)
    assert abs(corr) > 0.98, f"leak correlation too low: {corr}"

    ground_truth = {"kind": "leakage", "columns": [leak_column, target_column]}
    return out, ground_truth


def plant_duplicate_ids(
    df: pd.DataFrame,
    id_column: str,
    *,
    n_duplicates: int = 10,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Duplicate some existing id values in ``id_column``.

    Picks ``n_duplicates`` distinct existing ids and appends one extra copy
    of a row for each, so those ids now occur more than once.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    n_duplicates = min(n_duplicates, out[id_column].nunique())
    if n_duplicates == 0:
        raise ValueError("cannot duplicate ids in an empty DataFrame")

    ids_to_dup = rng.choice(out[id_column].unique(), size=n_duplicates, replace=False)
    dup_rows = out[out[id_column].isin(ids_to_dup)].drop_duplicates(subset=id_column, keep="first")
    out = pd.concat([out, dup_rows], ignore_index=True)

    counts = out[id_column].value_counts()
    n_actually_duplicated = int((counts > 1).sum())
    assert n_actually_duplicated >= n_duplicates, (
        f"expected at least {n_duplicates} duplicated ids in {id_column!r}, "
        f"found {n_actually_duplicated}"
    )

    ground_truth = {"kind": "duplicate_key", "columns": [id_column]}
    return out, ground_truth


def plant_sentinel_value(
    df: pd.DataFrame,
    column: str,
    *,
    sentinel: float = -999,
    rate: float = 0.05,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inject a sentinel literal into ``column`` at roughly ``rate`` of rows."""
    out = df.copy()
    rng = np.random.default_rng(seed)
    n = len(out)
    mask = rng.random(n) < rate
    if not mask.any():
        mask[rng.integers(0, n)] = True
    out.loc[mask, column] = sentinel

    n_sentinel = int((out[column] == sentinel).sum())
    assert n_sentinel > 0, f"no sentinel values were injected into {column!r}"

    ground_truth = {"kind": "sentinel", "columns": [column]}
    return out, ground_truth
