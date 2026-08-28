from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benchmark.planting import (
    plant_correlated_missingness,
    plant_duplicate_ids,
    plant_leaked_column,
    plant_outlier_cluster,
    plant_sentinel_value,
    plant_simpsons_paradox,
)


@pytest.fixture
def customers_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "age": rng.normal(40, 10, n),
            "annual_income": rng.lognormal(10.5, 0.4, n),
            "region": rng.choice(["north", "south", "east", "west"], n),
            "is_active": rng.integers(0, 2, n),
        }
    )


def test_plant_outlier_cluster_creates_far_values(customers_df):
    modified, ground_truth = plant_outlier_cluster(
        customers_df, "annual_income", n_outliers=15, magnitude=12.0
    )

    assert ground_truth == {"kind": "outlier", "columns": ["annual_income"]}
    mean = customers_df["annual_income"].mean()
    std = customers_df["annual_income"].std()
    z = (modified["annual_income"] - mean).abs() / std
    assert (z > 8).sum() >= 15
    # original frame untouched
    assert customers_df["annual_income"].max() < modified["annual_income"].max()


def test_plant_correlated_missingness_rate_differs_by_group(customers_df):
    modified, ground_truth = plant_correlated_missingness(
        customers_df, "age", "is_active", high_rate=0.8, low_rate=0.01
    )

    assert ground_truth == {"kind": "missingness", "columns": ["age", "is_active"]}
    is_high = modified["is_active"] == modified["is_active"].max()
    rate_high = modified.loc[is_high, "age"].isna().mean()
    rate_low = modified.loc[~is_high, "age"].isna().mean()
    assert rate_high > rate_low
    assert rate_high > 0.5
    assert rate_low < 0.1


def test_plant_simpsons_paradox_reverses_sign(customers_df):
    # give ourselves a clean grouping column with several groups and enough
    # rows per group for a stable within-group correlation estimate.
    modified, ground_truth = plant_simpsons_paradox(customers_df, "age", "annual_income", "region")

    assert ground_truth == {
        "kind": "subgroup_reversal",
        "columns": ["age", "annual_income", "region"],
    }
    aggregate_corr = modified["age"].corr(modified["annual_income"])
    within_group_corrs = modified.groupby("region", observed=True).apply(
        lambda g: g["age"].corr(g["annual_income"])
    )
    assert np.sign(aggregate_corr) != 0
    for corr in within_group_corrs:
        assert np.sign(corr) == -np.sign(aggregate_corr)


def test_plant_leaked_column_is_near_perfectly_correlated(customers_df):
    modified, ground_truth = plant_leaked_column(customers_df, "is_active")

    assert ground_truth == {
        "kind": "leakage",
        "columns": ["leaked_feature", "is_active"],
    }
    assert "leaked_feature" in modified.columns
    corr = modified["leaked_feature"].corr(modified["is_active"].astype(float))
    assert abs(corr) > 0.98


def test_plant_duplicate_ids_creates_repeated_id(customers_df):
    modified, ground_truth = plant_duplicate_ids(customers_df, "customer_id", n_duplicates=5)

    assert ground_truth == {"kind": "duplicate_key", "columns": ["customer_id"]}
    counts = modified["customer_id"].value_counts()
    assert (counts > 1).sum() >= 5
    assert len(modified) == len(customers_df) + 5


def test_plant_sentinel_value_injects_literal(customers_df):
    modified, ground_truth = plant_sentinel_value(customers_df, "age", sentinel=-999, rate=0.1)

    assert ground_truth == {"kind": "sentinel", "columns": ["age"]}
    n_sentinel = int((modified["age"] == -999).sum())
    assert n_sentinel > 0
    # roughly matches requested rate (loose bound given randomness)
    assert n_sentinel < len(customers_df) * 0.3


def test_plant_sentinel_value_low_rate_still_injects_at_least_one(customers_df):
    # even with a tiny rate that could round to zero draws, the function must
    # not silently no-op.
    modified, _ = plant_sentinel_value(customers_df, "age", sentinel=-999, rate=1e-6)
    assert (modified["age"] == -999).sum() >= 1
