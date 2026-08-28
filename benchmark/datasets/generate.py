"""Deterministic synthetic dataset generation.

Three base datasets, each resembling a plausible real-world table:

- ``customers``: one row per customer, with a binary-ish ``is_active`` column
  and ``customer_id`` as a candidate primary key.
- ``transactions``: one row per transaction, referencing ``customer_id`` (a
  foreign key, deliberately *not* unique), with a rare binary ``is_fraud``
  target.
- ``loans``: one row per loan application, with a binary ``defaulted`` target
  correlated with ``credit_score`` and ``employment_type``.

All three are generated from a single ``np.random.default_rng(SEED)`` stream
consumed in a fixed order (customers, then transactions, then loans), so
re-running ``generate_all`` always produces byte-identical Parquet files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eda_agent.log import get_logger

logger = get_logger(__name__)

SEED = 42
DATA_DIR = Path(__file__).resolve().parent / "data"

_N_CUSTOMERS = 2000
_N_TRANSACTIONS = 8000
_N_LOANS = 3000


def _make_customers(rng: np.random.Generator) -> pd.DataFrame:
    n = _N_CUSTOMERS
    customer_id = np.arange(1, n + 1)
    age = rng.normal(42, 13, n).clip(18, 90).round().astype(int)
    annual_income = rng.lognormal(mean=10.8, sigma=0.45, size=n).round(2)
    signup_channel = rng.choice(
        ["web", "referral", "store", "ad"], size=n, p=[0.45, 0.2, 0.15, 0.2]
    )
    region = rng.choice(["north", "south", "east", "west"], size=n)

    active_logit = 0.5 - 0.02 * (age - 42) + (signup_channel == "referral") * 0.6
    active_prob = 1.0 / (1.0 + np.exp(-active_logit))
    is_active = (rng.random(n) < active_prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": customer_id,
            "age": age,
            "annual_income": annual_income,
            "signup_channel": signup_channel,
            "region": region,
            "is_active": is_active,
        }
    )


def _make_transactions(rng: np.random.Generator, customer_ids: np.ndarray) -> pd.DataFrame:
    n = _N_TRANSACTIONS
    transaction_id = np.arange(1, n + 1)
    customer_id = rng.choice(customer_ids, size=n)
    category = rng.choice(
        ["groceries", "electronics", "travel", "dining", "other"],
        size=n,
        p=[0.3, 0.15, 0.1, 0.25, 0.2],
    )
    channel = rng.choice(["online", "in_store"], size=n, p=[0.6, 0.4])

    base_amount = {
        "groceries": 45.0,
        "electronics": 220.0,
        "travel": 400.0,
        "dining": 35.0,
        "other": 60.0,
    }
    amount = np.array([base_amount[c] for c in category])
    amount = amount * rng.lognormal(mean=0.0, sigma=0.5, size=n)
    amount = amount.round(2)

    fraud_logit = -4.5 + 0.01 * (amount - 60.0) + (channel == "online") * 0.8
    fraud_prob = 1.0 / (1.0 + np.exp(-fraud_logit))
    is_fraud = (rng.random(n) < fraud_prob).astype(int)

    return pd.DataFrame(
        {
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "amount": amount,
            "category": category,
            "channel": channel,
            "is_fraud": is_fraud,
        }
    )


def _make_loans(rng: np.random.Generator) -> pd.DataFrame:
    n = _N_LOANS
    loan_id = np.arange(1, n + 1)
    credit_score = rng.normal(680, 60, n).clip(300, 850).round().astype(int)
    applicant_income = rng.lognormal(mean=10.9, sigma=0.4, size=n).round(2)
    loan_amount = (applicant_income * rng.uniform(0.5, 3.0, n)).round(2)
    loan_term_months = rng.choice([12, 24, 36, 48, 60], size=n)
    employment_type = rng.choice(
        ["salaried", "self_employed", "unemployed"], size=n, p=[0.65, 0.28, 0.07]
    )

    default_logit = (
        -2.0
        - 0.015 * (credit_score - 680)
        + 0.9 * (employment_type == "unemployed")
        + 0.000002 * loan_amount
    )
    default_prob = 1.0 / (1.0 + np.exp(-default_logit))
    defaulted = (rng.random(n) < default_prob).astype(int)

    return pd.DataFrame(
        {
            "loan_id": loan_id,
            "credit_score": credit_score,
            "applicant_income": applicant_income,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term_months,
            "employment_type": employment_type,
            "defaulted": defaulted,
        }
    )


def generate_all(out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Generate the three base datasets and write them to Parquet.

    Deterministic: a single ``np.random.default_rng(SEED)`` stream is consumed
    in a fixed order, so repeated calls produce identical output.
    """
    target_dir = out_dir if out_dir is not None else DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    customers = _make_customers(rng)
    transactions = _make_transactions(rng, customers["customer_id"].to_numpy())
    loans = _make_loans(rng)

    frames = {"customers": customers, "transactions": transactions, "loans": loans}
    for name, frame in frames.items():
        path = target_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        logger.info("wrote %d rows to %s", len(frame), path)

    return frames


if __name__ == "__main__":
    generate_all()
