"""Loading interface for the benchmark's base datasets."""

from __future__ import annotations

import pandas as pd

from benchmark.datasets.generate import DATA_DIR, generate_all

DATASET_NAMES: tuple[str, ...] = ("customers", "transactions", "loans")


def list_datasets() -> list[str]:
    """Return the names of the available base datasets."""
    return list(DATASET_NAMES)


def load_dataset(name: str) -> pd.DataFrame:
    """Load a base dataset by name, generating it on first use if missing.

    Raises:
        ValueError: if ``name`` is not one of ``list_datasets()``.
    """
    if name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(DATASET_NAMES)}")

    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        generate_all()
    return pd.read_parquet(path)
