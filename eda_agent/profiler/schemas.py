from __future__ import annotations

from pydantic import BaseModel


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_fraction: float
    n_unique: int
    quantiles: dict[str, float] | None
    top_values: list[tuple[str, int]] | None
    sentinel_candidates: list[str]


class TableProfile(BaseModel):
    table_name: str
    row_count: int
    columns: list[ColumnProfile]
    candidate_primary_keys: list[str]
