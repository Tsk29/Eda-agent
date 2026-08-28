from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Claim(BaseModel):
    text: str
    sql: str
    stated_values: dict[str, float]


class VerifiedClaim(BaseModel):
    claim: Claim
    recomputed_values: dict[str, float]
    passed: bool
    max_relative_error: float


class Finding(BaseModel):
    kind: Literal[
        "missingness",
        "outlier",
        "leakage",
        "correlation",
        "subgroup_reversal",
        "duplicate_key",
        "sentinel",
        "drift",
    ]
    severity: Literal["low", "medium", "high"]
    columns: list[str]
    effect_size: float | None
    p_value: float | None
    p_value_corrected: float | None
    claim: VerifiedClaim
