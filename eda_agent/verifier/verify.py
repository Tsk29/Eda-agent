"""Independent re-verification of claims.

A `Claim` states a piece of prose together with the SQL that allegedly
produced it and the numeric values that were stated. `verify_claim`
re-executes that SQL through a caller-supplied `run_sql` callable and
compares the freshly computed values against what was stated. It never
trusts the LLM's arithmetic and never substitutes a "corrected" value back
into the claim: a mismatch is the finding, not something to repair.

`run_sql` is a minimal structural dependency (`Callable[[str], list[dict]]`)
so this module has zero import-time coupling to the real executor, which is
built independently. Any row-returning callable with that shape works,
including fakes in tests.
"""

from __future__ import annotations

from collections.abc import Callable

from eda_agent.log import get_logger
from eda_agent.schemas import Claim, VerifiedClaim

logger = get_logger(__name__)

# Relative tolerance for floating-point (non-count) comparisons, per
# CLAUDE.md's verification rules.
_RELATIVE_TOLERANCE = 1e-6

# Floor used as the denominator when the stated value is at or near zero, to
# avoid dividing by zero (or by a vanishingly small number that would make
# the relative error blow up for a practically-irrelevant discrepancy).
# Below this magnitude we treat "abs(stated)" as this epsilon instead.
_EPSILON = 1e-12


def _relative_error(stated: float, recomputed: float) -> float:
    denominator = max(abs(stated), _EPSILON)
    return abs(stated - recomputed) / denominator


def _reject(claim: Claim, recomputed_values: dict[str, float]) -> VerifiedClaim:
    return VerifiedClaim(
        claim=claim,
        recomputed_values=recomputed_values,
        passed=False,
        max_relative_error=0.0,
    )


def verify_claim(claim: Claim, run_sql: Callable[[str], list[dict]]) -> VerifiedClaim:
    """Re-execute `claim.sql` and check it against `claim.stated_values`.

    Never raises: any failure mode (SQL error, wrong row count, missing key,
    or an out-of-tolerance value) results in a `VerifiedClaim` with
    `passed=False`, never an exception. `stated_values` is never mutated or
    replaced with the recomputed number — both are carried independently so
    the mismatch itself is visible as the result.
    """
    try:
        rows = run_sql(claim.sql)
    except Exception:  # noqa: BLE001 - any executor failure rejects the claim
        logger.warning("verify_claim: run_sql raised for sql=%r", claim.sql, exc_info=True)
        return _reject(claim, {})

    if len(rows) != 1:
        logger.warning(
            "verify_claim: expected exactly 1 row, got %d for sql=%r",
            len(rows),
            claim.sql,
        )
        return _reject(claim, {})

    row = rows[0]

    recomputed_values: dict[str, float] = {}
    for key in claim.stated_values:
        if key not in row:
            logger.warning(
                "verify_claim: key %r missing from result row for sql=%r", key, claim.sql
            )
            return _reject(claim, recomputed_values)
        recomputed_values[key] = float(row[key])

    passed = True
    max_relative_error = 0.0

    for key, stated in claim.stated_values.items():
        recomputed = recomputed_values[key]

        if float(stated).is_integer():
            # Count-like value: exact match required, no tolerance applies.
            if recomputed != stated:
                passed = False
            continue

        relative_error = _relative_error(stated, recomputed)
        max_relative_error = max(max_relative_error, relative_error)
        if relative_error > _RELATIVE_TOLERANCE:
            passed = False

    return VerifiedClaim(
        claim=claim,
        recomputed_values=recomputed_values,
        passed=passed,
        max_relative_error=max_relative_error,
    )
