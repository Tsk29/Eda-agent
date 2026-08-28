from __future__ import annotations

import pytest

from eda_agent.schemas import Claim
from eda_agent.verifier import verify_claim


def make_claim(sql: str, stated_values: dict[str, float]) -> Claim:
    return Claim(text="claim text {value}", sql=sql, stated_values=stated_values)


def test_exact_match_passes() -> None:
    claim = make_claim("SELECT AVG(x) AS mean FROM t", {"mean": 3.0})

    def run_sql(_sql: str) -> list[dict]:
        return [{"mean": 3.0}]

    result = verify_claim(claim, run_sql)

    assert result.passed is True
    assert result.max_relative_error == pytest.approx(0.0)
    assert result.recomputed_values == {"mean": 3.0}
    assert result.claim.stated_values == {"mean": 3.0}


def test_null_recomputed_value_rejected_not_raised() -> None:
    # A legitimate SQL result (e.g. AVG() over zero matching rows) can be
    # NULL. That must reject the claim as unverifiable, never crash.
    claim = make_claim("SELECT AVG(x) AS mean FROM t WHERE 1=0", {"mean": 3.0})

    def run_sql(_sql: str) -> list[dict]:
        return [{"mean": None}]

    result = verify_claim(claim, run_sql)

    assert result.passed is False
    assert result.recomputed_values == {}


def test_float_within_tolerance_passes() -> None:
    stated = 1000.5
    # relative error ~5e-7, well under 1e-6 tolerance
    recomputed = stated * (1 + 5e-7)
    claim = make_claim("SELECT AVG(x) AS mean FROM t", {"mean": stated})

    def run_sql(_sql: str) -> list[dict]:
        return [{"mean": recomputed}]

    result = verify_claim(claim, run_sql)

    assert result.passed is True


def test_float_beyond_tolerance_fails() -> None:
    stated = 1000.5
    # relative error ~1e-4, well over 1e-6 tolerance
    recomputed = stated * (1 + 1e-4)
    claim = make_claim("SELECT AVG(x) AS mean FROM t", {"mean": stated})

    def run_sql(_sql: str) -> list[dict]:
        return [{"mean": recomputed}]

    result = verify_claim(claim, run_sql)

    assert result.passed is False
    expected_rel_error = abs(stated - recomputed) / abs(stated)
    assert result.max_relative_error == pytest.approx(expected_rel_error)
    # stated_values must never be mutated/replaced
    assert result.claim.stated_values == {"mean": stated}
    assert result.recomputed_values == {"mean": recomputed}


def test_integral_count_off_by_one_fails_exact_match_required() -> None:
    claim = make_claim("SELECT COUNT(*) AS n FROM t", {"n": 100.0})

    def run_sql(_sql: str) -> list[dict]:
        return [{"n": 101.0}]

    result = verify_claim(claim, run_sql)

    assert result.passed is False


def test_run_sql_raises_exception_rejected_wholesale() -> None:
    claim = make_claim("SELECT AVG(x) AS mean FROM t", {"mean": 3.0})

    def run_sql(_sql: str) -> list[dict]:
        raise RuntimeError("boom")

    result = verify_claim(claim, run_sql)

    assert result.passed is False
    assert result.recomputed_values == {}


def test_missing_key_in_result_rejected_wholesale() -> None:
    claim = make_claim(
        "SELECT AVG(x) AS mean, COUNT(*) AS n FROM t", {"mean": 3.0, "n": 10.0}
    )

    def run_sql(_sql: str) -> list[dict]:
        return [{"mean": 3.0}]

    result = verify_claim(claim, run_sql)

    assert result.passed is False
    assert result.recomputed_values == {"mean": 3.0}


def test_zero_rows_rejected_wholesale() -> None:
    claim = make_claim("SELECT AVG(x) AS mean FROM t WHERE 1=0", {"mean": 3.0})

    def run_sql(_sql: str) -> list[dict]:
        return []

    result = verify_claim(claim, run_sql)

    assert result.passed is False
    assert result.recomputed_values == {}


def test_multiple_rows_rejected_wholesale() -> None:
    claim = make_claim("SELECT x AS mean FROM t", {"mean": 3.0})

    def run_sql(_sql: str) -> list[dict]:
        return [{"mean": 3.0}, {"mean": 4.0}]

    result = verify_claim(claim, run_sql)

    assert result.passed is False
    assert result.recomputed_values == {}


def test_near_zero_stated_value_does_not_divide_by_zero() -> None:
    # A non-integral, near-zero stated value exercises the epsilon-guarded
    # relative-error branch (not the exact-match "count" branch) without
    # raising ZeroDivisionError.
    stated = 1e-10
    claim = make_claim("SELECT CORR(x, y) AS corr FROM t", {"corr": stated})

    def run_sql(_sql: str) -> list[dict]:
        return [{"corr": stated}]

    result = verify_claim(claim, run_sql)

    assert result.passed is True
    assert result.max_relative_error == pytest.approx(0.0)


def test_near_zero_stated_value_with_tiny_mismatch_fails_gracefully() -> None:
    # stated is near-zero (but non-integral, so it's on the float/relative
    # tolerance path) and recomputed is meaningfully non-zero; must not raise
    # ZeroDivisionError and must fail the check.
    claim = make_claim("SELECT CORR(x, y) AS corr FROM t", {"corr": 1e-10})

    def run_sql(_sql: str) -> list[dict]:
        return [{"corr": 0.5}]

    result = verify_claim(claim, run_sql)

    assert result.passed is False
    assert result.max_relative_error > 0.0


def test_max_relative_error_across_multiple_float_keys() -> None:
    claim = make_claim(
        "SELECT AVG(x) AS mean, STDDEV(x) AS std FROM t",
        {"mean": 10.5, "std": 2.5},
    )
    # mean off by 1e-4 relative, std off by 1e-3 relative -> max should be ~1e-3
    def run_sql(_sql: str) -> list[dict]:
        return [{"mean": 10.5 * (1 + 1e-4), "std": 2.5 * (1 + 1e-3)}]

    result = verify_claim(claim, run_sql)

    assert result.passed is False
    assert result.max_relative_error == pytest.approx(1e-3, rel=1e-2)
