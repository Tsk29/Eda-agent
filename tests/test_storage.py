"""Tests for eda_agent.storage.db.

Production target is Postgres 16 (see docker-compose.yml); these tests
substitute a temporary on-disk SQLite database for dependency-free testing,
since SQLAlchemy Core table definitions/inserts/selects are dialect-agnostic.
"""

from __future__ import annotations

from pathlib import Path

from eda_agent.schemas import Claim, Finding, VerifiedClaim
from eda_agent.storage.db import (
    create_all,
    get_claims,
    get_engine,
    get_findings,
    save_finding,
    save_run,
    save_verified_claim,
)


def _engine(tmp_path: Path):
    engine = get_engine(f"sqlite:///{tmp_path}/test.db")
    create_all(engine)
    return engine


def test_create_all_creates_all_tables(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    from sqlalchemy import inspect

    table_names = set(inspect(engine).get_table_names())
    assert {"runs", "claims", "findings"}.issubset(table_names)


def test_save_run_and_get_claims_roundtrip(tmp_path: Path) -> None:
    engine = _engine(tmp_path)

    run_id = save_run(engine, dataset_name="titanic.csv", model_name="qwen2.5-coder:14b")
    assert isinstance(run_id, int)

    passing_claim = VerifiedClaim(
        claim=Claim(
            text="Column age has {n} nulls",
            sql="SELECT COUNT(*) AS n FROM t WHERE age IS NULL",
            stated_values={"n": 177.0},
        ),
        recomputed_values={"n": 177.0},
        passed=True,
        max_relative_error=0.0,
    )
    failing_claim = VerifiedClaim(
        claim=Claim(
            text="Column fare has {n} nulls",
            sql="SELECT COUNT(*) AS n FROM t WHERE fare IS NULL",
            stated_values={"n": 5.0},
        ),
        recomputed_values={"n": 0.0},
        passed=False,
        max_relative_error=1.0,
    )

    passing_id = save_verified_claim(engine, run_id, passing_claim)
    failing_id = save_verified_claim(engine, run_id, failing_claim)
    assert isinstance(passing_id, int)
    assert isinstance(failing_id, int)
    assert passing_id != failing_id

    rows = get_claims(engine, run_id)
    assert len(rows) == 2

    by_text = {row["text"]: row for row in rows}

    good = by_text["Column age has {n} nulls"]
    assert good["sql"] == "SELECT COUNT(*) AS n FROM t WHERE age IS NULL"
    assert good["stated_values"] == {"n": 177.0}
    assert good["recomputed_values"] == {"n": 177.0}
    assert good["passed"] is True
    assert good["max_relative_error"] == 0.0
    assert good["created_at"] is not None

    # Rejected claim must be persisted too, not discarded.
    bad = by_text["Column fare has {n} nulls"]
    assert bad["sql"] == "SELECT COUNT(*) AS n FROM t WHERE fare IS NULL"
    assert bad["stated_values"] == {"n": 5.0}
    assert bad["recomputed_values"] == {"n": 0.0}
    assert bad["passed"] is False
    assert bad["max_relative_error"] == 1.0
    assert bad["created_at"] is not None


def test_save_finding_and_get_findings_roundtrip(tmp_path: Path) -> None:
    engine = _engine(tmp_path)

    run_id = save_run(engine, dataset_name="titanic.csv", model_name="qwen2.5-coder:14b")
    verified = VerifiedClaim(
        claim=Claim(
            text="Cabin missingness correlates with survival",
            sql="SELECT 1",
            stated_values={"x": 1.0},
        ),
        recomputed_values={"x": 1.0},
        passed=True,
        max_relative_error=0.0,
    )
    claim_id = save_verified_claim(engine, run_id, verified)

    finding_with_stats = Finding(
        kind="missingness",
        severity="high",
        columns=["cabin", "survived"],
        effect_size=0.42,
        p_value=0.001,
        p_value_corrected=0.005,
        claim=verified,
    )
    finding_without_stats = Finding(
        kind="duplicate_key",
        severity="medium",
        columns=["passenger_id"],
        effect_size=None,
        p_value=None,
        p_value_corrected=None,
        claim=verified,
    )

    id_with_stats = save_finding(engine, run_id, claim_id, finding_with_stats)
    id_without_stats = save_finding(engine, run_id, claim_id, finding_without_stats)
    assert isinstance(id_with_stats, int)
    assert isinstance(id_without_stats, int)

    rows = get_findings(engine, run_id)
    assert len(rows) == 2

    by_kind = {row["kind"]: row for row in rows}

    with_stats = by_kind["missingness"]
    assert with_stats["severity"] == "high"
    assert with_stats["columns"] == ["cabin", "survived"]
    assert with_stats["effect_size"] == 0.42
    assert with_stats["p_value"] == 0.001
    assert with_stats["p_value_corrected"] == 0.005
    assert with_stats["claim_id"] == claim_id
    assert with_stats["created_at"] is not None

    without_stats = by_kind["duplicate_key"]
    assert without_stats["severity"] == "medium"
    assert without_stats["columns"] == ["passenger_id"]
    assert without_stats["effect_size"] is None
    assert without_stats["p_value"] is None
    assert without_stats["p_value_corrected"] is None
    assert without_stats["claim_id"] == claim_id
