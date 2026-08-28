"""SQLAlchemy 2.x Core storage layer for runs, claims, and findings.

Production target is Postgres 16 (see docker-compose.yml / .env.example).
All access goes through explicit Core `Table` definitions and `insert()` /
`select()` statements -- no ORM declarative classes.

Per the project's non-negotiable rules, rejected claims (``passed=False``)
are persisted exactly like accepted ones. Nothing here silently drops or
repairs a claim.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
)

from eda_agent.log import get_logger
from eda_agent.schemas import Finding, VerifiedClaim

logger = get_logger(__name__)

metadata = MetaData()

runs = Table(
    "runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("dataset_name", String, nullable=False),
    Column("model_name", String, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True), nullable=True),
)

claims = Table(
    "claims",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, ForeignKey("runs.id"), nullable=False),
    Column("text", String, nullable=False),
    Column("sql", String, nullable=False),
    Column("stated_values", JSON, nullable=False),
    Column("recomputed_values", JSON, nullable=False),
    Column("passed", Boolean, nullable=False),
    Column("max_relative_error", Float, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

findings = Table(
    "findings",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, ForeignKey("runs.id"), nullable=False),
    Column("claim_id", Integer, ForeignKey("claims.id"), nullable=False),
    Column("kind", String, nullable=False),
    Column("severity", String, nullable=False),
    Column("columns", JSON, nullable=False),
    Column("effect_size", Float, nullable=True),
    Column("p_value", Float, nullable=True),
    Column("p_value_corrected", Float, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
)


def get_engine(dsn: str) -> Engine:
    """Create a SQLAlchemy engine for the given DSN.

    Production DSN is Postgres via psycopg3, e.g.
    ``postgresql+psycopg://eda:eda@localhost:5432/eda`` (see .env.example).
    """
    return create_engine(dsn, future=True)


def create_all(engine: Engine) -> None:
    """Create all tables (runs, claims, findings) if they do not exist."""
    metadata.create_all(engine)
    logger.info("storage tables created (or already present)")


def save_run(engine: Engine, dataset_name: str, model_name: str) -> int:
    """Insert a new run row with started_at=now, finished_at=NULL.

    Returns the new run id.
    """
    started_at = datetime.now(UTC)
    with engine.begin() as conn:
        result = conn.execute(
            insert(runs).values(
                dataset_name=dataset_name,
                model_name=model_name,
                started_at=started_at,
                finished_at=None,
            )
        )
        run_id = result.inserted_primary_key[0]
    logger.info("saved run %s (dataset=%s, model=%s)", run_id, dataset_name, model_name)
    return int(run_id)


def save_verified_claim(engine: Engine, run_id: int, verified: VerifiedClaim) -> int:
    """Insert a verified (or rejected) claim tied to a run.

    Stored regardless of ``verified.passed`` -- rejected claims must be
    persisted, not discarded, per the project's verification rules.
    Returns the new claim id.
    """
    created_at = datetime.now(UTC)
    with engine.begin() as conn:
        result = conn.execute(
            insert(claims).values(
                run_id=run_id,
                text=verified.claim.text,
                sql=verified.claim.sql,
                stated_values=verified.claim.stated_values,
                recomputed_values=verified.recomputed_values,
                passed=verified.passed,
                max_relative_error=verified.max_relative_error,
                created_at=created_at,
            )
        )
        claim_id = result.inserted_primary_key[0]
    logger.info("saved claim %s for run %s (passed=%s)", claim_id, run_id, verified.passed)
    return int(claim_id)


def save_finding(engine: Engine, run_id: int, claim_id: int, finding: Finding) -> int:
    """Insert a finding tied to a run and its underlying claim.

    Returns the new finding id.
    """
    created_at = datetime.now(UTC)
    with engine.begin() as conn:
        result = conn.execute(
            insert(findings).values(
                run_id=run_id,
                claim_id=claim_id,
                kind=finding.kind,
                severity=finding.severity,
                columns=finding.columns,
                effect_size=finding.effect_size,
                p_value=finding.p_value,
                p_value_corrected=finding.p_value_corrected,
                created_at=created_at,
            )
        )
        finding_id = result.inserted_primary_key[0]
    logger.info("saved finding %s for run %s (kind=%s)", finding_id, run_id, finding.kind)
    return int(finding_id)


def get_claims(engine: Engine, run_id: int) -> list[dict[str, Any]]:
    """Return all claims for a run as dicts.

    Keys: text, sql, stated_values, recomputed_values, passed,
    max_relative_error, created_at.
    """
    with engine.connect() as conn:
        rows = conn.execute(select(claims).where(claims.c.run_id == run_id)).mappings().all()
    return [
        {
            "text": row["text"],
            "sql": row["sql"],
            "stated_values": row["stated_values"],
            "recomputed_values": row["recomputed_values"],
            "passed": row["passed"],
            "max_relative_error": row["max_relative_error"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def get_findings(engine: Engine, run_id: int) -> list[dict[str, Any]]:
    """Return all findings for a run as dicts.

    Keys: kind, severity, columns, effect_size, p_value, p_value_corrected,
    claim_id, created_at.
    """
    with engine.connect() as conn:
        rows = conn.execute(select(findings).where(findings.c.run_id == run_id)).mappings().all()
    return [
        {
            "kind": row["kind"],
            "severity": row["severity"],
            "columns": row["columns"],
            "effect_size": row["effect_size"],
            "p_value": row["p_value"],
            "p_value_corrected": row["p_value_corrected"],
            "claim_id": row["claim_id"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]
