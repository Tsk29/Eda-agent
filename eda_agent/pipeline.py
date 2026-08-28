"""End-to-end wiring: profile -> plan -> verify -> store.

Every piece this module touches (profiler, executor, LLM client, planner,
verifier, storage) was built independently against frozen structural
contracts. Nothing here introduces new decision logic -- it only adapts
those pieces to each other:

- `make_run_sql` adapts the sandboxed executor's `ExecutionResult` (which
  never raises) into the raising `Callable[[str], list[dict]]` shape the
  verifier expects, so an executor failure correctly rejects the claim
  wholesale rather than crashing the pipeline.
- `profile_database` opens the on-disk DuckDB file read-only for profiling
  (a plain connection, not the sandboxed subprocess -- profiling only reads
  metadata/aggregates the profiler itself issues, never LLM-generated SQL).
- `run_pipeline` / `run_pipeline_and_store` wire profiler -> planner
  (linear or LangGraph) -> verifier -> storage.

Deliberately NOT wired here: turning guardrail results (leakage, sentinels,
subgroup reversal, BH correction) into stored `Finding` rows. That mapping
-- which guardrail check produces which `Finding.kind`/`severity`, and how
it's tied to the `VerifiedClaim` that surfaced it -- is a real design
decision belonging to Stage 5 and deserves its own plan rather than being
bolted on during reconciliation. `eda_agent.guardrails` is fully built and
tested; only its integration into this pipeline is left open.
"""

from __future__ import annotations

from collections.abc import Callable

import duckdb

from eda_agent.executor.execute import execute_sql
from eda_agent.llm.client import OpenAICompatibleClient
from eda_agent.llm.config import LLMConfig
from eda_agent.log import get_logger
from eda_agent.planner.graph import run_graph
from eda_agent.planner.linear import run_linear
from eda_agent.planner.protocols import LLMClient
from eda_agent.profiler.profile import profile_table
from eda_agent.profiler.schemas import TableProfile
from eda_agent.schemas import Claim, VerifiedClaim
from eda_agent.storage.db import save_run, save_verified_claim
from eda_agent.verifier.verify import verify_claim

logger = get_logger(__name__)


def make_run_sql(db_path: str) -> Callable[[str], list[dict]]:
    """Adapt the sandboxed executor to the shape `verify_claim` expects.

    `execute_sql` never raises -- it returns a structured `ExecutionResult`.
    `verify_claim` expects a callable that raises on failure so its own
    try/except rejects the claim wholesale. This adapter is the seam
    between those two conventions.
    """

    def run_sql(sql: str) -> list[dict]:
        result = execute_sql(db_path, sql)
        if not result.ok:
            error = result.error
            message = f"{error.kind}: {error.message}" if error is not None else "unknown error"
            raise RuntimeError(message)
        return result.rows or []

    return run_sql


def profile_database(db_path: str, table: str) -> TableProfile:
    """Profile `table` in the DuckDB file at `db_path`."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        return profile_table(con, table)
    finally:
        con.close()


def run_pipeline(
    db_path: str,
    table: str,
    llm: LLMClient | None = None,
    use_graph: bool = True,
    max_retries: int = 1,
) -> list[VerifiedClaim]:
    """Profile -> plan -> verify. Returns every VerifiedClaim, rejected or not.

    `llm` defaults to `OpenAICompatibleClient(LLMConfig())`, which reads its
    provider/base_url/model from environment (see `.env.example`).
    """
    profile = profile_database(db_path, table)
    resolved_llm = llm or OpenAICompatibleClient(LLMConfig())
    run_sql = make_run_sql(db_path)

    def verify(claim: Claim) -> VerifiedClaim:
        return verify_claim(claim, run_sql)

    logger.info("run_pipeline: profiled %s.%s (%d rows)", db_path, table, profile.row_count)

    if use_graph:
        return run_graph(profile, resolved_llm, verify, max_retries=max_retries)
    return run_linear(profile, resolved_llm, verify)


def run_pipeline_and_store(
    dsn: str,
    db_path: str,
    table: str,
    dataset_name: str,
    llm: LLMClient | None = None,
    use_graph: bool = True,
    max_retries: int = 1,
) -> tuple[int, list[VerifiedClaim]]:
    """Run the pipeline and persist every claim (passed or rejected).

    Returns the new run id and the verified claims. Storage schema must
    already exist (see `eda_agent.storage.db.create_all`).
    """
    from eda_agent.storage.db import get_engine

    engine = get_engine(dsn)
    resolved_llm = llm or OpenAICompatibleClient(LLMConfig())
    model_name = getattr(getattr(resolved_llm, "_config", None), "model", "unknown")

    run_id = save_run(engine, dataset_name=dataset_name, model_name=model_name)
    verified_claims = run_pipeline(
        db_path, table, llm=resolved_llm, use_graph=use_graph, max_retries=max_retries
    )
    for verified in verified_claims:
        save_verified_claim(engine, run_id, verified)

    logger.info(
        "run_pipeline_and_store: run %d stored %d claim(s)", run_id, len(verified_claims)
    )
    return run_id, verified_claims
