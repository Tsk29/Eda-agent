from __future__ import annotations

import multiprocessing as mp
import queue as queue_module
from typing import Any

import duckdb

from eda_agent.executor.schemas import ExecutionError, ExecutionResult
from eda_agent.log import get_logger

logger = get_logger(__name__)

# Best-effort OS-level memory backstop: how much headroom (as a multiple of
# the DuckDB memory_limit pragma) we grant the RLIMIT_AS cap. DuckDB's own
# pragma is the primary defense and is what most queries will hit first; the
# rlimit exists only to bound the worst case where DuckDB's own accounting is
# bypassed (e.g. non-DuckDB allocations, extension code). The multiplier
# leaves room for Python interpreter startup + DuckDB baseline RSS, which
# would otherwise get killed before the query even starts.
_RLIMIT_HEADROOM_MULTIPLIER = 4

# How long to wait for a terminated/killed subprocess to actually exit before
# giving up and returning a timeout result anyway.
_TERMINATE_GRACE_SECONDS = 2.0

# How long to wait on the result queue after the child process has already
# exited. In normal operation the result is already flushed to the pipe by
# the time join() returns, so this is just a small safety margin.
_QUEUE_DRAIN_SECONDS = 2.0


def _apply_best_effort_memory_rlimit(memory_limit_mb: int) -> None:
    """Best-effort OS-level address-space cap for the child process.

    This is a backstop, not the primary defense — DuckDB's own memory_limit
    pragma (set on the connection) is what actually bounds query memory in
    practice. RLIMIT_AS enforcement is platform-dependent: on Linux it caps
    virtual address space fairly reliably; on macOS it is much weaker (the
    kernel does not consistently enforce RLIMIT_AS the way Linux does), and
    the resource module is unavailable on Windows entirely. We therefore
    treat any failure here as non-fatal and keep going with just the DuckDB
    pragma in effect.
    """
    try:
        import resource
    except ImportError:
        return

    try:
        limit_bytes = memory_limit_mb * 1024 * 1024 * _RLIMIT_HEADROOM_MULTIPLIER
        resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except (ValueError, OSError):
        # Platform doesn't support this rlimit, or the hard limit already
        # forbids raising it. Fall back to relying on the DuckDB pragma alone.
        return


def _classify_duckdb_error(exc: duckdb.Error) -> str:
    if isinstance(exc, duckdb.OutOfMemoryException):
        return "memory_exceeded"
    if "read-only" in str(exc).lower():
        return "disallowed"
    return "sql_error"


def _run_query(
    db_path: str,
    sql: str,
    memory_limit_mb: int,
    read_only: bool,
    result_queue: mp.Queue,
) -> None:
    """Entry point for the subprocess. Never raises; always enqueues a dict."""
    _apply_best_effort_memory_rlimit(memory_limit_mb)

    con: duckdb.DuckDBPyConnection | None = None
    try:
        con = duckdb.connect(db_path, read_only=read_only)
        # No network: never reach out to pull an extension.
        con.execute("SET autoinstall_known_extensions=false")
        con.execute("SET autoload_known_extensions=false")
        con.execute(f"SET memory_limit='{memory_limit_mb}MB'")

        cursor = con.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]

        result_queue.put({"ok": True, "columns": columns, "rows": rows, "error": None})
    except duckdb.Error as exc:
        kind = _classify_duckdb_error(exc)
        result_queue.put(
            {
                "ok": False,
                "columns": None,
                "rows": None,
                "error": {"kind": kind, "message": str(exc)},
            }
        )
    except Exception as exc:  # noqa: BLE001 - last-resort guard, must never crash silently
        result_queue.put(
            {
                "ok": False,
                "columns": None,
                "rows": None,
                "error": {"kind": "unknown", "message": f"{type(exc).__name__}: {exc}"},
            }
        )
    finally:
        if con is not None:
            con.close()


def _terminate(process: mp.Process) -> None:
    process.terminate()
    process.join(_TERMINATE_GRACE_SECONDS)
    if process.is_alive():
        process.kill()
        process.join(_TERMINATE_GRACE_SECONDS)


def _result_from_dict(payload: dict[str, Any]) -> ExecutionResult:
    error = ExecutionError(**payload["error"]) if payload["error"] is not None else None
    return ExecutionResult(
        ok=payload["ok"], columns=payload["columns"], rows=payload["rows"], error=error
    )


def execute_sql(
    db_path: str,
    sql: str,
    timeout_seconds: float = 10.0,
    memory_limit_mb: int = 512,
    read_only: bool = True,
) -> ExecutionResult:
    """Run `sql` against the DuckDB database at `db_path` in a subprocess.

    Every query this project runs is a read-only analysis query, so
    `read_only=True` is the default; it is kept as a parameter for a future
    caller that legitimately needs to write. The query always runs in its own
    process (never in-process, per project rule) so a hung or memory-hungry
    query cannot take down or block the caller. This function never raises:
    every failure mode is captured and returned as a structured
    ExecutionResult.
    """
    try:
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()
        process = ctx.Process(
            target=_run_query,
            args=(db_path, sql, memory_limit_mb, read_only, result_queue),
            daemon=True,
        )
        process.start()
        process.join(timeout_seconds)

        if process.is_alive():
            _terminate(process)
            return ExecutionResult(
                ok=False,
                columns=None,
                rows=None,
                error=ExecutionError(
                    kind="timeout",
                    message=f"Query did not complete within {timeout_seconds}s and was terminated.",
                ),
            )

        try:
            payload = result_queue.get(timeout=_QUEUE_DRAIN_SECONDS)
        except queue_module.Empty:
            exitcode = process.exitcode
            return ExecutionResult(
                ok=False,
                columns=None,
                rows=None,
                error=ExecutionError(
                    kind="unknown",
                    message=f"Subprocess exited (code={exitcode}) without returning a result.",
                ),
            )

        return _result_from_dict(payload)
    except Exception as exc:  # noqa: BLE001 - caller must always get a structured result
        logger.exception("execute_sql failed unexpectedly")
        return ExecutionResult(
            ok=False,
            columns=None,
            rows=None,
            error=ExecutionError(kind="unknown", message=f"{type(exc).__name__}: {exc}"),
        )
