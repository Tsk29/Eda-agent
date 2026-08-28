from __future__ import annotations

import time

import duckdb
import pytest

from eda_agent.executor.execute import execute_sql


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(path)
    con.execute(
        "CREATE TABLE t AS SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) AS t(id, label)"
    )
    con.close()
    return path


def test_successful_query_returns_columns_and_rows(db_path):
    result = execute_sql(db_path, "SELECT id, label FROM t ORDER BY id")

    assert result.ok is True
    assert result.error is None
    assert result.columns == ["id", "label"]
    assert result.rows == [
        {"id": 1, "label": "a"},
        {"id": 2, "label": "b"},
        {"id": 3, "label": "c"},
    ]


def test_sql_syntax_error_returns_sql_error(db_path):
    result = execute_sql(db_path, "SELECT * FORM t")

    assert result.ok is False
    assert result.columns is None
    assert result.rows is None
    assert result.error is not None
    assert result.error.kind == "sql_error"


def test_unknown_table_returns_sql_error(db_path):
    result = execute_sql(db_path, "SELECT * FROM nonexistent_table")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind == "sql_error"


def test_write_statement_against_read_only_connection_is_rejected(db_path):
    result = execute_sql(db_path, "INSERT INTO t VALUES (4, 'd')")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind == "disallowed"

    # Confirm nothing was actually written.
    con = duckdb.connect(db_path, read_only=True)
    count = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    con.close()
    assert count == 3


def test_create_table_against_read_only_connection_is_rejected(db_path):
    result = execute_sql(db_path, "CREATE TABLE new_t AS SELECT * FROM t")

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind == "disallowed"


def test_timeout_is_enforced(db_path):
    slow_sql = """
        WITH RECURSIVE t(n) AS (
            SELECT 1
            UNION ALL
            SELECT n + 1 FROM t WHERE n < 100000000
        )
        SELECT count(*) FROM t
    """

    start = time.monotonic()
    result = execute_sql(db_path, slow_sql, timeout_seconds=1.5)
    elapsed = time.monotonic() - start

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind == "timeout"
    # Give generous headroom for process spawn + terminate overhead, but this
    # must not be allowed to run anywhere near query completion time.
    assert elapsed < 10.0


def test_memory_limit_pragma_is_applied(db_path):
    result = execute_sql(
        db_path,
        "SELECT current_setting('memory_limit') AS lim",
        memory_limit_mb=256,
    )

    assert result.ok is True
    assert result.rows is not None
    # DuckDB reports the limit back in its own human-readable units; just
    # confirm the pragma took effect rather than the default (unset) value.
    assert "256" in result.rows[0]["lim"] or "244" in result.rows[0]["lim"]


def test_low_memory_limit_triggers_memory_exceeded(tmp_path):
    path = str(tmp_path / "big.duckdb")
    con = duckdb.connect(path)
    con.execute("CREATE TABLE big AS SELECT * FROM range(2000000) AS t(x)")
    con.close()

    result = execute_sql(
        path,
        "SELECT a.x, b.x, c.x FROM big a, big b, big c",
        memory_limit_mb=1,
        timeout_seconds=15.0,
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.kind in ("memory_exceeded", "timeout")


def test_never_raises_on_nonexistent_db_path(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.duckdb")

    result = execute_sql(missing_path, "SELECT 1")

    assert result.ok is False
    assert result.error is not None
